"""Checkpoint-compatible LTX-2.5 x2 latent spatial upscaler for MLX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError
from lara_ltx.models.loading import iter_component_weight_batches, load_safetensors_shard
from lara_ltx.models.spatial_upscaler import (
    KERNEL_SIZE,
    LATENT_CHANNELS,
    RESIDUAL_BLOCK_COUNT,
    UPSAMPLER_CHANNELS,
    spatial_upscaler_target_shapes,
    validate_spatial_upscaler_mapping,
)

from .ops import PerChannelStatistics
from .resnet import (
    CHANNEL_FIRST_PERMUTATION,
    CHANNEL_LAST_PERMUTATION,
    PYTORCH_TO_MLX_CONV3D_WEIGHT_PERMUTATION,
)

GROUP_COUNT = 32
GROUP_NORM_EPSILON = 1e-5
SPATIAL_SCALE = 2
VAE_MEAN_SOURCE_KEY = "per_channel_statistics.mean-of-means"
VAE_STD_SOURCE_KEY = "per_channel_statistics.std-of-means"


def _silu(value: mx.array) -> mx.array:
    return nn.silu(value.astype(mx.float32)).astype(value.dtype)


class SymmetricConv3d(nn.Module):
    """PyTorch-layout Conv3d with symmetric zero padding on NCTHW input."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = KERNEL_SIZE) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, value: mx.array) -> mx.array:
        channel_last = mx.transpose(value, CHANNEL_LAST_PERMUTATION)
        weight = mx.transpose(self.weight, PYTORCH_TO_MLX_CONV3D_WEIGHT_PERMUTATION)
        result = mx.conv3d(channel_last, weight, stride=1, padding=self.kernel_size // 2)
        result = result + self.bias[None, None, None, None, :]
        return mx.transpose(result, CHANNEL_FIRST_PERMUTATION)


class FramewiseConv2d(nn.Module):
    """PyTorch-layout Conv2d applied independently to every video frame."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = KERNEL_SIZE) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, value: mx.array) -> mx.array:
        batch, _, frames, height, width = value.shape
        frame_batch = mx.transpose(value, (0, 2, 3, 4, 1)).reshape(batch * frames, height, width, -1)
        weight = mx.transpose(self.weight, (0, 2, 3, 1))
        result = mx.conv2d(frame_batch, weight, stride=1, padding=self.kernel_size // 2)
        result = result + self.bias[None, None, None, :]
        return mx.transpose(result.reshape(batch, frames, height, width, -1), (0, 4, 1, 2, 3))


class UpscalerGroupNorm(nn.Module):
    """PyTorch GroupNorm semantics with float32 accumulation and BF16 output."""

    def __init__(self, channels: int, groups: int = GROUP_COUNT) -> None:
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, value: mx.array) -> mx.array:
        batch, channels, frames, height, width = value.shape
        input_dtype = value.dtype
        grouped = value.astype(mx.float32).reshape(
            batch,
            self.groups,
            channels // self.groups,
            frames,
            height,
            width,
        )
        mean = mx.mean(grouped, axis=(2, 3, 4, 5), keepdims=True)
        variance = mx.var(grouped, axis=(2, 3, 4, 5), keepdims=True)
        normalized = (grouped - mean) * mx.rsqrt(variance + GROUP_NORM_EPSILON)
        normalized = normalized.reshape(batch, channels, frames, height, width)
        affine = normalized * self.weight.astype(mx.float32)[None, :, None, None, None]
        affine = affine + self.bias.astype(mx.float32)[None, :, None, None, None]
        return affine.astype(input_dtype)


class SpatialPixelShuffle(nn.Module):
    """Move two channel factors into height and width, matching PixelShuffleND(2)."""

    def __init__(self, scale: int = SPATIAL_SCALE) -> None:
        super().__init__()
        self.scale = scale

    def __call__(self, value: mx.array) -> mx.array:
        batch, channels, frames, height, width = value.shape
        expansion = self.scale**2
        if channels % expansion:
            raise LaraError("LARA-TENSOR-023", details={"shape": tuple(value.shape)})
        output_channels = channels // expansion
        shuffled = value.reshape(batch, output_channels, self.scale, self.scale, frames, height, width)
        return mx.transpose(shuffled, (0, 1, 4, 5, 2, 6, 3)).reshape(
            batch,
            output_channels,
            frames,
            height * self.scale,
            width * self.scale,
        )


class SpatialUpscalerResBlock(nn.Module):
    """The official Conv3d/GroupNorm/SiLU residual block."""

    def __init__(self, channels: int, *, groups: int = GROUP_COUNT) -> None:
        super().__init__()
        self.conv1 = SymmetricConv3d(channels, channels)
        self.norm1 = UpscalerGroupNorm(channels, groups)
        self.conv2 = SymmetricConv3d(channels, channels)
        self.norm2 = UpscalerGroupNorm(channels, groups)

    def __call__(self, value: mx.array) -> mx.array:
        hidden = _silu(self.norm1(self.conv1(value)))
        hidden = self.norm2(self.conv2(hidden))
        return _silu(hidden + value)


@dataclass(frozen=True)
class LatentSpatialUpscalerConfig:
    in_channels: int = LATENT_CHANNELS
    mid_channels: int = UPSAMPLER_CHANNELS
    num_blocks_per_stage: int = RESIDUAL_BLOCK_COUNT
    groups: int = GROUP_COUNT
    spatial_scale: int = SPATIAL_SCALE

    def __post_init__(self) -> None:
        if (
            self.in_channels <= 0
            or self.mid_channels <= 0
            or self.num_blocks_per_stage <= 0
            or self.groups <= 0
            or self.mid_channels % self.groups
            or self.spatial_scale <= 0
        ):
            raise LaraError("LARA-TENSOR-023", details={"shape": "invalid_configuration"})


class LatentSpatialUpscaler(nn.Module):
    """Official 3D residual x2 latent upscaler with source-compatible keys."""

    def __init__(self, config: LatentSpatialUpscalerConfig | None = None) -> None:
        super().__init__()
        config = config or LatentSpatialUpscalerConfig()
        self.config = config
        self.initial_conv = SymmetricConv3d(config.in_channels, config.mid_channels)
        self.initial_norm = UpscalerGroupNorm(config.mid_channels, config.groups)
        self.res_blocks = [
            SpatialUpscalerResBlock(config.mid_channels, groups=config.groups)
            for _ in range(config.num_blocks_per_stage)
        ]
        self.upsampler = [
            FramewiseConv2d(
                config.mid_channels,
                config.mid_channels * config.spatial_scale**2,
            )
        ]
        self.pixel_shuffle = SpatialPixelShuffle(config.spatial_scale)
        self.post_upsample_res_blocks = [
            SpatialUpscalerResBlock(config.mid_channels, groups=config.groups)
            for _ in range(config.num_blocks_per_stage)
        ]
        self.final_conv = SymmetricConv3d(config.mid_channels, config.in_channels)

    def __call__(self, latent: mx.array) -> mx.array:
        if latent.ndim != 5 or latent.shape[1] != self.config.in_channels:
            raise LaraError("LARA-TENSOR-023", details={"shape": tuple(latent.shape)})
        hidden = _silu(self.initial_norm(self.initial_conv(latent)))
        for block in self.res_blocks:
            hidden = block(hidden)
        hidden = self.pixel_shuffle(self.upsampler[0](hidden))
        for block in self.post_upsample_res_blocks:
            hidden = block(hidden)
        return self.final_conv(hidden)


class SpatialVideoUpscaler(nn.Module):
    """Normalize around the official x2 network as the CUDA pipeline does."""

    def __init__(self, upscaler: LatentSpatialUpscaler | None = None) -> None:
        super().__init__()
        self.upscaler = upscaler or LatentSpatialUpscaler()
        self.per_channel_statistics = PerChannelStatistics(self.upscaler.config.in_channels)

    def __call__(self, latent: mx.array) -> mx.array:
        unnormalized = self.per_channel_statistics.un_normalize(latent)
        upscaled = self.upscaler(unnormalized)
        return self.per_channel_statistics.normalize(upscaled)


def load_spatial_video_upscaler(
    *,
    upscaler_checkpoint: Path,
    video_vae_checkpoint: Path,
    mapping: dict[str, object],
) -> SpatialVideoUpscaler:
    """Strict-load all 72 upscaler tensors and the two VAE statistics tensors."""

    rules = validate_spatial_upscaler_mapping(mapping)
    owner = SpatialVideoUpscaler()
    for batch in iter_component_weight_batches(
        (upscaler_checkpoint,),
        rules,
        expected_target_shapes=spatial_upscaler_target_shapes(),
    ):
        owner.upscaler.load_weights(batch)
        mx.eval(*[value for _, value in batch])

    statistics = load_safetensors_shard(
        video_vae_checkpoint,
        required_names=(VAE_MEAN_SOURCE_KEY, VAE_STD_SOURCE_KEY),
    )
    mean = statistics.tensors[VAE_MEAN_SOURCE_KEY]
    std = statistics.tensors[VAE_STD_SOURCE_KEY]
    expected_shape = (owner.upscaler.config.in_channels,)
    if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
        raise LaraError("LARA-MODEL-030", details={"key": "per_channel_statistics"})
    owner.per_channel_statistics.load_weights(
        (
            ("mean_of_means", mean),
            ("std_of_means", std),
        )
    )
    mx.eval(mean, std)
    return owner
