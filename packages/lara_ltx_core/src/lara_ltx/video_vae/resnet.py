"""Checkpoint-compatible causal 3D ResNet primitives for the Diffusion VAE."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError
from lara_ltx.transformer.timestep import PixArtAlphaCombinedTimestepSizeEmbeddings

PIXEL_NORM_EPSILON = 1e-8
GROUP_NORM_EPSILON = 1e-6
KERNEL_SIZE = 3
SPATIAL_PADDING_ZEROS = "zeros"
SPATIAL_PADDING_REFLECT = "reflect"
CHANNEL_AXIS = 1
TIME_AXIS = 2
CHANNEL_LAST_PERMUTATION = (0, 2, 3, 4, 1)
CHANNEL_FIRST_PERMUTATION = (0, 4, 1, 2, 3)
PYTORCH_TO_MLX_CONV3D_WEIGHT_PERMUTATION = (0, 2, 3, 4, 1)


def pixel_norm(x: mx.array, *, eps: float = PIXEL_NORM_EPSILON) -> mx.array:
    """Apply upstream channel-wise PixelNorm to NCTHW tensors."""

    return x * mx.rsqrt(mx.mean(mx.square(x), axis=CHANNEL_AXIS, keepdims=True) + eps)


class GroupNormNCTHW(nn.Module):
    """GroupNorm for upstream channel-first video tensors and key layout."""

    def __init__(self, channels: int, groups: int, *, eps: float = GROUP_NORM_EPSILON) -> None:
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        batch, channels, time, height, width = x.shape
        channels_per_group = channels // self.groups
        grouped = x.reshape(batch, self.groups, channels_per_group, time, height, width)
        mean = mx.mean(grouped, axis=(2, 3, 4, 5), keepdims=True)
        variance = mx.mean(mx.square(grouped - mean), axis=(2, 3, 4, 5), keepdims=True)
        normalized = (grouped - mean) * mx.rsqrt(variance + self.eps)
        normalized = normalized.reshape(batch, channels, time, height, width)
        return normalized * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]


class Conv3dWeights(nn.Module):
    """Source-layout Conv3d parameters stored under upstream-compatible names."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, bias: bool = True) -> None:
        super().__init__()
        self.weight = mx.zeros((out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None


class CausalConv3d(nn.Module):
    """Upstream first-frame-repeat causal Conv3d with NCTHW inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = KERNEL_SIZE,
        bias: bool = True,
        spatial_padding_mode: str = SPATIAL_PADDING_ZEROS,
    ) -> None:
        super().__init__()
        self.time_kernel_size = kernel_size
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = Conv3dWeights(in_channels, out_channels, kernel_size, bias=bias)

    def __call__(self, x: mx.array, *, causal: bool = True) -> mx.array:
        pad_count = self.time_kernel_size - 1
        first = x[:, :, :1]
        if causal:
            padded = mx.concatenate([first] * pad_count + [x], axis=TIME_AXIS)
        else:
            side_count = pad_count // 2
            last = x[:, :, -1:]
            padded = mx.concatenate([first] * side_count + [x] + [last] * side_count, axis=TIME_AXIS)
        if self.spatial_padding_mode == SPATIAL_PADDING_REFLECT:
            padded = mx.concatenate([padded[:, :, :, 1:2], padded, padded[:, :, :, -2:-1]], axis=3)
            padded = mx.concatenate([padded[:, :, :, :, 1:2], padded, padded[:, :, :, :, -2:-1]], axis=4)
            spatial_padding = (0, 0, 0)
        else:
            spatial_padding = (0, 1, 1)
        channel_last = mx.transpose(padded, CHANNEL_LAST_PERMUTATION)
        weight = mx.transpose(self.conv.weight, PYTORCH_TO_MLX_CONV3D_WEIGHT_PERMUTATION)
        result = mx.conv3d(channel_last, weight, stride=1, padding=spatial_padding)
        if self.conv.bias is not None:
            result = result + self.conv.bias[None, None, None, None, :]
        return mx.transpose(result, CHANNEL_FIRST_PERMUTATION)


class PointwiseConv3d(nn.Module):
    """One-by-one-by-one NCTHW convolution with upstream weight layout."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = mx.zeros((out_channels, in_channels, 1, 1, 1))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        channel_last = mx.transpose(x, CHANNEL_LAST_PERMUTATION)
        weight = mx.transpose(self.weight, PYTORCH_TO_MLX_CONV3D_WEIGHT_PERMUTATION)
        result = mx.conv3d(channel_last, weight)
        return mx.transpose(result + self.bias[None, None, None, None, :], CHANNEL_FIRST_PERMUTATION)


@dataclass(frozen=True)
class ResnetBlock3DConfig:
    """The supported non-stochastic Diffusion VAE ResNet block configuration."""

    in_channels: int
    out_channels: int
    groups: int = 1
    norm_layer: str = "pixel_norm"
    spatial_padding_mode: str = SPATIAL_PADDING_ZEROS
    timestep_conditioning: bool = False


class ResnetBlock3D(nn.Module):
    """PixelNorm/GroupNorm causal 3D ResNet block matching upstream operation order."""

    def __init__(self, config: ResnetBlock3DConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = self._normalization(config.in_channels, config.groups)
        self.conv1 = CausalConv3d(
            config.in_channels, config.out_channels, spatial_padding_mode=config.spatial_padding_mode
        )
        self.norm2 = self._normalization(config.out_channels, config.groups)
        self.conv2 = CausalConv3d(
            config.out_channels, config.out_channels, spatial_padding_mode=config.spatial_padding_mode
        )
        self.conv_shortcut = (
            PointwiseConv3d(config.in_channels, config.out_channels)
            if config.in_channels != config.out_channels
            else None
        )
        self.norm3 = GroupNormNCTHW(config.in_channels, 1) if config.in_channels != config.out_channels else None
        self.scale_shift_table = mx.zeros((4, config.in_channels)) if config.timestep_conditioning else None

    def _normalization(self, channels: int, groups: int) -> GroupNormNCTHW | None:
        if self.config.norm_layer == "pixel_norm":
            return None
        if self.config.norm_layer == "group_norm":
            return GroupNormNCTHW(channels, groups)
        raise LaraError("LARA-TENSOR-009", details={"norm_layer": self.config.norm_layer})

    def _apply_norm(self, x: mx.array, normalization: GroupNormNCTHW | None) -> mx.array:
        return pixel_norm(x) if normalization is None else normalization(x)

    def __call__(self, x: mx.array, *, causal: bool = True, timestep: mx.array | None = None) -> mx.array:
        hidden = self._apply_norm(x, self.norm1)
        shift1 = scale1 = shift2 = scale2 = None
        if self.config.timestep_conditioning:
            if timestep is None or self.scale_shift_table is None:
                raise LaraError("LARA-TENSOR-014", details={"block_name": "ResnetBlock3D"})
            batch = hidden.shape[0]
            spatial_shape = timestep.shape[-3:]
            values = self.scale_shift_table[None, :, :, None, None, None] + timestep.reshape(
                batch, 4, -1, *spatial_shape
            )
            shift1, scale1, shift2, scale2 = mx.split(values, 4, axis=1)
            shift1, scale1, shift2, scale2 = (value[:, 0] for value in (shift1, scale1, shift2, scale2))
            hidden = hidden * (1 + scale1) + shift1
        hidden = hidden * mx.sigmoid(hidden)
        hidden = self.conv1(hidden, causal=causal)
        hidden = self._apply_norm(hidden, self.norm2)
        if scale2 is not None and shift2 is not None:
            hidden = hidden * (1 + scale2) + shift2
        hidden = hidden * mx.sigmoid(hidden)
        hidden = self.conv2(hidden, causal=causal)
        residual = x if self.norm3 is None else self.norm3(x)
        residual = residual if self.conv_shortcut is None else self.conv_shortcut(residual)
        return residual + hidden


@dataclass(frozen=True)
class UNetMidBlock3DConfig:
    """Supported convolutional VAE mid-block configuration."""

    in_channels: int
    num_layers: int
    groups: int = 1
    norm_layer: str = "pixel_norm"
    timestep_conditioning: bool = False
    spatial_padding_mode: str = SPATIAL_PADDING_ZEROS


class UNetMidBlock3D(nn.Module):
    """Sequence of same-channel VAE ResNet blocks with optional timestep embedding."""

    def __init__(self, config: UNetMidBlock3DConfig) -> None:
        super().__init__()
        self.config = config
        self.time_embedder = (
            PixArtAlphaCombinedTimestepSizeEmbeddings(config.in_channels * 4) if config.timestep_conditioning else None
        )
        self.res_blocks = [
            ResnetBlock3D(
                ResnetBlock3DConfig(
                    in_channels=config.in_channels,
                    out_channels=config.in_channels,
                    groups=config.groups,
                    norm_layer=config.norm_layer,
                    spatial_padding_mode=config.spatial_padding_mode,
                    timestep_conditioning=config.timestep_conditioning,
                )
            )
            for _ in range(config.num_layers)
        ]

    def __call__(self, hidden_states: mx.array, *, causal: bool = True, timestep: mx.array | None = None) -> mx.array:
        timestep_embedding = None
        if self.config.timestep_conditioning:
            if timestep is None or self.time_embedder is None:
                raise LaraError("LARA-TENSOR-014", details={"block_name": "UNetMidBlock3D"})
            timestep_embedding = self.time_embedder(timestep.reshape(-1), hidden_dtype=hidden_states.dtype)
            timestep_embedding = timestep_embedding.reshape(hidden_states.shape[0], -1, 1, 1, 1)
        for block in self.res_blocks:
            hidden_states = block(hidden_states, causal=causal, timestep=timestep_embedding)
        return hidden_states
