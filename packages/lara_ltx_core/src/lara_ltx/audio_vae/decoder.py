"""Native MLX decoder for the official LTX-2.5 causal Audio VAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError
from lara_ltx.models.audio_vae import (
    BASE_CHANNELS,
    CHANNEL_MULTIPLIERS,
    DECODER_RESIDUAL_BLOCK_COUNT,
    KERNEL_SIZE,
    LATENT_CHANNELS,
    audio_vae_target_shapes,
    validate_audio_vae_mapping,
)
from lara_ltx.models.loading import iter_component_weight_batches

AUDIO_OUTPUT_CHANNELS = 2
LATENT_DOWNSAMPLE_FACTOR = 4
PIXEL_NORM_EPSILON = 1e-6
SPATIAL_SCALE = 2
CHANNEL_AXIS = 1
CHANNEL_LAST_PERMUTATION = (0, 2, 3, 1)
CHANNEL_FIRST_PERMUTATION = (0, 3, 1, 2)
PYTORCH_TO_MLX_CONV2D_WEIGHT_PERMUTATION = (0, 2, 3, 1)
DECODER_PREFIX = "decoder."
STATISTICS_PREFIX = "per_channel_statistics."


def _silu(value: mx.array) -> mx.array:
    return nn.silu(value.astype(mx.float32)).astype(value.dtype)


def pixel_norm(value: mx.array) -> mx.array:
    return value * mx.rsqrt(mx.mean(mx.square(value), axis=CHANNEL_AXIS, keepdims=True) + PIXEL_NORM_EPSILON)


class Conv2dWeights(nn.Module):
    """Source-layout Conv2d parameters nested below the upstream ``conv`` key."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = mx.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = mx.zeros((out_channels,))


class HeightCausalConv2d(nn.Module):
    """Audio time-axis causal Conv2d with symmetric frequency padding."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = KERNEL_SIZE) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = Conv2dWeights(in_channels, out_channels, kernel_size)

    def __call__(self, value: mx.array) -> mx.array:
        if value.ndim != 4:
            raise LaraError("LARA-TENSOR-024", details={"shape": tuple(value.shape)})
        padding = self.kernel_size - 1
        if padding:
            frequency_before = padding // 2
            frequency_after = padding - frequency_before
            value = mx.pad(
                value,
                ((0, 0), (0, 0), (padding, 0), (frequency_before, frequency_after)),
            )
        channel_last = mx.transpose(value, CHANNEL_LAST_PERMUTATION)
        weight = mx.transpose(self.conv.weight, PYTORCH_TO_MLX_CONV2D_WEIGHT_PERMUTATION)
        result = mx.conv2d(channel_last, weight)
        result = result + self.conv.bias[None, None, None, :]
        return mx.transpose(result, CHANNEL_FIRST_PERMUTATION)


class AudioResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = HeightCausalConv2d(in_channels, out_channels)
        self.conv2 = HeightCausalConv2d(out_channels, out_channels)
        self.nin_shortcut = (
            HeightCausalConv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        )

    def __call__(self, value: mx.array) -> mx.array:
        hidden = self.conv1(_silu(pixel_norm(value)))
        hidden = self.conv2(_silu(pixel_norm(hidden)))
        residual = self.nin_shortcut(value) if self.nin_shortcut is not None else value
        return residual + hidden


class AudioMidBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block_1 = AudioResBlock(channels, channels)
        self.block_2 = AudioResBlock(channels, channels)

    def __call__(self, value: mx.array) -> mx.array:
        return self.block_2(self.block_1(value))


class AudioUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = HeightCausalConv2d(channels, channels)

    def __call__(self, value: mx.array) -> mx.array:
        value = mx.repeat(mx.repeat(value, SPATIAL_SCALE, axis=2), SPATIAL_SCALE, axis=3)
        value = self.conv(value)
        return value[:, :, 1:, :]


class AudioDecoderStage(nn.Module):
    def __init__(self, blocks: list[AudioResBlock], upsample: AudioUpsample | None) -> None:
        super().__init__()
        self.block = blocks
        self.upsample = upsample


class AudioPerChannelStatistics(nn.Module):
    def __init__(self, feature_width: int) -> None:
        super().__init__()
        self.mean_of_means = mx.zeros((feature_width,))
        self.std_of_means = mx.ones((feature_width,))

    def un_normalize(self, latent: mx.array) -> mx.array:
        batch, channels, frames, mel_bins = latent.shape
        flattened = mx.transpose(latent, (0, 2, 1, 3)).reshape(batch, frames, channels * mel_bins)
        if flattened.shape[-1] != self.mean_of_means.shape[0]:
            raise LaraError("LARA-TENSOR-024", details={"shape": tuple(latent.shape)})
        flattened = flattened * self.std_of_means + self.mean_of_means
        return mx.transpose(flattened.reshape(batch, frames, channels, mel_bins), (0, 2, 1, 3))


@dataclass(frozen=True)
class AudioVAEDecoderConfig:
    base_channels: int = BASE_CHANNELS
    channel_multipliers: tuple[int, ...] = CHANNEL_MULTIPLIERS
    latent_channels: int = LATENT_CHANNELS
    output_channels: int = AUDIO_OUTPUT_CHANNELS
    residual_blocks_per_level: int = DECODER_RESIDUAL_BLOCK_COUNT
    latent_downsample_factor: int = LATENT_DOWNSAMPLE_FACTOR
    output_mel_bins: int = 64

    def __post_init__(self) -> None:
        if (
            self.base_channels <= 0
            or not self.channel_multipliers
            or any(value <= 0 for value in self.channel_multipliers)
            or self.latent_channels <= 0
            or self.output_channels <= 0
            or self.residual_blocks_per_level <= 0
            or self.latent_downsample_factor <= 0
            or self.output_mel_bins <= 0
        ):
            raise LaraError("LARA-TENSOR-024", details={"shape": "invalid_configuration"})


class AudioVAEDecoder(nn.Module):
    """Attention-free pixel-normalized causal decoder from the pinned config."""

    def __init__(self, config: AudioVAEDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioVAEDecoderConfig()
        base_block_channels = self.config.base_channels * self.config.channel_multipliers[-1]
        self.per_channel_statistics = AudioPerChannelStatistics(
            self.config.latent_channels * (self.config.output_mel_bins // self.config.latent_downsample_factor)
        )
        self.conv_in = HeightCausalConv2d(self.config.latent_channels, base_block_channels)
        self.mid = AudioMidBlock(base_block_channels)
        self.up = self._build_stages(base_block_channels)
        self.conv_out = HeightCausalConv2d(self.config.base_channels, self.config.output_channels)

    def _build_stages(self, initial_channels: int) -> list[AudioDecoderStage]:
        stages: list[AudioDecoderStage] = []
        current_channels = initial_channels
        for level in reversed(range(len(self.config.channel_multipliers))):
            output_channels = self.config.base_channels * self.config.channel_multipliers[level]
            blocks = []
            for _ in range(self.config.residual_blocks_per_level):
                blocks.append(AudioResBlock(current_channels, output_channels))
                current_channels = output_channels
            upsample = AudioUpsample(current_channels) if level > 0 else None
            stages.insert(0, AudioDecoderStage(blocks, upsample))
        return stages

    def __call__(self, latent: mx.array) -> mx.array:
        if latent.ndim != 4 or latent.shape[1] != self.config.latent_channels:
            raise LaraError("LARA-TENSOR-024", details={"shape": tuple(latent.shape)})
        target_frames = max(
            latent.shape[2] * self.config.latent_downsample_factor - (self.config.latent_downsample_factor - 1),
            1,
        )
        hidden = self.conv_in(self.per_channel_statistics.un_normalize(latent))
        hidden = self.mid(hidden)
        for level in reversed(range(len(self.up))):
            stage = self.up[level]
            for block in stage.block:
                hidden = block(hidden)
            if stage.upsample is not None:
                hidden = stage.upsample(hidden)
        hidden = self.conv_out(_silu(pixel_norm(hidden)))
        return hidden[:, : self.config.output_channels, :target_frames, : self.config.output_mel_bins]


def _decoder_rules(mapping: dict[str, object]) -> tuple[tuple[dict[str, object], ...], dict[str, tuple[int, ...]]]:
    rules = validate_audio_vae_mapping(mapping)
    source_shapes = audio_vae_target_shapes()
    selected: list[dict[str, object]] = []
    target_shapes: dict[str, tuple[int, ...]] = {}
    for rule in rules:
        target_key = str(rule["target_key"])
        if target_key.startswith(DECODER_PREFIX):
            runtime_key = target_key.removeprefix(DECODER_PREFIX)
        elif target_key.startswith(STATISTICS_PREFIX):
            runtime_key = target_key.replace("-", "_")
        else:
            continue
        selected.append({**rule, "target_key": runtime_key})
        target_shapes[runtime_key] = source_shapes[target_key]
    return tuple(selected), target_shapes


def load_audio_vae_decoder(*, checkpoint: Path, mapping: dict[str, object]) -> AudioVAEDecoder:
    """Strict-load the decoder and normalization subset from the 102-key core mapping."""

    rules, target_shapes = _decoder_rules(mapping)
    decoder = AudioVAEDecoder()
    for batch in iter_component_weight_batches((checkpoint,), rules, expected_target_shapes=target_shapes):
        decoder.load_weights(batch)
        mx.eval(*[value for _, value in batch])
    return decoder
