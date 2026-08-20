"""Incremental checkpoint-compatible convolutional Diffusion VAE decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError

from .ops import PerChannelStatistics, unpatchify
from .resnet import (
    SPATIAL_PADDING_REFLECT,
    CausalConv3d,
    GroupNormNCTHW,
    ResnetBlock3D,
    ResnetBlock3DConfig,
    pixel_norm,
)
from .sampling import DepthToSpaceUpsample, DepthToSpaceUpsampleConfig

BLOCK_RES_X_Y: Final = "res_x_y"
BLOCK_COMPRESS_TIME: Final = "compress_time"
BLOCK_COMPRESS_SPACE: Final = "compress_space"
BLOCK_COMPRESS_ALL: Final = "compress_all"
SUPPORTED_BLOCK_NAMES: Final = frozenset((BLOCK_RES_X_Y, BLOCK_COMPRESS_TIME, BLOCK_COMPRESS_SPACE, BLOCK_COMPRESS_ALL))
COMPRESS_TIME_STRIDE: Final = (2, 1, 1)
COMPRESS_SPACE_STRIDE: Final = (1, 2, 2)
COMPRESS_ALL_STRIDE: Final = (2, 2, 2)


@dataclass(frozen=True)
class DecoderBlockConfig:
    """One supported non-attention convolutional VAE decoder block."""

    name: str
    multiplier: int = 1
    residual: bool = False


@dataclass(frozen=True)
class ConvVideoDecoderConfig:
    """Configuration for the presently supported Conv VAE decoder path."""

    in_channels: int
    out_channels: int
    base_channels: int
    decoder_blocks: tuple[DecoderBlockConfig, ...]
    patch_size: int = 1
    groups: int = 1
    norm_layer: str = "pixel_norm"
    causal: bool = True


def _decoder_bottleneck_channels(config: ConvVideoDecoderConfig) -> int:
    channels = config.base_channels
    for block in config.decoder_blocks:
        if block.name not in SUPPORTED_BLOCK_NAMES or block.multiplier <= 0:
            raise LaraError("LARA-TENSOR-013", details={"block_name": block.name})
        channels *= block.multiplier
    return channels


class ConvVideoDecoder(nn.Module):
    """Conv VAE decoder subset with ResNet and depth-to-space blocks.

    Attention and timestep-conditioned blocks intentionally remain outside this
    component until their checkpoint-backed counterparts have parity coverage.
    """

    def __init__(self, config: ConvVideoDecoderConfig) -> None:
        super().__init__()
        if config.patch_size <= 0 or config.out_channels <= 0:
            raise LaraError("LARA-TENSOR-013", details={"block_name": "decoder_configuration"})
        self.config = config
        bottleneck_channels = _decoder_bottleneck_channels(config)
        self.per_channel_statistics = PerChannelStatistics(config.in_channels)
        self.conv_in = CausalConv3d(
            config.in_channels, bottleneck_channels, spatial_padding_mode=SPATIAL_PADDING_REFLECT
        )
        self.up_blocks = []
        feature_channels = bottleneck_channels
        for block in reversed(config.decoder_blocks):
            layer, feature_channels = self._make_block(block, feature_channels)
            self.up_blocks.append(layer)
        self.conv_norm_out = (
            GroupNormNCTHW(feature_channels, config.groups) if config.norm_layer == "group_norm" else None
        )
        if config.norm_layer not in ("pixel_norm", "group_norm"):
            raise LaraError("LARA-TENSOR-009", details={"norm_layer": config.norm_layer})
        self.conv_out = CausalConv3d(
            feature_channels, config.out_channels * config.patch_size**2, spatial_padding_mode=SPATIAL_PADDING_REFLECT
        )

    def _make_block(self, block: DecoderBlockConfig, in_channels: int) -> tuple[nn.Module, int]:
        if block.name == BLOCK_RES_X_Y:
            out_channels = in_channels // block.multiplier
            if in_channels % block.multiplier:
                raise LaraError("LARA-TENSOR-013", details={"block_name": block.name})
            return (
                ResnetBlock3D(
                    ResnetBlock3DConfig(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        groups=self.config.groups,
                        norm_layer=self.config.norm_layer,
                        spatial_padding_mode=SPATIAL_PADDING_REFLECT,
                    )
                ),
                out_channels,
            )
        stride_by_name = {
            BLOCK_COMPRESS_TIME: COMPRESS_TIME_STRIDE,
            BLOCK_COMPRESS_SPACE: COMPRESS_SPACE_STRIDE,
            BLOCK_COMPRESS_ALL: COMPRESS_ALL_STRIDE,
        }
        stride = stride_by_name.get(block.name)
        if stride is None:
            raise LaraError("LARA-TENSOR-013", details={"block_name": block.name})
        if in_channels % block.multiplier:
            raise LaraError("LARA-TENSOR-013", details={"block_name": block.name})
        return (
            DepthToSpaceUpsample(
                DepthToSpaceUpsampleConfig(
                    in_channels=in_channels,
                    stride=stride,
                    residual=block.residual,
                    out_channels_reduction_factor=block.multiplier,
                    spatial_padding_mode=SPATIAL_PADDING_REFLECT,
                )
            ),
            in_channels // block.multiplier,
        )

    def __call__(self, sample: mx.array) -> mx.array:
        output_dtype = sample.dtype
        hidden = self.per_channel_statistics.un_normalize(sample)
        hidden = self.conv_in(hidden, causal=self.config.causal)
        for block in self.up_blocks:
            hidden = block(hidden, causal=self.config.causal)
        hidden = pixel_norm(hidden) if self.conv_norm_out is None else self.conv_norm_out(hidden)
        hidden = hidden * mx.sigmoid(hidden)
        hidden = self.conv_out(hidden, causal=self.config.causal)
        return unpatchify(hidden, patch_size_hw=self.config.patch_size).astype(output_dtype)
