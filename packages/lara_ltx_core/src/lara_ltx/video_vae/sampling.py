"""Checkpoint-compatible 3D depth-to-space upsampling for the Diffusion VAE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError

from .resnet import CausalConv3d

STRIDE_DIMENSION_COUNT: Final = 3
CHANNEL_AXIS: Final = 1
STRIDE_TIME_INDEX: Final = 0


def _stride_volume(stride: tuple[int, int, int]) -> int:
    if len(stride) != STRIDE_DIMENSION_COUNT or any(value <= 0 for value in stride):
        raise LaraError("LARA-TENSOR-010", details={"stride": stride, "channels": "unknown"})
    return math.prod(stride)


def _depth_to_space(x: mx.array, stride: tuple[int, int, int]) -> mx.array:
    """Rearrange NCTHW channels into time/height/width blocks."""

    batch, channels, time, height, width = x.shape
    stride_time, stride_height, stride_width = stride
    stride_volume = _stride_volume(stride)
    if channels % stride_volume:
        raise LaraError("LARA-TENSOR-010", details={"stride": stride, "channels": channels})
    result_channels = channels // stride_volume
    reshaped = x.reshape(
        batch,
        result_channels,
        stride_time,
        stride_height,
        stride_width,
        time,
        height,
        width,
    )
    return mx.transpose(reshaped, (0, 1, 5, 2, 6, 3, 7, 4)).reshape(
        batch,
        result_channels,
        time * stride_time,
        height * stride_height,
        width * stride_width,
    )


@dataclass(frozen=True)
class DepthToSpaceUpsampleConfig:
    """Supported upstream 3D depth-to-space upsampling configuration."""

    in_channels: int
    stride: tuple[int, int, int]
    residual: bool = False
    out_channels_reduction_factor: int = 1
    spatial_padding_mode: str = "zeros"


class DepthToSpaceUpsample(nn.Module):
    """Causal Conv3d plus upstream depth-to-space and optional residual path."""

    def __init__(self, config: DepthToSpaceUpsampleConfig) -> None:
        super().__init__()
        self.config = config
        stride_volume = _stride_volume(config.stride)
        if (
            config.out_channels_reduction_factor <= 0
            or config.in_channels % config.out_channels_reduction_factor
            or (config.residual and config.in_channels % stride_volume)
        ):
            raise LaraError("LARA-TENSOR-010", details={"stride": config.stride, "channels": config.in_channels})
        self.out_channels = stride_volume * config.in_channels // config.out_channels_reduction_factor
        self.conv = CausalConv3d(
            config.in_channels, self.out_channels, spatial_padding_mode=config.spatial_padding_mode
        )

    def __call__(self, x: mx.array, *, causal: bool = True) -> mx.array:
        stride_time = self.config.stride[STRIDE_TIME_INDEX]
        residual: mx.array | None = None
        if self.config.residual:
            residual = _depth_to_space(x, self.config.stride)
            repeat_count = _stride_volume(self.config.stride) // self.config.out_channels_reduction_factor
            residual = mx.concatenate([residual] * repeat_count, axis=CHANNEL_AXIS)
            if stride_time == 2:
                residual = residual[:, :, 1:]
        output = _depth_to_space(self.conv(x, causal=causal), self.config.stride)
        if stride_time == 2:
            output = output[:, :, 1:]
        return output if residual is None else output + residual


@dataclass(frozen=True)
class SpaceToDepthDownsampleConfig:
    """Supported upstream 3D space-to-depth downsampling configuration."""

    in_channels: int
    out_channels: int
    stride: tuple[int, int, int]


class SpaceToDepthDownsample(nn.Module):
    """Causal Conv3d plus averaged space-to-depth residual downsampling."""

    def __init__(self, config: SpaceToDepthDownsampleConfig) -> None:
        super().__init__()
        self.config = config
        stride_volume = _stride_volume(config.stride)
        input_volume = config.in_channels * stride_volume
        if config.out_channels <= 0 or config.out_channels % stride_volume or input_volume % config.out_channels:
            raise LaraError(
                "LARA-TENSOR-011",
                details={
                    "in_channels": config.in_channels,
                    "out_channels": config.out_channels,
                    "stride": config.stride,
                },
            )
        self.group_size = input_volume // config.out_channels
        self.conv = CausalConv3d(config.in_channels, config.out_channels // stride_volume)

    def _space_to_depth(self, x: mx.array) -> mx.array:
        batch, channels, time, height, width = x.shape
        stride_time, stride_height, stride_width = self.config.stride
        if time % stride_time or height % stride_height or width % stride_width:
            raise LaraError(
                "LARA-TENSOR-011",
                details={
                    "in_channels": channels,
                    "out_channels": self.config.out_channels,
                    "stride": self.config.stride,
                },
            )
        return mx.transpose(
            x.reshape(
                batch,
                channels,
                time // stride_time,
                stride_time,
                height // stride_height,
                stride_height,
                width // stride_width,
                stride_width,
            ),
            (0, 1, 3, 5, 7, 2, 4, 6),
        ).reshape(
            batch,
            channels * _stride_volume(self.config.stride),
            time // stride_time,
            height // stride_height,
            width // stride_width,
        )

    def __call__(self, x: mx.array, *, causal: bool = True) -> mx.array:
        if self.config.stride[STRIDE_TIME_INDEX] == 2:
            x = mx.concatenate([x[:, :, :1], x], axis=2)
        residual = self._space_to_depth(x)
        batch, channels, time, height, width = residual.shape
        residual = mx.mean(
            residual.reshape(batch, channels // self.group_size, self.group_size, time, height, width),
            axis=2,
        )
        output = self._space_to_depth(self.conv(x, causal=causal))
        return output + residual
