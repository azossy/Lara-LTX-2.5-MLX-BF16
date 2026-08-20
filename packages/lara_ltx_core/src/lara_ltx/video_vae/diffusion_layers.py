"""Shared MLX layers for the production Diffusion VAE transformer stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError

STRIDE_DIMENSION_COUNT: Final = 3
ADALN_CHUNK_COUNT: Final = 7


@dataclass(frozen=True)
class LinearPixelShuffleUpsampleConfig:
    """Production channel-last VAE pixel-shuffle configuration."""

    in_channels: int
    stride: tuple[int, int, int]
    out_channels_reduction_factor: int = 1


class LinearPixelShuffleUpsample(nn.Module):
    """Linear channel expansion followed by channel-last 3D pixel shuffle."""

    def __init__(self, config: LinearPixelShuffleUpsampleConfig) -> None:
        super().__init__()
        if (
            len(config.stride) != STRIDE_DIMENSION_COUNT
            or any(value <= 0 for value in config.stride)
            or config.out_channels_reduction_factor <= 0
            or config.in_channels % config.out_channels_reduction_factor
        ):
            raise LaraError(
                "LARA-TENSOR-015",
                details={"stride": config.stride, "channels": config.in_channels},
            )
        self.config = config
        stride_volume = math.prod(config.stride)
        self.proj_out_channels = stride_volume * config.in_channels // config.out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // stride_volume
        self.proj = nn.Linear(config.in_channels, self.proj_out_channels, bias=True)

    def __call__(self, x: mx.array, *, drop_leading_frame: bool = True) -> mx.array:
        batch, time, height, width, _ = x.shape
        stride_time, stride_height, stride_width = self.config.stride
        projected = self.proj(x).reshape(
            batch,
            time,
            height,
            width,
            self.out_channels,
            stride_time,
            stride_height,
            stride_width,
        )
        output = mx.transpose(projected, (0, 1, 5, 2, 6, 3, 7, 4)).reshape(
            batch,
            time * stride_time,
            height * stride_height,
            width * stride_width,
            self.out_channels,
        )
        if stride_time == 2 and drop_leading_frame:
            output = output[:, 1:]
        return output


class AdaLNZero(nn.Module):
    """Seven-way timestep modulation projection used by Diffusion VAE blocks."""

    def __init__(self, dim: int, timestep_embedding_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(timestep_embedding_dim, ADALN_CHUNK_COUNT * dim, bias=True)

    def __call__(self, timestep_embedding: mx.array) -> tuple[mx.array, ...]:
        activated = timestep_embedding * mx.sigmoid(timestep_embedding)
        chunks = mx.split(self.proj(activated), ADALN_CHUNK_COUNT, axis=-1)
        return tuple(chunk[:, None, None, None, :] for chunk in chunks)


class SwiGLU(nn.Module):
    """Checkpoint-compatible bias-free gated MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.w_gate(x)
        return self.w_down((gate * mx.sigmoid(gate)) * self.w_up(x))
