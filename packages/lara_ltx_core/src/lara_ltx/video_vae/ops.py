"""Checkpoint-compatible tensor operations used at Diffusion VAE boundaries."""

from __future__ import annotations

from typing import Final

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError

CHANNEL_AXIS: Final = 1
FOUR_DIMENSIONS: Final = 4
FIVE_DIMENSIONS: Final = 5


def _validate_patch_layout(x: mx.array, patch_size_hw: int, patch_size_t: int) -> None:
    if x.ndim not in (FOUR_DIMENSIONS, FIVE_DIMENSIONS) or patch_size_hw <= 0 or patch_size_t <= 0:
        raise LaraError(
            "LARA-TENSOR-012",
            details={"shape": tuple(x.shape), "patch_size_hw": patch_size_hw, "patch_size_t": patch_size_t},
        )
    spatial_axes = x.shape[-2:]
    if any(size % patch_size_hw for size in spatial_axes):
        raise LaraError(
            "LARA-TENSOR-012",
            details={"shape": tuple(x.shape), "patch_size_hw": patch_size_hw, "patch_size_t": patch_size_t},
        )
    if x.ndim == FIVE_DIMENSIONS and x.shape[2] % patch_size_t:
        raise LaraError(
            "LARA-TENSOR-012",
            details={"shape": tuple(x.shape), "patch_size_hw": patch_size_hw, "patch_size_t": patch_size_t},
        )


def patchify(x: mx.array, *, patch_size_hw: int, patch_size_t: int = 1) -> mx.array:
    """Move NCTHW patch dimensions into channels using upstream ordering."""

    _validate_patch_layout(x, patch_size_hw, patch_size_t)
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.ndim == FOUR_DIMENSIONS:
        batch, channels, height, width = x.shape
        return mx.transpose(
            x.reshape(batch, channels, height // patch_size_hw, patch_size_hw, width // patch_size_hw, patch_size_hw),
            (0, 1, 5, 3, 2, 4),
        ).reshape(batch, channels * patch_size_hw**2, height // patch_size_hw, width // patch_size_hw)
    batch, channels, frames, height, width = x.shape
    return mx.transpose(
        x.reshape(
            batch,
            channels,
            frames // patch_size_t,
            patch_size_t,
            height // patch_size_hw,
            patch_size_hw,
            width // patch_size_hw,
            patch_size_hw,
        ),
        (0, 1, 3, 7, 5, 2, 4, 6),
    ).reshape(
        batch,
        channels * patch_size_t * patch_size_hw**2,
        frames // patch_size_t,
        height // patch_size_hw,
        width // patch_size_hw,
    )


def unpatchify(x: mx.array, *, patch_size_hw: int, patch_size_t: int = 1) -> mx.array:
    """Move NCTHW channel patches into spatial/temporal dimensions."""

    _validate_patch_layout(x, patch_size_hw=1, patch_size_t=1)
    patch_volume = patch_size_t * patch_size_hw**2
    if patch_size_hw <= 0 or patch_size_t <= 0 or x.shape[CHANNEL_AXIS] % patch_volume:
        raise LaraError(
            "LARA-TENSOR-012",
            details={"shape": tuple(x.shape), "patch_size_hw": patch_size_hw, "patch_size_t": patch_size_t},
        )
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.ndim == FOUR_DIMENSIONS:
        batch, channels, height, width = x.shape
        output_channels = channels // patch_volume
        return mx.transpose(
            x.reshape(batch, output_channels, patch_size_hw, patch_size_hw, height, width),
            (0, 1, 4, 3, 5, 2),
        ).reshape(batch, output_channels, height * patch_size_hw, width * patch_size_hw)
    batch, channels, frames, height, width = x.shape
    output_channels = channels // patch_volume
    return mx.transpose(
        x.reshape(batch, output_channels, patch_size_t, patch_size_hw, patch_size_hw, frames, height, width),
        (0, 1, 5, 2, 6, 4, 7, 3),
    ).reshape(
        batch,
        output_channels,
        frames * patch_size_t,
        height * patch_size_hw,
        width * patch_size_hw,
    )


class PerChannelStatistics(nn.Module):
    """Stored latent mean/std tensors with the upstream normalization semantics."""

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.std_of_means = mx.ones((latent_channels,))
        self.mean_of_means = mx.zeros((latent_channels,))

    def un_normalize(self, x: mx.array) -> mx.array:
        return x * self.std_of_means[None, :, None, None, None] + self.mean_of_means[None, :, None, None, None]

    def normalize(self, x: mx.array) -> mx.array:
        return (x - self.mean_of_means[None, :, None, None, None]) / self.std_of_means[None, :, None, None, None]
