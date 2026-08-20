"""Production deterministic neighborhood-attention blocks for the Diffusion VAE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError
from lara_ltx.transformer.attention import RMSNorm

from .diffusion_layers import ADALN_CHUNK_COUNT, LinearPixelShuffleUpsample, LinearPixelShuffleUpsampleConfig, SwiGLU
from .neighborhood_attention import neighborhood_attention_3d

DEFAULT_ROPE_BASE: Final = 10_000.0
DEFAULT_MASK_ELEMENT_BUDGET: Final = 33_554_432
MLP_ALIGNMENT: Final = 16


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Return the production (time, height, width) absolute-RoPE split."""

    if head_dim % 8:
        raise LaraError("LARA-TENSOR-007", details={"reason": "head_dim_not_multiple_of_8", "shape": head_dim})
    time_dim = (head_dim // 4) // 2 * 2
    spatial_dim = (head_dim - time_dim) // 2
    if spatial_dim % 2:
        time_dim -= 2
        spatial_dim = (head_dim - time_dim) // 2
    if time_dim <= 0 or spatial_dim <= 0:
        raise LaraError("LARA-TENSOR-007", details={"reason": "invalid_rope_split", "shape": head_dim})
    return time_dim, spatial_dim, spatial_dim


def _inverse_frequencies(dim: int, base: float) -> mx.array:
    return mx.power(mx.array(base, dtype=mx.float32), -mx.arange(0, dim, 2, dtype=mx.float32) / dim)


def _rotate_axis(x: mx.array, *, axis: int, inverse_frequencies: mx.array) -> mx.array:
    output_dtype = x.dtype
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).astype(mx.float32)
    positions = mx.arange(x.shape[axis], dtype=mx.float32)
    angle_shape = [1, 1, 1, 1, 1, inverse_frequencies.shape[0]]
    angle_shape[axis] = positions.shape[0]
    angles = (positions[:, None] * inverse_frequencies[None]).reshape(angle_shape)
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated = mx.stack([even * cosine - odd * sine, even * sine + odd * cosine], axis=-1).reshape(x.shape)
    return rotated.astype(output_dtype)


class QKVProjections(nn.Module):
    """Separately owned Q/K/V projections matching the post-load upstream layout."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        return self.to_q(x), self.to_k(x), self.to_v(x)


@dataclass(frozen=True)
class NeighborhoodAttention3DConfig:
    """Production deterministic Diffusion VAE attention configuration."""

    dim: int
    kernel_size: tuple[int, int, int]
    head_dim: int
    rope_dim_split: tuple[int, int, int] | None = None
    rope_base: float = DEFAULT_ROPE_BASE
    mask_element_budget: int = DEFAULT_MASK_ELEMENT_BUDGET


class NeighborhoodAttention3D(nn.Module):
    """QKV projection, RMSNorm, absolute 3D RoPE, bounded NA, and output projection."""

    def __init__(self, config: NeighborhoodAttention3DConfig) -> None:
        super().__init__()
        if config.dim % config.head_dim:
            raise LaraError("LARA-TENSOR-007", details={"reason": "dim_not_divisible", "shape": config.dim})
        self.config = config
        self.num_heads = config.dim // config.head_dim
        self.scale = config.head_dim**-0.5
        self.rope_dim_split = config.rope_dim_split or default_rope_dim_split(config.head_dim)
        if sum(self.rope_dim_split) != config.head_dim or any(dim % 2 for dim in self.rope_dim_split):
            raise LaraError("LARA-TENSOR-007", details={"reason": "invalid_rope_split", "shape": self.rope_dim_split})
        self.qkv = QKVProjections(config.dim)
        self.proj = nn.Linear(config.dim, config.dim, bias=True)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)
        self.inverse_frequencies = tuple(_inverse_frequencies(dim, config.rope_base) for dim in self.rope_dim_split)

    def _project(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        batch, time, height, width, _ = x.shape
        shape = (batch, time, height, width, self.num_heads, self.config.head_dim)
        query, key, value = self.qkv(x)
        return query.reshape(shape), key.reshape(shape), value.reshape(shape)

    def _rope(self, x: mx.array) -> mx.array:
        time_dim, height_dim, _ = self.rope_dim_split
        time = _rotate_axis(x[..., :time_dim], axis=1, inverse_frequencies=self.inverse_frequencies[0])
        height = _rotate_axis(
            x[..., time_dim : time_dim + height_dim],
            axis=2,
            inverse_frequencies=self.inverse_frequencies[1],
        )
        width = _rotate_axis(
            x[..., time_dim + height_dim :],
            axis=3,
            inverse_frequencies=self.inverse_frequencies[2],
        )
        return mx.concatenate([time, height, width], axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        dimensions = x.shape[1:4]
        if any(dimension < kernel for dimension, kernel in zip(dimensions, self.config.kernel_size, strict=True)):
            raise LaraError(
                "LARA-TENSOR-007",
                details={"reason": "dimensions_below_kernel", "shape": (dimensions, self.config.kernel_size)},
            )
        query, key, value = self._project(x)
        query = self._rope(self.q_norm(query) * self.scale)
        key = self._rope(self.k_norm(key))
        output = neighborhood_attention_3d(
            query,
            key,
            value,
            kernel_size=self.config.kernel_size,
            mask_element_budget=self.config.mask_element_budget,
            scale=1.0,
        )
        return self.proj(output.reshape(*x.shape[:-1], self.config.dim))


@dataclass(frozen=True)
class NABlockConfig:
    """Production deterministic-stage NA block configuration."""

    dim: int
    kernel_size: tuple[int, int, int]
    head_dim: int
    mlp_ratio: float = 4.0
    rope_dim_split: tuple[int, int, int] | None = None
    mask_element_budget: int = DEFAULT_MASK_ELEMENT_BUDGET


class NABlock(nn.Module):
    """Pre-norm neighborhood attention followed by a SwiGLU residual."""

    def __init__(self, config: NABlockConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.dim)
        self.attn = NeighborhoodAttention3D(
            NeighborhoodAttention3DConfig(
                dim=config.dim,
                kernel_size=config.kernel_size,
                head_dim=config.head_dim,
                rope_dim_split=config.rope_dim_split,
                mask_element_budget=config.mask_element_budget,
            )
        )
        self.norm2 = RMSNorm(config.dim)
        hidden_dim = math.ceil(int(config.dim * config.mlp_ratio) / MLP_ALIGNMENT) * MLP_ALIGNMENT
        self.mlp = SwiGLU(config.dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@dataclass(frozen=True)
class DeterministicStageConfig:
    """One production pre-diffusion decoder stage."""

    channels: int
    depth: int
    kernel_size: tuple[int, int, int]
    head_dim: int
    upsample_stride: tuple[int, int, int]
    out_channels_reduction_factor: int
    mask_element_budget: int = DEFAULT_MASK_ELEMENT_BUDGET


class DeterministicStage(nn.Module):
    """A sequence of deterministic NABlocks followed by pixel-shuffle upsampling."""

    def __init__(self, config: DeterministicStageConfig) -> None:
        super().__init__()
        self.blocks = [
            NABlock(
                NABlockConfig(
                    dim=config.channels,
                    kernel_size=config.kernel_size,
                    head_dim=config.head_dim,
                    mask_element_budget=config.mask_element_budget,
                )
            )
            for _ in range(config.depth)
        ]
        self.upsample = LinearPixelShuffleUpsample(
            LinearPixelShuffleUpsampleConfig(
                in_channels=config.channels,
                stride=config.upsample_stride,
                out_channels_reduction_factor=config.out_channels_reduction_factor,
            )
        )

    def __call__(self, x: mx.array, *, drop_leading_frame: bool = True) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return self.upsample(x, drop_leading_frame=drop_leading_frame)


@dataclass(frozen=True)
class CombinedDiffusionNABlockConfig:
    """Production stage-5 combined context/diffusion block configuration."""

    dim: int
    context_channels: int
    kernel_size: tuple[int, int, int]
    head_dim: int
    mlp_ratio: float = 4.0
    rope_dim_split: tuple[int, int, int] | None = None
    mask_element_budget: int = DEFAULT_MASK_ELEMENT_BUDGET


class CombinedDiffusionNABlock(nn.Module):
    """Context injection followed by modulated NA and SwiGLU residuals."""

    def __init__(self, config: CombinedDiffusionNABlockConfig) -> None:
        super().__init__()
        self.config = config
        self.context_proj = nn.Linear(config.context_channels, config.dim, bias=True)
        self.scale_shift_table = mx.zeros((ADALN_CHUNK_COUNT, config.dim))
        self.norm1 = RMSNorm(config.dim)
        self.attn = NeighborhoodAttention3D(
            NeighborhoodAttention3DConfig(
                dim=config.dim,
                kernel_size=config.kernel_size,
                head_dim=config.head_dim,
                rope_dim_split=config.rope_dim_split,
                mask_element_budget=config.mask_element_budget,
            )
        )
        self.norm2 = RMSNorm(config.dim)
        hidden_dim = math.ceil(int(config.dim * config.mlp_ratio) / MLP_ALIGNMENT) * MLP_ALIGNMENT
        self.mlp = SwiGLU(config.dim, hidden_dim)

    def _modulation(self, modulation: tuple[mx.array, ...]) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if len(modulation) != ADALN_CHUNK_COUNT:
            raise LaraError(
                "LARA-TENSOR-014",
                details={"block_name": "CombinedDiffusionNABlock"},
            )
        values = tuple(
            modulation[index] + self.scale_shift_table[index][None, None, None, None, :]
            for index in range(ADALN_CHUNK_COUNT)
        )
        return values[0], values[1], values[3], values[4]

    def __call__(self, context_and_x: mx.array, modulation: tuple[mx.array, ...]) -> mx.array:
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        context = context_and_x[..., : self.config.context_channels]
        x = context_and_x[..., self.config.context_channels :]
        x = x + self.context_proj(context)
        attention_input = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn(attention_input)
        mlp_input = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(mlp_input)
