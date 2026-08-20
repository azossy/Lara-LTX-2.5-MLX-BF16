"""Checkpoint-compatible PixArt timestep embeddings shared by LTX components."""

from __future__ import annotations

import math
from typing import Final

import mlx.core as mx
import mlx.nn as nn

TIMESTEP_PROJECTION_CHANNELS: Final = 256
MAX_PERIOD: Final = 10_000


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = MAX_PERIOD,
) -> mx.array:
    """Create the upstream float32 sinusoidal timestep projection."""

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    arguments = scale * timesteps[:, None].astype(mx.float32) * mx.exp(exponent)[None]
    embedding = mx.concatenate([mx.sin(arguments), mx.cos(arguments)], axis=-1)
    if flip_sin_to_cos:
        embedding = mx.concatenate([embedding[:, half_dim:], embedding[:, :half_dim]], axis=-1)
    if embedding_dim % 2:
        embedding = mx.concatenate([embedding, mx.zeros((embedding.shape[0], 1))], axis=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    """Two-layer SiLU MLP with upstream parameter names."""

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = sample * mx.sigmoid(sample)
        return self.linear_2(sample)


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """The timestep-only path used by the VAE and transformer blocks."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(TIMESTEP_PROJECTION_CHANNELS, embedding_dim)

    def __call__(self, timestep: mx.array, *, hidden_dtype: mx.Dtype) -> mx.array:
        projection = get_timestep_embedding(
            timestep,
            TIMESTEP_PROJECTION_CHANNELS,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        return self.timestep_embedder(projection.astype(hidden_dtype))
