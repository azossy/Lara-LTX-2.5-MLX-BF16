"""Checkpoint-compatible video/audio transformer output modulation."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError

DEFAULT_LAYER_NORM_EPSILON = 1e-6
OUTPUT_MODULATION_COUNT = 2
OUTPUT_SHIFT_INDEX = 0
OUTPUT_SCALE_INDEX = 1


@dataclass(frozen=True)
class TransformerOutputConfig:
    hidden_dimension: int
    output_channels: int
    norm_epsilon: float = DEFAULT_LAYER_NORM_EPSILON


class TransformerOutputHead(nn.Module):
    """Apply the official final LayerNorm, AdaLN scale/shift and projection."""

    def __init__(self, config: TransformerOutputConfig) -> None:
        super().__init__()
        self.config = config
        self.scale_shift_table = mx.zeros((OUTPUT_MODULATION_COUNT, config.hidden_dimension))
        self.proj_out = nn.Linear(config.hidden_dimension, config.output_channels, bias=True)

    def __call__(self, x: mx.array, embedded_timestep: mx.array) -> mx.array:
        if x.ndim != 3 or embedded_timestep.ndim != 3 or x.shape != embedded_timestep.shape:
            raise LaraError(
                "LARA-TENSOR-020",
                details={"hidden_shape": tuple(x.shape), "timestep_shape": tuple(embedded_timestep.shape)},
            )
        if x.shape[-1] != self.config.hidden_dimension:
            raise LaraError(
                "LARA-TENSOR-020",
                details={"hidden_shape": tuple(x.shape), "timestep_shape": tuple(embedded_timestep.shape)},
            )
        modulation = self.scale_shift_table[None, None].astype(x.dtype) + embedded_timestep[:, :, None]
        shift = modulation[:, :, OUTPUT_SHIFT_INDEX]
        scale = modulation[:, :, OUTPUT_SCALE_INDEX]
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * mx.rsqrt(variance + self.config.norm_epsilon)
        return self.proj_out(normalized * (1 + scale) + shift)


class AVTransformerOutput(nn.Module):
    """Mapping root for checkpoint-compatible video and audio output heads."""

    def __init__(self, *, video: TransformerOutputConfig, audio: TransformerOutputConfig) -> None:
        super().__init__()
        self.video = TransformerOutputHead(video)
        self.audio = TransformerOutputHead(audio)

    def __call__(
        self,
        video_hidden: mx.array,
        video_timestep: mx.array,
        audio_hidden: mx.array,
        audio_timestep: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return self.video(video_hidden, video_timestep), self.audio(audio_hidden, audio_timestep)
