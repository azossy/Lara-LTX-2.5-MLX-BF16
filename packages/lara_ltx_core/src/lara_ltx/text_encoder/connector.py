"""Checkpoint-backed 8-layer video/audio prompt connectors for MLX."""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lara_ltx.errors import LaraError
from lara_ltx.models.gemma_connector import (
    PromptConnectorConfig,
    prompt_connector_target_shapes,
    validate_prompt_connector_mapping,
)
from lara_ltx.models.loading import iter_component_weight_batches
from lara_ltx.transformer import Attention, FeedForward, LTXRopeType
from lara_ltx.transformer.layers import rms_norm


def _connector_frequencies(
    *,
    sequence_length: int,
    dimensions: int,
    heads: int,
    theta: float,
    maximum_position: int,
    dtype: mx.Dtype,
    double_precision: bool,
) -> tuple[mx.array, mx.array]:
    frequency_count = dimensions // 2
    numpy_dtype = np.float64 if double_precision else np.float32
    exponents = np.linspace(0.0, 1.0, frequency_count, dtype=numpy_dtype)
    frequency_grid = np.power(theta, exponents) * (math.pi / 2.0)
    frequencies = mx.array(frequency_grid.astype(np.float32))
    positions = mx.arange(sequence_length, dtype=mx.float32) / maximum_position
    phases = mx.expand_dims(positions * 2.0 - 1.0, -1) * frequencies
    head_dimension = dimensions // heads
    cosines = mx.swapaxes(mx.cos(phases).reshape(1, sequence_length, heads, head_dimension // 2), 1, 2)
    sines = mx.swapaxes(mx.sin(phases).reshape(1, sequence_length, heads, head_dimension // 2), 1, 2)
    return cosines.astype(dtype), sines.astype(dtype)


class ConnectorTransformerBlock(nn.Module):
    def __init__(self, *, dimensions: int, heads: int, head_dim: int, gated: bool) -> None:
        super().__init__()
        self.attn1 = Attention(
            dimensions,
            heads=heads,
            dim_head=head_dim,
            rope_type=LTXRopeType.SPLIT,
            apply_gated_attention=gated,
        )
        self.ff = FeedForward(dimensions, dimensions, bias=True)

    def __call__(self, hidden: mx.array, frequencies: tuple[mx.array, mx.array]) -> mx.array:
        hidden = hidden + self.attn1(rms_norm(hidden), pe=frequencies)
        return hidden + self.ff(rms_norm(hidden))


class Embeddings1DConnector(nn.Module):
    def __init__(self, config: PromptConnectorConfig, *, modality: str) -> None:
        super().__init__()
        if modality not in {"video", "audio"}:
            raise LaraError("LARA-MODEL-038", details={"key": "invalid_modality"})
        self.config = config
        self.modality = modality
        self.dimensions = config.dimensions(modality)
        self.heads = config.heads(modality)
        self.head_dim = self.dimensions // self.heads
        self.learnable_registers = mx.zeros((config.register_count, self.dimensions), dtype=mx.bfloat16)
        self.transformer_1d_blocks = [
            ConnectorTransformerBlock(
                dimensions=self.dimensions,
                heads=self.heads,
                head_dim=self.head_dim,
                gated=config.apply_gated_attention,
            )
            for _ in range(config.layer_count)
        ]

    def __call__(self, hidden: mx.array, right_attention_mask: mx.array) -> mx.array:
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != self.dimensions
            or tuple(right_attention_mask.shape) != tuple(hidden.shape[:2])
            or hidden.shape[1] % self.config.register_count
        ):
            raise LaraError("LARA-TENSOR-031", details={"reason": "invalid_connector_input"})
        sequence_length = hidden.shape[1]
        registers = mx.tile(self.learnable_registers, (sequence_length // self.config.register_count, 1))
        registers = mx.broadcast_to(registers[None], hidden.shape)
        hidden = mx.where(right_attention_mask[..., None].astype(mx.bool_), hidden, registers)
        frequencies = _connector_frequencies(
            sequence_length=sequence_length,
            dimensions=self.dimensions,
            heads=self.heads,
            theta=self.config.positional_theta,
            maximum_position=self.config.positional_maximum,
            dtype=hidden.dtype,
            double_precision=self.config.double_precision_frequencies,
        )
        for block in self.transformer_1d_blocks:
            hidden = block(hidden, frequencies)
            mx.eval(hidden)
        hidden = rms_norm(hidden)
        mx.eval(hidden)
        return hidden


class PromptConnectorProcessor(nn.Module):
    def __init__(self, config: PromptConnectorConfig) -> None:
        super().__init__()
        self.config = config
        self.video_connector = Embeddings1DConnector(config, modality="video")
        self.audio_connector = Embeddings1DConnector(config, modality="audio")

    @staticmethod
    def _right_pad(features: mx.array, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        mask = np.asarray(attention_mask)
        if mask.ndim != 2 or features.shape[:2] != mask.shape:
            raise LaraError("LARA-TENSOR-031", details={"reason": "invalid_connector_mask"})
        order = np.argsort(-mask, axis=1, kind="stable")
        indices = mx.array(order, dtype=mx.int32)
        expanded = mx.broadcast_to(indices[..., None], features.shape)
        return mx.take_along_axis(features, expanded, axis=1), mx.take_along_axis(attention_mask, indices, axis=1)

    def __call__(
        self,
        video_features: mx.array,
        audio_features: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        video, right_mask = self._right_pad(video_features, attention_mask)
        audio, audio_mask = self._right_pad(audio_features, attention_mask)
        video = self.video_connector(video, right_mask)
        audio = self.audio_connector(audio, audio_mask)
        mx.eval(video, audio, right_mask)
        return video, audio, right_mask


def load_prompt_connector_processor(
    *,
    checkpoint: Path,
    mapping: dict[str, object],
    config: PromptConnectorConfig,
) -> PromptConnectorProcessor:
    rules = validate_prompt_connector_mapping(mapping, config)
    processor = PromptConnectorProcessor(config)
    for batch in iter_component_weight_batches(
        (checkpoint,),
        rules,
        expected_target_shapes=prompt_connector_target_shapes(config),
    ):
        processor.load_weights(batch, strict=False)
        mx.eval(*[value for _, value in batch])
    return processor
