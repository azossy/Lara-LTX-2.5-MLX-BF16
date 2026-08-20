"""MLX runtime for the checkpoint-mapped Gemma V2 LTX feature projections."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError
from lara_ltx.models.gemma_feature import (
    GEMMA_AUDIO_FEATURE_DIMENSION,
    GEMMA_FEATURE_INPUT_DIMENSION,
    GEMMA_HIDDEN_DIMENSION,
    GEMMA_HIDDEN_STATE_COUNT,
    GEMMA_VIDEO_FEATURE_DIMENSION,
)

RMS_EPSILON = 1e-6
HIDDEN_RANK = 4
MASK_RANK = 2


class GemmaFeatureExtractorV2(nn.Module):
    """Per-token RMS normalization plus LTX video/audio feature projections."""

    def __init__(self) -> None:
        super().__init__()
        self.video_aggregate_embed = nn.Linear(
            GEMMA_FEATURE_INPUT_DIMENSION,
            GEMMA_VIDEO_FEATURE_DIMENSION,
            bias=True,
        )
        self.audio_aggregate_embed = nn.Linear(
            GEMMA_FEATURE_INPUT_DIMENSION,
            GEMMA_AUDIO_FEATURE_DIMENSION,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        """Project Gemma hidden states with upstream V2 masking and rescaling."""

        _validate_inputs(hidden_states, attention_mask)
        variance = mx.mean(mx.square(hidden_states), axis=2, keepdims=True)
        normalized = hidden_states * mx.rsqrt(variance + RMS_EPSILON)
        batch, sequence_length, _, _ = hidden_states.shape
        flattened = normalized.reshape(batch, sequence_length, GEMMA_FEATURE_INPUT_DIMENSION)
        masked = mx.where(attention_mask.astype(mx.bool_)[..., None], flattened, mx.zeros_like(flattened))
        video = self.video_aggregate_embed(masked * math.sqrt(GEMMA_VIDEO_FEATURE_DIMENSION / GEMMA_HIDDEN_DIMENSION))
        audio = self.audio_aggregate_embed(masked * math.sqrt(GEMMA_AUDIO_FEATURE_DIMENSION / GEMMA_HIDDEN_DIMENSION))
        return video, audio


class GemmaFeatureProjector(nn.Module):
    """Mapping-root wrapper whose MLX parameter keys match the reviewed manifest."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = GemmaFeatureExtractorV2()

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        return self.feature_extractor(hidden_states, attention_mask)


def _validate_inputs(hidden_states: mx.array, attention_mask: mx.array) -> None:
    hidden_shape = tuple(hidden_states.shape)
    mask_shape = tuple(attention_mask.shape)
    if (
        hidden_states.ndim != HIDDEN_RANK
        or hidden_shape[2] != GEMMA_HIDDEN_DIMENSION
        or hidden_shape[3] != GEMMA_HIDDEN_STATE_COUNT
        or attention_mask.ndim != MASK_RANK
        or mask_shape != hidden_shape[:2]
    ):
        raise LaraError(
            "LARA-TENSOR-019",
            details={"hidden_shape": hidden_shape, "mask_shape": mask_shape},
        )
