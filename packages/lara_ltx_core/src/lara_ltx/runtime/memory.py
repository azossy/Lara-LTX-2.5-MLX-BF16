"""Shape-derived memory estimates used for safe preflight checks."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from lara_ltx.errors import LaraError

BF16_BYTES = 2
ATTENTION_IO_TENSOR_COUNT = 4
SELF_ATTENTION_PROJECTION_COUNT = 4
FEED_FORWARD_PROJECTION_COUNT = 2
BLOCK_LIVE_HIDDEN_STATE_COUNT = 7
BYTES_PER_GIBIBYTE = 1024**3
SYSTEM_PAGE_SIZE_KEY = "SC_PAGE_SIZE"
SYSTEM_PHYSICAL_PAGE_COUNT_KEY = "SC_PHYS_PAGES"


@dataclass(frozen=True)
class GenerationResourcePolicy:
    """Versioned, evidence-based limits for one public generation profile."""

    enforce: bool
    minimum_unified_memory_bytes: int
    maximum_stage_two_video_tokens: int
    video_time_scale: int
    stage_two_spatial_scale: int
    recommended_height: int
    recommended_width: int
    recommended_num_frames: int


@dataclass(frozen=True)
class GenerationResourceAssessment:
    """Calculated resource envelope for one requested output grid."""

    requested_stage_two_video_tokens: int
    detected_unified_memory_bytes: int | None
    token_limit_satisfied: bool
    memory_requirement_satisfied: bool


@dataclass(frozen=True)
class AttentionShape:
    """Batch/head/token dimensions at the fused SDPA boundary."""

    batch_size: int
    heads: int
    tokens: int
    head_dim: int

    def elements_per_tensor(self) -> int:
        values = (self.batch_size, self.heads, self.tokens, self.head_dim)
        if any(value <= 0 for value in values):
            raise LaraError(
                "LARA-CONFIG-003",
                details={"key": "attention_shape", "value": values, "path": "runtime"},
            )
        return self.batch_size * self.heads * self.tokens * self.head_dim


def stage_video_tokens(*, frames: int, height: int, width: int, time_scale: int, spatial_scale: int) -> int:
    """Return the LTX video-token count for a pixel-space stage shape."""

    values = (frames, height, width, time_scale, spatial_scale)
    if any(value <= 0 for value in values):
        raise LaraError(
            "LARA-CONFIG-003",
            details={"key": "canonical_hq_shape", "value": values, "path": "runtime"},
        )
    if height % spatial_scale != 0 or width % spatial_scale != 0 or (frames - 1) % time_scale != 0:
        raise LaraError(
            "LARA-CONFIG-003",
            details={"key": "canonical_hq_grid", "value": values, "path": "runtime"},
        )
    latent_frames = (frames - 1) // time_scale + 1
    return latent_frames * (height // spatial_scale) * (width // spatial_scale)


def physical_memory_bytes() -> int | None:
    """Return physical unified-memory capacity without launching a subprocess."""

    try:
        page_size = int(os.sysconf(SYSTEM_PAGE_SIZE_KEY))
        page_count = int(os.sysconf(SYSTEM_PHYSICAL_PAGE_COUNT_KEY))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return page_size * page_count


def validate_generation_resources(
    *,
    height: int,
    width: int,
    num_frames: int,
    policy: GenerationResourcePolicy,
    detected_unified_memory_bytes: int | None,
) -> GenerationResourceAssessment:
    """Reject unmeasured grids before model loading can trigger swap or OOM."""

    requested_tokens = stage_video_tokens(
        frames=num_frames,
        height=height,
        width=width,
        time_scale=policy.video_time_scale,
        spatial_scale=policy.stage_two_spatial_scale,
    )
    token_limit_satisfied = requested_tokens <= policy.maximum_stage_two_video_tokens
    memory_requirement_satisfied = (
        detected_unified_memory_bytes is not None
        and detected_unified_memory_bytes >= policy.minimum_unified_memory_bytes
    )
    assessment = GenerationResourceAssessment(
        requested_stage_two_video_tokens=requested_tokens,
        detected_unified_memory_bytes=detected_unified_memory_bytes,
        token_limit_satisfied=token_limit_satisfied,
        memory_requirement_satisfied=memory_requirement_satisfied,
    )
    if not policy.enforce or (token_limit_satisfied and memory_requirement_satisfied):
        return assessment

    if detected_unified_memory_bytes is None:
        reason = "unavailable_unified_memory_capacity"
        detected_memory_gib: str | float = "unknown"
    elif not memory_requirement_satisfied:
        reason = "insufficient_unified_memory"
        detected_memory_gib = round(detected_unified_memory_bytes / BYTES_PER_GIBIBYTE, 1)
    else:
        reason = "stage_two_video_token_limit"
        detected_memory_gib = round(detected_unified_memory_bytes / BYTES_PER_GIBIBYTE, 1)
    raise LaraError(
        "LARA-RUNTIME-010",
        details={
            "reason": reason,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "requested_tokens": requested_tokens,
            "maximum_tokens": policy.maximum_stage_two_video_tokens,
            "detected_memory_gib": detected_memory_gib,
            "minimum_memory_gib": math.ceil(policy.minimum_unified_memory_bytes / BYTES_PER_GIBIBYTE),
            "recommended_height": policy.recommended_height,
            "recommended_width": policy.recommended_width,
            "recommended_num_frames": policy.recommended_num_frames,
        },
    )


def bf16_attention_io_bytes(shape: AttentionShape) -> int:
    """Estimate resident Q/K/V/output bytes, excluding implementation workspace."""

    return shape.elements_per_tensor() * ATTENTION_IO_TENSOR_COUNT * BF16_BYTES


def bf16_transformer_block_probe_bytes(*, tokens: int, hidden_size: int, feed_forward_multiplier: int) -> int:
    """Conservative score-matrix-free bytes for the synthetic block probe."""

    values = (tokens, hidden_size, feed_forward_multiplier)
    if any(value <= 0 for value in values):
        raise LaraError(
            "LARA-CONFIG-003",
            details={"key": "transformer_block_shape", "value": values, "path": "runtime"},
        )
    attention_weights = SELF_ATTENTION_PROJECTION_COUNT * hidden_size * hidden_size
    feed_forward_weights = FEED_FORWARD_PROJECTION_COUNT * hidden_size * hidden_size * feed_forward_multiplier
    live_activations = tokens * hidden_size * (BLOCK_LIVE_HIDDEN_STATE_COUNT + feed_forward_multiplier)
    return (attention_weights + feed_forward_weights + live_activations) * BF16_BYTES
