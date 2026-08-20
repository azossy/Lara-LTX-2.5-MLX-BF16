"""Shape-derived memory estimates used for safe preflight checks."""

from __future__ import annotations

from dataclasses import dataclass

from lara_ltx.errors import LaraError

BF16_BYTES = 2
ATTENTION_IO_TENSOR_COUNT = 4
SELF_ATTENTION_PROJECTION_COUNT = 4
FEED_FORWARD_PROJECTION_COUNT = 2
BLOCK_LIVE_HIDDEN_STATE_COUNT = 7


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
