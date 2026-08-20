"""MLX port of the canonical LTX rotary-position implementation."""

from __future__ import annotations

import math
from enum import Enum

import mlx.core as mx

from lara_ltx.errors import LaraError

DEFAULT_ROPE_THETA = 10_000.0
DEFAULT_MAX_POSITION_COUNTS = (20, 2_048, 2_048)
DEFAULT_ATTENTION_HEADS = 32
ROTARY_PAIR_SIZE = 2


class LTXRopeType(Enum):
    """Rotary layouts supported by the upstream checkpoint."""

    INTERLEAVED = "interleaved"
    SPLIT = "split"


def _validate_frequency_shapes(cos_freqs: mx.array, sin_freqs: mx.array) -> None:
    if cos_freqs.shape != sin_freqs.shape:
        raise LaraError(
            "LARA-TENSOR-001",
            details={"left": tuple(cos_freqs.shape), "right": tuple(sin_freqs.shape)},
        )


def apply_split_rotary_emb(input_tensor: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    """Apply the split-half RoPE layout used by LTX-2.5."""

    _validate_frequency_shapes(cos_freqs, sin_freqs)
    needs_reshape = input_tensor.ndim != 4 and cos_freqs.ndim == 4
    if needs_reshape:
        batch, tokens = input_tensor.shape[:2]
        frequency_batch, heads = cos_freqs.shape[:2]
        if frequency_batch not in (1, batch):
            raise LaraError(
                "LARA-TENSOR-002",
                details={"frequency_batch": frequency_batch, "input_batch": batch},
            )
        input_tensor = mx.swapaxes(input_tensor.reshape(batch, tokens, heads, -1), 1, 2)

    feature_size = input_tensor.shape[-1]
    if feature_size % ROTARY_PAIR_SIZE:
        raise LaraError("LARA-TENSOR-003", details={"size": feature_size})

    half_size = feature_size // ROTARY_PAIR_SIZE
    first_half = input_tensor[..., :half_size]
    second_half = input_tensor[..., half_size:]
    first_output = first_half * cos_freqs - second_half * sin_freqs
    second_output = second_half * cos_freqs + first_half * sin_freqs
    output = mx.concatenate((first_output, second_output), axis=-1)

    if needs_reshape:
        output = mx.swapaxes(output, 1, 2).reshape(batch, tokens, -1)
    return output


def apply_interleaved_rotary_emb(input_tensor: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    """Apply the legacy adjacent-pair RoPE layout."""

    _validate_frequency_shapes(cos_freqs, sin_freqs)
    feature_size = input_tensor.shape[-1]
    if feature_size % ROTARY_PAIR_SIZE:
        raise LaraError("LARA-TENSOR-003", details={"size": feature_size})
    pairs = input_tensor.reshape(*input_tensor.shape[:-1], -1, ROTARY_PAIR_SIZE)
    rotated = mx.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(input_tensor.shape)
    return input_tensor * cos_freqs + rotated * sin_freqs


def apply_rotary_emb(
    input_tensor: mx.array,
    frequencies: tuple[mx.array, mx.array],
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> mx.array:
    """Dispatch to the checkpoint's configured rotary layout."""

    cos_freqs, sin_freqs = frequencies
    if rope_type is LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    if rope_type is LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    raise LaraError("LARA-TENSOR-004", details={"rope_type": str(rope_type)})


def generate_frequency_grid(theta: float, positional_dimensions: int, inner_dim: int) -> mx.array:
    """Generate upstream-equivalent float32 angular frequencies."""

    element_count = ROTARY_PAIR_SIZE * positional_dimensions
    frequency_count = inner_dim // element_count
    exponents = mx.linspace(0.0, 1.0, frequency_count, dtype=mx.float32)
    return mx.power(mx.array(theta, dtype=mx.float32), exponents) * (math.pi / ROTARY_PAIR_SIZE)


def _fractional_positions(indices_grid: mx.array, max_positions: tuple[int, ...]) -> mx.array:
    positional_dimensions = indices_grid.shape[1]
    if positional_dimensions != len(max_positions):
        raise LaraError(
            "LARA-TENSOR-005",
            details={"dimensions": positional_dimensions, "max_positions": len(max_positions)},
        )
    columns = [indices_grid[:, index] / max_positions[index] for index in range(positional_dimensions)]
    return mx.stack(columns, axis=-1)


def _generate_frequencies(indices: mx.array, indices_grid: mx.array, max_positions: tuple[int, ...]) -> mx.array:
    fractional_positions = _fractional_positions(indices_grid, max_positions)
    frequencies = indices * (mx.expand_dims(fractional_positions, -1) * ROTARY_PAIR_SIZE - 1)
    frequencies = mx.swapaxes(frequencies, -1, -2)
    return frequencies.reshape(*frequencies.shape[:-2], -1)


def _split_frequency_components(
    frequencies: mx.array,
    pad_size: int,
    attention_heads: int,
) -> tuple[mx.array, mx.array]:
    cos_freqs = mx.cos(frequencies)
    sin_freqs = mx.sin(frequencies)
    if pad_size:
        padding_shape = (*cos_freqs.shape[:-1], pad_size)
        cos_freqs = mx.concatenate((mx.ones(padding_shape, dtype=cos_freqs.dtype), cos_freqs), axis=-1)
        sin_freqs = mx.concatenate((mx.zeros(padding_shape, dtype=sin_freqs.dtype), sin_freqs), axis=-1)
    batch, tokens = cos_freqs.shape[:2]
    cos_freqs = mx.swapaxes(cos_freqs.reshape(batch, tokens, attention_heads, -1), 1, 2)
    sin_freqs = mx.swapaxes(sin_freqs.reshape(batch, tokens, attention_heads, -1), 1, 2)
    return cos_freqs, sin_freqs


def precompute_freqs_cis(
    indices_grid: mx.array,
    dim: int,
    *,
    theta: float = DEFAULT_ROPE_THETA,
    max_positions: tuple[int, ...] = DEFAULT_MAX_POSITION_COUNTS,
    use_middle_indices_grid: bool = False,
    attention_heads: int = DEFAULT_ATTENTION_HEADS,
) -> tuple[mx.array, mx.array]:
    """Precompute Split RoPE cosine/sine tensors for LTX-2.5."""

    if indices_grid.ndim == 4:
        indices_grid = mx.mean(indices_grid, axis=-1) if use_middle_indices_grid else indices_grid[..., 0]
    indices = generate_frequency_grid(theta, indices_grid.shape[1], dim)
    frequencies = _generate_frequencies(indices, indices_grid, max_positions)
    expected_frequency_count = dim // ROTARY_PAIR_SIZE
    pad_size = expected_frequency_count - frequencies.shape[-1]
    return _split_frequency_components(frequencies, pad_size, attention_heads)
