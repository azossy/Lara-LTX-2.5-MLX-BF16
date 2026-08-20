"""Adaptive normalization functions shared by LTX transformer blocks."""

from __future__ import annotations

import mlx.core as mx

from lara_ltx.errors import LaraError

from .layers import DEFAULT_NORM_EPSILON, rms_norm

ADALN_BASE_PARAMETER_COUNT = 6
ADALN_CROSS_ATTENTION_PARAMETER_COUNT = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    """Return the checkpoint modulation count for one transformer block."""

    return ADALN_BASE_PARAMETER_COUNT + (ADALN_CROSS_ATTENTION_PARAMETER_COUNT if cross_attention_adaln else 0)


def get_ada_values(
    scale_shift_table: mx.array,
    timestep: mx.array,
    indices: slice,
) -> tuple[mx.array, ...]:
    """Add static block modulation values to per-token timestep embeddings."""

    if scale_shift_table.ndim != 2 or timestep.ndim != 3:
        raise LaraError(
            "LARA-TENSOR-006",
            details={"table_shape": tuple(scale_shift_table.shape), "timestep_shape": tuple(timestep.shape)},
        )
    parameter_count, feature_dim = scale_shift_table.shape
    expected_features = parameter_count * feature_dim
    if timestep.shape[-1] != expected_features:
        raise LaraError(
            "LARA-TENSOR-006",
            details={"table_shape": tuple(scale_shift_table.shape), "timestep_shape": tuple(timestep.shape)},
        )
    batch, tokens = timestep.shape[:2]
    modulation = timestep.reshape(batch, tokens, parameter_count, feature_dim)
    modulation = modulation[:, :, indices, :] + scale_shift_table[indices][None, None, :, :]
    return tuple(modulation[:, :, index, :] for index in range(modulation.shape[2]))


def ada_zero(
    x: mx.array,
    scale: mx.array,
    shift: mx.array,
    *,
    eps: float = DEFAULT_NORM_EPSILON,
) -> mx.array:
    """Apply the upstream RMSNorm plus adaptive scale/shift operation."""

    return rms_norm(x, eps=eps) * (1 + scale) + shift


def post_self_attention(
    x: mx.array,
    attention_output: mx.array,
    gate: mx.array,
    *,
    norm_weight: mx.array | None = None,
    eps: float = DEFAULT_NORM_EPSILON,
) -> tuple[mx.array, mx.array]:
    """Apply the gated residual and return both residual and normalized states."""

    updated = x + attention_output * gate
    return updated, rms_norm(updated, norm_weight, eps=eps)
