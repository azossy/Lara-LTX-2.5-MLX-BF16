"""Memory-efficient MLX attention matching the LTX-2.5 projection layout."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .layers import DEFAULT_NORM_EPSILON, NORM_CALCULATION_DTYPE, rms_norm
from .rope import LTXRopeType, apply_rotary_emb


class RMSNorm(nn.Module):
    """Learnable RMSNorm with a PyTorch-compatible ``weight`` parameter."""

    def __init__(self, dimensions: int, *, eps: float = DEFAULT_NORM_EPSILON) -> None:
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(
            x,
            self.weight,
            eps=self.eps,
            calculation_dtype=NORM_CALCULATION_DTYPE,
        )


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    heads: int,
    mask: mx.array | None = None,
) -> mx.array:
    """Run MLX fused SDPA without materializing the complete score matrix."""

    batch = query.shape[0]
    head_dim = query.shape[-1] // heads
    query = mx.swapaxes(query.reshape(batch, -1, heads, head_dim), 1, 2)
    key = mx.swapaxes(key.reshape(batch, -1, heads, head_dim), 1, 2)
    value = mx.swapaxes(value.reshape(batch, -1, heads, head_dim), 1, 2)
    if mask is not None:
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, 0)
        if mask.ndim == 3:
            mask = mx.expand_dims(mask, 1)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=1.0 / math.sqrt(head_dim),
        mask=mask,
    )
    return mx.swapaxes(output, 1, 2).reshape(batch, -1, heads * head_dim)


class Attention(nn.Module):
    """LTX self/cross attention with QK RMSNorm, RoPE, and optional head gates."""

    def __init__(
        self,
        query_dim: int,
        *,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = DEFAULT_NORM_EPSILON,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = query_dim if context_dim is None else context_dim
        self.heads = heads
        self.dim_head = dim_head
        self.rope_type = rope_type
        self.q_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = RMSNorm(inner_dim, eps=norm_eps)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_gate_logits = nn.Linear(query_dim, heads, bias=True) if apply_gated_attention else None
        self.to_out = [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]

    def __call__(
        self,
        x: mx.array,
        *,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        pe: tuple[mx.array, mx.array] | None = None,
        k_pe: tuple[mx.array, mx.array] | None = None,
        perturbation_mask: mx.array | None = None,
        all_perturbed: bool = False,
    ) -> mx.array:
        context = x if context is None else context
        value = self.to_v(context)
        if all_perturbed:
            output = value
        else:
            query = self.q_norm(self.to_q(x))
            key = self.k_norm(self.to_k(context))
            if pe is not None:
                query = apply_rotary_emb(query, pe, self.rope_type)
                key = apply_rotary_emb(key, pe if k_pe is None else k_pe, self.rope_type)
            output = scaled_dot_product_attention(query, key, value, self.heads, mask)
            if perturbation_mask is not None:
                output = output * perturbation_mask + value * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            batch, tokens = output.shape[:2]
            gates = 2.0 * mx.sigmoid(self.to_gate_logits(x))
            output = output.reshape(batch, tokens, self.heads, self.dim_head) * mx.expand_dims(gates, -1)
            output = output.reshape(batch, tokens, self.heads * self.dim_head)

        for layer in self.to_out:
            output = layer(output)
        return output
