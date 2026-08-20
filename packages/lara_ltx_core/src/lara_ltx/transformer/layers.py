"""Small MLX layers shared by the LTX transformer port."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

DEFAULT_NORM_EPSILON = 1e-6
GELU_TANH_COEFFICIENT = 0.044715


def rms_norm(x: mx.array, weight: mx.array | None = None, *, eps: float = DEFAULT_NORM_EPSILON) -> mx.array:
    """Normalize the final dimension with PyTorch RMSNorm semantics."""

    normalized = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    return normalized if weight is None else normalized * weight


def gelu_approx(x: mx.array) -> mx.array:
    """Tanh GELU with CUDA-compatible FP32 activation evaluation."""

    output_dtype = x.dtype
    x = x.astype(mx.float32)
    coefficient = math.sqrt(2.0 / math.pi)
    result = 0.5 * x * (1.0 + mx.tanh(coefficient * (x + GELU_TANH_COEFFICIENT * mx.power(x, 3))))
    return result.astype(output_dtype)


class GELUApprox(nn.Module):
    """Linear projection followed by the checkpoint-compatible GELU."""

    def __init__(self, dim_in: int, dim_out: int, *, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return gelu_approx(self.proj(x))


class FeedForward(nn.Module):
    """Upstream LTX MLP with its state-dict-compatible module layout."""

    def __init__(self, dim: int, dim_out: int, *, mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.net = [GELUApprox(dim, inner_dim, bias=bias), nn.Identity(), nn.Linear(inner_dim, dim_out, bias=bias)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.net:
            x = layer(x)
        return x
