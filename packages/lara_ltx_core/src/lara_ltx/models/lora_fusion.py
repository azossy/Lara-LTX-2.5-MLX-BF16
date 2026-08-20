"""One-weight-at-a-time MLX LoRA fusion for the two-stage LTX runtime."""

from __future__ import annotations

import math

import mlx.core as mx

from lara_ltx.errors import LaraError


def fuse_lora_weight(base: mx.array, a: mx.array, b: mx.array, strength: float) -> mx.array:
    """Fuse ``(B * strength) @ A`` into one BF16 base matrix.

    The caller releases ``a`` and ``b`` before moving to the next base tensor;
    this function never constructs a second transformer state dictionary.
    """

    if not math.isfinite(strength) or strength < 0:
        raise LaraError("LARA-MODEL-027", details={"value": strength})
    if (
        base.ndim != 2
        or a.ndim != 2
        or b.ndim != 2
        or a.shape[0] != b.shape[1]
        or base.shape != (b.shape[0], a.shape[1])
    ):
        raise LaraError("LARA-MODEL-026", details={"key": "runtime_lora_pair"})
    scaled_b = b * mx.array(strength, dtype=b.dtype)
    delta = mx.matmul(scaled_b, a)
    return (delta + base).astype(base.dtype)
