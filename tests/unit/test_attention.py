import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.transformer.attention import scaled_dot_product_attention

FUSED_SDPA_RTOL = 2e-3
FUSED_SDPA_ATOL = 2e-3


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def test_fused_attention_matches_numpy_reference() -> None:
    generator = np.random.default_rng(7)
    query = generator.normal(size=(2, 3, 8)).astype(np.float32)
    key = generator.normal(size=(2, 5, 8)).astype(np.float32)
    value = generator.normal(size=(2, 5, 8)).astype(np.float32)
    heads = 2
    head_dim = query.shape[-1] // heads

    result = scaled_dot_product_attention(mx.array(query), mx.array(key), mx.array(value), heads)

    q_heads = query.reshape(2, 3, heads, head_dim).swapaxes(1, 2)
    k_heads = key.reshape(2, 5, heads, head_dim).swapaxes(1, 2)
    v_heads = value.reshape(2, 5, heads, head_dim).swapaxes(1, 2)
    weights = _softmax(q_heads @ k_heads.swapaxes(-1, -2) / math.sqrt(head_dim))
    expected = (weights @ v_heads).swapaxes(1, 2).reshape(2, 3, 8)
    # MLX's fused Metal kernel uses a numerically approximate fused reduction,
    # while keeping the softmax accumulator in float32. CUDA parity will freeze
    # the release tolerance; this unit bound catches scaling/layout mistakes.
    np.testing.assert_allclose(np.asarray(result), expected, rtol=FUSED_SDPA_RTOL, atol=FUSED_SDPA_ATOL)
