import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.transformer.adaln import ada_zero, get_ada_values, post_self_attention


def test_get_ada_values_matches_reference_layout() -> None:
    table = np.arange(24, dtype=np.float32).reshape(6, 4)
    timestep = np.arange(48, dtype=np.float32).reshape(1, 2, 24)

    values = get_ada_values(mx.array(table), mx.array(timestep), slice(1, 4))

    expected = timestep.reshape(1, 2, 6, 4)[:, :, 1:4] + table[None, None, 1:4]
    assert len(values) == 3
    for index, value in enumerate(values):
        np.testing.assert_array_equal(np.asarray(value), expected[:, :, index])


def test_get_ada_values_rejects_wrong_timestep_width() -> None:
    with pytest.raises(LaraError) as raised:
        get_ada_values(mx.zeros((6, 4)), mx.zeros((1, 2, 20)), slice(0, 3))

    assert raised.value.code == "LARA-TENSOR-006"


def test_ada_zero_and_post_attention_match_reference_equations() -> None:
    x = np.array([[[1.0, -2.0, 3.0, -4.0]]], dtype=np.float32)
    scale = np.full_like(x, 0.25)
    shift = np.full_like(x, -0.5)
    attention = np.full_like(x, 0.75)
    gate = np.full_like(x, 0.2)

    normalized = ada_zero(mx.array(x), mx.array(scale), mx.array(shift))
    updated, post_norm = post_self_attention(mx.array(x), mx.array(attention), mx.array(gate))

    rms = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(np.asarray(normalized), rms * (1 + scale) + shift, rtol=1e-6, atol=1e-6)
    expected_updated = x + attention * gate
    expected_post = expected_updated / np.sqrt(np.mean(expected_updated**2, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(np.asarray(updated), expected_updated, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(post_norm), expected_post, rtol=1e-6, atol=1e-6)
