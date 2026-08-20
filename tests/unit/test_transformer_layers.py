import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.transformer.layers import gelu_approx, rms_norm


def test_rms_norm_matches_reference_equation() -> None:
    values = np.array([[[1.0, -2.0, 3.0, -4.0]]], dtype=np.float32)
    weights = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)

    result = rms_norm(mx.array(values), mx.array(weights), eps=1e-6)

    expected = values / np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + 1e-6) * weights
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6, atol=1e-6)


def test_tanh_gelu_matches_pytorch_formula() -> None:
    values = np.linspace(-3.0, 3.0, 25, dtype=np.float32)

    result = gelu_approx(mx.array(values))

    coefficient = math.sqrt(2.0 / math.pi)
    expected = 0.5 * values * (1.0 + np.tanh(coefficient * (values + 0.044715 * values**3)))
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6, atol=1e-6)


def test_tanh_gelu_preserves_bfloat16_runtime_dtype() -> None:
    result = gelu_approx(mx.linspace(-3.0, 3.0, 25).astype(mx.bfloat16))
    mx.eval(result)

    assert result.dtype == mx.bfloat16
    assert np.isfinite(np.asarray(result.astype(mx.float32))).all()
