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


def test_rms_norm_uses_float32_calculation_and_restores_bfloat16() -> None:
    values = mx.array([[[0.03125, -9.75, 0.5, 4.125]]], dtype=mx.bfloat16)
    weights = mx.array([0.75, 1.25, 0.875, 1.5], dtype=mx.bfloat16)

    result = rms_norm(values, weights, calculation_dtype=mx.float32)
    values_fp32 = values.astype(mx.float32)
    weights_fp32 = weights.astype(mx.float32)
    expected = values_fp32 * mx.rsqrt(mx.mean(mx.square(values_fp32), axis=-1, keepdims=True) + 1e-6)
    expected = (expected * weights_fp32).astype(mx.bfloat16)
    mx.eval(result, expected)

    assert result.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.asarray(result.astype(mx.float32)), np.asarray(expected.astype(mx.float32)))


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
