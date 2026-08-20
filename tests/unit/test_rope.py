import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.transformer.rope import apply_split_rotary_emb


def test_split_rotary_matches_reference_equations() -> None:
    values = np.arange(16, dtype=np.float32).reshape(1, 2, 8)
    angles = np.linspace(0.0, 0.7, 8, dtype=np.float32).reshape(1, 1, 2, 4)
    cos_freqs = np.cos(angles)
    sin_freqs = np.sin(angles)

    result = apply_split_rotary_emb(mx.array(values), mx.array(cos_freqs), mx.array(sin_freqs))

    headed = values.reshape(1, 2, 1, 8).swapaxes(1, 2)
    first, second = np.split(headed, 2, axis=-1)
    expected = np.concatenate((first * cos_freqs - second * sin_freqs, second * cos_freqs + first * sin_freqs), -1)
    expected = expected.swapaxes(1, 2).reshape(values.shape)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6, atol=1e-6)


def test_split_rotary_restores_bfloat16_after_float32_calculation() -> None:
    values = mx.array([[[[0.03125, -9.75, 0.5, 4.125]]]], dtype=mx.bfloat16)
    cos_freqs = mx.array([[[[0.75, 0.875]]]], dtype=mx.bfloat16)
    sin_freqs = mx.array([[[[0.625, -0.5]]]], dtype=mx.bfloat16)

    result = apply_split_rotary_emb(values, cos_freqs, sin_freqs)
    values_fp32 = values.astype(mx.float32)
    expected = mx.concatenate(
        (
            values_fp32[..., :2] * cos_freqs.astype(mx.float32) - values_fp32[..., 2:] * sin_freqs.astype(mx.float32),
            values_fp32[..., 2:] * cos_freqs.astype(mx.float32) + values_fp32[..., :2] * sin_freqs.astype(mx.float32),
        ),
        axis=-1,
    ).astype(mx.bfloat16)
    mx.eval(result, expected)

    assert result.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.asarray(result.astype(mx.float32)), np.asarray(expected.astype(mx.float32)))


def test_split_rotary_rejects_frequency_shape_mismatch() -> None:
    with pytest.raises(LaraError) as raised:
        apply_split_rotary_emb(mx.zeros((1, 1, 4)), mx.zeros((1, 1, 1, 2)), mx.zeros((1, 1, 1, 1)))

    assert raised.value.code == "LARA-TENSOR-001"
