from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.transformer.output import TransformerOutputConfig, TransformerOutputHead

HIDDEN_DIMENSION = 8
OUTPUT_CHANNELS = 4
TOKEN_COUNT = 3


def test_transformer_output_matches_explicit_modulation_order() -> None:
    head = TransformerOutputHead(
        TransformerOutputConfig(hidden_dimension=HIDDEN_DIMENSION, output_channels=OUTPUT_CHANNELS)
    )
    x = mx.arange(TOKEN_COUNT * HIDDEN_DIMENSION, dtype=mx.float32).reshape(1, TOKEN_COUNT, HIDDEN_DIMENSION)
    timestep = mx.full(x.shape, 0.25, dtype=mx.float32)
    table = mx.stack([mx.full((HIDDEN_DIMENSION,), -0.5), mx.full((HIDDEN_DIMENSION,), 0.75)])
    weight = mx.arange(OUTPUT_CHANNELS * HIDDEN_DIMENSION, dtype=mx.float32).reshape(OUTPUT_CHANNELS, HIDDEN_DIMENSION)
    bias = mx.arange(OUTPUT_CHANNELS, dtype=mx.float32)
    head.load_weights((("scale_shift_table", table), ("proj_out.weight", weight), ("proj_out.bias", bias)))

    result = head(x, timestep)
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * mx.rsqrt(variance + head.config.norm_epsilon)
    expected = (normalized * 2.0 - 0.25) @ mx.transpose(weight) + bias
    mx.eval(result, expected)

    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_transformer_output_rejects_incompatible_timestep_shape() -> None:
    head = TransformerOutputHead(
        TransformerOutputConfig(hidden_dimension=HIDDEN_DIMENSION, output_channels=OUTPUT_CHANNELS)
    )
    with pytest.raises(LaraError) as raised:
        head(mx.zeros((1, TOKEN_COUNT, HIDDEN_DIMENSION)), mx.zeros((1, 1, HIDDEN_DIMENSION)))

    assert raised.value.code == "LARA-TENSOR-020"
