from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.video_vae.spatial_upscaler import (
    LatentSpatialUpscaler,
    LatentSpatialUpscalerConfig,
    SpatialPixelShuffle,
)


def test_spatial_pixel_shuffle_matches_channel_factor_order() -> None:
    value = mx.arange(16, dtype=mx.float32).reshape(1, 4, 1, 2, 2)

    actual = SpatialPixelShuffle(scale=2)(value)

    expected = np.array([[[[[0, 4, 1, 5], [8, 12, 9, 13], [2, 6, 3, 7], [10, 14, 11, 15]]]]])
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_small_spatial_upscaler_preserves_time_and_doubles_space() -> None:
    model = LatentSpatialUpscaler(
        LatentSpatialUpscalerConfig(
            in_channels=2,
            mid_channels=4,
            num_blocks_per_stage=1,
            groups=2,
            spatial_scale=2,
        )
    )

    output = model(mx.ones((1, 2, 3, 4, 5), dtype=mx.bfloat16))
    mx.eval(output)

    assert output.shape == (1, 2, 3, 8, 10)
    np.testing.assert_array_equal(np.asarray(output.astype(mx.float32)), 0.0)


def test_spatial_upscaler_rejects_invalid_channel_layout() -> None:
    model = LatentSpatialUpscaler(
        LatentSpatialUpscalerConfig(
            in_channels=2,
            mid_channels=4,
            num_blocks_per_stage=1,
            groups=2,
            spatial_scale=2,
        )
    )

    with pytest.raises(LaraError) as error:
        model(mx.zeros((1, 3, 2, 2, 2)))

    assert error.value.code == "LARA-TENSOR-023"
