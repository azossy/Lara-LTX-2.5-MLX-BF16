import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.attention import AttnBlock3D


def test_vae_attention_preserves_shape_and_residual_with_zero_projection() -> None:
    block = AttnBlock3D(in_channels=4)
    source = mx.arange(1 * 4 * 2 * 2 * 3, dtype=mx.float32).reshape(1, 4, 2, 2, 3)
    result = block(source)
    assert result.shape == source.shape
    np.testing.assert_array_equal(np.asarray(result), np.asarray(source))
