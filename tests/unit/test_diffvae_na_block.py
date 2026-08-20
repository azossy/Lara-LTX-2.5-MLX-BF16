import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.diffusion_blocks import NABlock, NABlockConfig


def test_diffvae_na_block_preserves_shape_and_produces_finite_values() -> None:
    block = NABlock(NABlockConfig(dim=16, kernel_size=(3, 3, 3), head_dim=16, mask_element_budget=20_000))
    source = mx.arange(1 * 3 * 3 * 4 * 16, dtype=mx.float32).reshape(1, 3, 3, 4, 16)
    result = block(source)
    assert result.shape == source.shape
    assert np.isfinite(np.asarray(result)).all()


def test_diffvae_na_block_rejects_grid_below_kernel() -> None:
    block = NABlock(NABlockConfig(dim=16, kernel_size=(3, 3, 3), head_dim=16))
    with pytest.raises(LaraError, match="LARA-TENSOR-007"):
        block(mx.zeros((1, 2, 3, 3, 16)))
