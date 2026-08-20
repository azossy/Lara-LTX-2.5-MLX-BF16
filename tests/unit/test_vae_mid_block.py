import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.resnet import UNetMidBlock3D, UNetMidBlock3DConfig


def test_timestep_mid_block_preserves_shape() -> None:
    block = UNetMidBlock3D(UNetMidBlock3DConfig(in_channels=4, num_layers=2, timestep_conditioning=True))
    source = mx.ones((1, 4, 2, 3, 3), dtype=mx.bfloat16)
    result = block(source, timestep=mx.array([0.05], dtype=mx.bfloat16))
    assert result.shape == source.shape


def test_timestep_mid_block_requires_scalar_conditioning() -> None:
    block = UNetMidBlock3D(UNetMidBlock3DConfig(in_channels=4, num_layers=1, timestep_conditioning=True))
    with pytest.raises(LaraError, match="LARA-TENSOR-014"):
        block(mx.zeros((1, 4, 1, 2, 2)))
