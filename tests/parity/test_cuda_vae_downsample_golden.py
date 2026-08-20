from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.sampling import SpaceToDepthDownsample, SpaceToDepthDownsampleConfig

GOLDEN_PATH = Path("golden/cuda/vae_space_to_depth_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "vae_space_to_depth_weight__"
STRIDE = (2, 2, 2)
OUTPUT_CHANNELS = 8


def _golden() -> dict[str, np.ndarray]:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA VAE space-to-depth golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        return {key: archive[key] for key in archive.files}


def test_space_to_depth_downsample_matches_cuda_bf16() -> None:
    golden = _golden()
    block = SpaceToDepthDownsample(
        SpaceToDepthDownsampleConfig(
            in_channels=golden["input"].shape[1],
            out_channels=OUTPUT_CHANNELS,
            stride=STRIDE,
        )
    )
    weights = [
        (name.removeprefix(WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
        for name, value in golden.items()
        if name.startswith(WEIGHT_PREFIX)
    ]
    block.load_weights(weights)
    result = block(mx.array(golden["input"], dtype=mx.bfloat16), causal=True)
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_space_to_depth_rejects_invalid_channel_layout() -> None:
    with pytest.raises(LaraError, match="LARA-TENSOR-011"):
        SpaceToDepthDownsample(SpaceToDepthDownsampleConfig(in_channels=3, out_channels=16, stride=STRIDE))
