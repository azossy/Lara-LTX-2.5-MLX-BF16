from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.sampling import DepthToSpaceUpsample, DepthToSpaceUpsampleConfig

GOLDEN_PATH = Path("golden/cuda/vae_depth_to_space_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "vae_depth_to_space_weight__"
STRIDE = (2, 2, 2)
OUT_CHANNELS_REDUCTION_FACTOR = 1


def _golden() -> dict[str, np.ndarray]:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA VAE depth-to-space golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        return {key: archive[key] for key in archive.files}


def test_depth_to_space_upsample_matches_cuda_bf16() -> None:
    golden = _golden()
    block = DepthToSpaceUpsample(
        DepthToSpaceUpsampleConfig(
            in_channels=golden["input"].shape[1],
            stride=STRIDE,
            residual=True,
            out_channels_reduction_factor=OUT_CHANNELS_REDUCTION_FACTOR,
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


def test_depth_to_space_rejects_invalid_residual_channel_layout() -> None:
    with pytest.raises(LaraError, match="LARA-TENSOR-010"):
        DepthToSpaceUpsample(DepthToSpaceUpsampleConfig(in_channels=3, stride=STRIDE, residual=True))
