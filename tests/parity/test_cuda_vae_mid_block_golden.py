from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.resnet import UNetMidBlock3D, UNetMidBlock3DConfig

GOLDEN_PATH = Path("golden/cuda/vae_mid_block_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "vae_mid_block_weight__"


def test_timestep_vae_mid_block_matches_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA timestep VAE mid-block golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    block = UNetMidBlock3D(
        UNetMidBlock3DConfig(
            in_channels=golden["input"].shape[1],
            num_layers=2,
            timestep_conditioning=True,
        )
    )
    block.load_weights(
        [
            (name.removeprefix(WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
            for name, value in golden.items()
            if name.startswith(WEIGHT_PREFIX)
        ]
    )
    result = block(
        mx.array(golden["input"], dtype=mx.bfloat16),
        timestep=mx.array(golden["timestep"], dtype=mx.bfloat16),
    )
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
