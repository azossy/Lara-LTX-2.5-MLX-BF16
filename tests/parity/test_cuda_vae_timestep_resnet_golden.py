from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.resnet import ResnetBlock3D, ResnetBlock3DConfig

GOLDEN_PATH = Path("golden/cuda/vae_timestep_resnet_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "vae_timestep_resnet_weight__"


def _golden() -> dict[str, np.ndarray]:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA timestep VAE ResNet golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        return {key: archive[key] for key in archive.files}


def test_timestep_vae_resnet_matches_cuda_bf16() -> None:
    golden = _golden()
    channels = golden["input"].shape[1]
    block = ResnetBlock3D(
        ResnetBlock3DConfig(
            in_channels=channels,
            out_channels=channels,
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


def test_timestep_vae_resnet_requires_conditioning() -> None:
    block = ResnetBlock3D(ResnetBlock3DConfig(in_channels=4, out_channels=4, timestep_conditioning=True))
    with pytest.raises(LaraError, match="LARA-TENSOR-014"):
        block(mx.zeros((1, 4, 1, 2, 2)))
