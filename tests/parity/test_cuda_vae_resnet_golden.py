from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.resnet import ResnetBlock3D, ResnetBlock3DConfig

PIXEL_NORM_GOLDEN_PATH = Path("golden/cuda/vae_resnet_bf16_cuda.npz")
GROUP_NORM_GOLDEN_PATH = Path("golden/cuda/vae_resnet_group_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "vae_resnet_weight__"
PIXEL_NORM_NAME = "pixel_norm"
GROUP_NORM_NAME = "group_norm"
GROUP_COUNT = 1


def _golden(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        pytest.skip("CUDA VAE ResNet golden artifact has not been generated")
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.fixture(scope="module")
def pixel_norm_golden() -> dict[str, np.ndarray]:
    return _golden(PIXEL_NORM_GOLDEN_PATH)


@pytest.fixture(scope="module")
def group_norm_golden() -> dict[str, np.ndarray]:
    return _golden(GROUP_NORM_GOLDEN_PATH)


@pytest.mark.parametrize("norm_layer", (PIXEL_NORM_NAME, GROUP_NORM_NAME))
def test_vae_resnet_block_matches_cuda_bf16(
    norm_layer: str,
    pixel_norm_golden: dict[str, np.ndarray],
    group_norm_golden: dict[str, np.ndarray],
) -> None:
    golden = pixel_norm_golden if norm_layer == PIXEL_NORM_NAME else group_norm_golden
    block = ResnetBlock3D(
        ResnetBlock3DConfig(
            in_channels=golden["input"].shape[1],
            out_channels=golden["output"].shape[1],
            groups=GROUP_COUNT,
            norm_layer=norm_layer,
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
