from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.attention import AttnBlock3D

GOLDEN_PATH = Path("golden/cuda/vae_attention_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "vae_attention_weight__"


def _golden() -> dict[str, np.ndarray]:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA VAE attention golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        return {key: archive[key] for key in archive.files}


def test_vae_attention_matches_cuda_bf16() -> None:
    golden = _golden()
    block = AttnBlock3D(in_channels=golden["input"].shape[1])
    block.load_weights(
        [
            (name.removeprefix(WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
            for name, value in golden.items()
            if name.startswith(WEIGHT_PREFIX)
        ]
    )
    result = block(mx.array(golden["input"], dtype=mx.bfloat16))
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
