from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.diffusion_blocks import NABlock, NABlockConfig

GOLDEN_PATH = Path("golden/cuda/diffvae_na_block_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 3e-2
BF16_RELATIVE_TOLERANCE = 3e-2
WEIGHT_PREFIX = "diffvae_na_block_weight__"
DIM = 16
HEAD_DIM = 16
KERNEL_SIZE = (3, 3, 3)


def test_diffvae_na_block_matches_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA Diffusion VAE NABlock golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    block = NABlock(NABlockConfig(dim=DIM, kernel_size=KERNEL_SIZE, head_dim=HEAD_DIM))
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
