from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.diffusion_blocks import CombinedDiffusionNABlock, CombinedDiffusionNABlockConfig

GOLDEN_PATH = Path("golden/cuda/diffvae_combined_block_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 4e-2
BF16_RELATIVE_TOLERANCE = 4e-2
WEIGHT_PREFIX = "diffvae_combined_block_weight__"


def test_diffvae_combined_block_matches_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA Diffusion VAE combined block golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    block = CombinedDiffusionNABlock(
        CombinedDiffusionNABlockConfig(
            dim=16,
            context_channels=8,
            kernel_size=(3, 3, 3),
            head_dim=16,
        )
    )
    block.load_weights(
        [
            (name.removeprefix(WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
            for name, value in golden.items()
            if name.startswith(WEIGHT_PREFIX)
        ]
    )
    stacked_modulation = mx.array(golden["modulation"], dtype=mx.bfloat16)
    modulation = tuple(stacked_modulation[:, index] for index in range(stacked_modulation.shape[1]))
    result = block(mx.array(golden["context_and_x"], dtype=mx.bfloat16), modulation)
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
