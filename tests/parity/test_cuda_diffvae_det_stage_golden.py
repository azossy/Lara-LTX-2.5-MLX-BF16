from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.diffusion_blocks import DeterministicStage, DeterministicStageConfig

GOLDEN_PATH = Path("golden/cuda/diffvae_det_stage_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 4e-2
BF16_RELATIVE_TOLERANCE = 4e-2
WEIGHT_PREFIX = "diffvae_det_stage_weight__"


def test_diffvae_deterministic_stage_matches_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA Diffusion VAE deterministic stage golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    stage = DeterministicStage(
        DeterministicStageConfig(
            channels=16,
            depth=2,
            kernel_size=(3, 3, 3),
            head_dim=16,
            upsample_stride=(1, 2, 2),
            out_channels_reduction_factor=2,
        )
    )
    stage.load_weights(
        [
            (name.removeprefix(WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
            for name, value in golden.items()
            if name.startswith(WEIGHT_PREFIX)
        ]
    )
    result = stage(mx.array(golden["input"], dtype=mx.bfloat16))
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
