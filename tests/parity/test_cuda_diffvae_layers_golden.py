from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.diffusion_layers import (
    AdaLNZero,
    LinearPixelShuffleUpsample,
    LinearPixelShuffleUpsampleConfig,
    SwiGLU,
)

GOLDEN_PATH = Path("golden/cuda/diffvae_layers_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
DIM = 8
HIDDEN_DIM = 32
TIMESTEP_EMBEDDING_DIM = 16


def _weights(golden: dict[str, np.ndarray], prefix: str) -> list[tuple[str, mx.array]]:
    return [
        (name.removeprefix(prefix), mx.array(value, dtype=mx.bfloat16))
        for name, value in golden.items()
        if name.startswith(prefix)
    ]


def test_shared_diffvae_layers_match_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA Diffusion VAE layer golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    upsample = LinearPixelShuffleUpsample(
        LinearPixelShuffleUpsampleConfig(in_channels=DIM, stride=(2, 2, 2), out_channels_reduction_factor=2)
    )
    adaln = AdaLNZero(DIM, TIMESTEP_EMBEDDING_DIM)
    swiglu = SwiGLU(DIM, HIDDEN_DIM)
    upsample.load_weights(_weights(golden, "upsample_weight__"))
    adaln.load_weights(_weights(golden, "adaln_weight__"))
    swiglu.load_weights(_weights(golden, "swiglu_weight__"))

    upsample_output = upsample(mx.array(golden["upsample_input"], dtype=mx.bfloat16))
    adaln_output = mx.stack(adaln(mx.array(golden["timestep_input"], dtype=mx.bfloat16)), axis=1)
    swiglu_output = swiglu(mx.array(golden["swiglu_input"], dtype=mx.bfloat16))
    for actual, expected in (
        (upsample_output, golden["upsample_output"]),
        (adaln_output, golden["adaln_output"]),
        (swiglu_output, golden["swiglu_output"]),
    ):
        np.testing.assert_allclose(
            np.asarray(actual.astype(mx.float32)),
            expected,
            rtol=BF16_RELATIVE_TOLERANCE,
            atol=BF16_ABSOLUTE_TOLERANCE,
        )
