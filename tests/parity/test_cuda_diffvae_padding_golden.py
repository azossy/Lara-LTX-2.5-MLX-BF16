from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.diffusion_decoder import DiffusionVideoDecoder, DiffusionVideoDecoderConfig

GOLDEN_PATH = Path("golden/cuda/diffvae_padding_bf16_cuda.npz")
WEIGHT_PREFIX = "diffvae_padding_weight__"


def _mlx_key(source_key: str) -> str:
    return source_key.replace("std-of-means", "std_of_means").replace("mean-of-means", "mean_of_means")


def test_production_padding_and_crop_matches_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA Diffusion VAE padding golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    decoder = DiffusionVideoDecoder(
        DiffusionVideoDecoderConfig(
            in_channels=4,
            out_channels=3,
            patch_size=4,
            head_dim=16,
            stage_channels=(16, 16, 16, 16, 16),
            stage_depths=(1, 1, 1, 1, 1),
            stage_kernels=((3, 3, 3),) * 5,
            upsamples=(((1, 2, 2), 1), ((2, 1, 1), 1), ((2, 2, 2), 1), ((2, 2, 2), 1)),
            stage5_kernel=(3, 3, 3),
            timestep_embedding_dim=16,
            default_num_inference_steps=1,
            model_output_type="v",
        )
    )
    decoder.load_weights(
        [
            (_mlx_key(name.removeprefix(WEIGHT_PREFIX)), mx.array(value, dtype=mx.bfloat16))
            for name, value in golden.items()
            if name.startswith(WEIGHT_PREFIX)
        ]
    )
    result = decoder.decode(
        mx.array(golden["latent"], dtype=mx.bfloat16),
        mx.array(golden["initial_noise"], dtype=mx.bfloat16),
    )
    np.testing.assert_allclose(np.asarray(result.astype(mx.float32)), golden["output"], rtol=8e-2, atol=8e-2)
