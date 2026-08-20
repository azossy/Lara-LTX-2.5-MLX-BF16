from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.diffusion_decoder import (
    DiffusionVideoDecoder,
    DiffusionVideoDecoderConfig,
    Stage4TilingConfig,
    Stage5TilingConfig,
)

GOLDEN_PATH = Path("golden/cuda/diffvae_decoder_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 8e-2
BF16_RELATIVE_TOLERANCE = 8e-2
WEIGHT_PREFIX = "diffvae_decoder_weight__"
SOURCE_STD_KEY = "per_channel_statistics.std-of-means"
TARGET_STD_KEY = "per_channel_statistics.std_of_means"
SOURCE_MEAN_KEY = "per_channel_statistics.mean-of-means"
TARGET_MEAN_KEY = "per_channel_statistics.mean_of_means"


def _mlx_key(source_key: str) -> str:
    return source_key.replace(SOURCE_STD_KEY, TARGET_STD_KEY).replace(SOURCE_MEAN_KEY, TARGET_MEAN_KEY)


def test_complete_untiled_diffvae_decoder_matches_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA Diffusion VAE decoder golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    decoder = DiffusionVideoDecoder(
        DiffusionVideoDecoderConfig(
            in_channels=4,
            out_channels=3,
            patch_size=2,
            head_dim=16,
            stage_channels=(16, 16, 16, 16, 16),
            stage_depths=(1, 1, 1, 1, 2),
            stage_kernels=((3, 3, 3),) * 5,
            upsamples=(((1, 2, 2), 1), ((1, 1, 1), 1), ((1, 1, 1), 1), ((1, 1, 1), 1)),
            stage5_kernel=(3, 3, 3),
            timestep_embedding_dim=16,
            default_num_inference_steps=2,
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
    result = decoder.decode_untiled(
        mx.array(golden["latent"], dtype=mx.bfloat16),
        mx.array(golden["initial_noise"], dtype=mx.bfloat16),
        timesteps=mx.array(golden["timesteps"], dtype=mx.float32),
    )
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )

    tiled_result = decoder.decode_stage5_tiled(
        mx.array(golden["latent"], dtype=mx.bfloat16),
        mx.array(golden["initial_noise"], dtype=mx.bfloat16),
        Stage5TilingConfig(tile_shape=(2, 3, 3)),
        timesteps=mx.array(golden["timesteps"], dtype=mx.float32),
    )
    np.testing.assert_allclose(
        np.asarray(tiled_result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(tiled_result.astype(mx.float32)),
        np.asarray(result.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-2,
    )

    fully_tiled_result = decoder.decode_tiled(
        mx.array(golden["latent"], dtype=mx.bfloat16),
        mx.array(golden["initial_noise"], dtype=mx.bfloat16),
        Stage4TilingConfig(tile_shape=(2, 3, 3)),
        Stage5TilingConfig(tile_shape=(2, 3, 3)),
        timesteps=mx.array(golden["timesteps"], dtype=mx.float32),
    )
    np.testing.assert_allclose(
        np.asarray(fully_tiled_result.astype(mx.float32)),
        np.asarray(result.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
