from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.video_vae.conv_decoder import ConvVideoDecoder, ConvVideoDecoderConfig, DecoderBlockConfig

GOLDEN_PATH = Path("golden/cuda/conv_vae_decoder_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "conv_vae_decoder_weight__"
SOURCE_STD_KEY = "per_channel_statistics.std-of-means"
TARGET_STD_KEY = "per_channel_statistics.std_of_means"
SOURCE_MEAN_KEY = "per_channel_statistics.mean-of-means"
TARGET_MEAN_KEY = "per_channel_statistics.mean_of_means"


def _golden() -> dict[str, np.ndarray]:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA Conv VAE decoder golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        return {key: archive[key] for key in archive.files}


def _mlx_key(source_key: str) -> str:
    return source_key.replace(SOURCE_STD_KEY, TARGET_STD_KEY).replace(SOURCE_MEAN_KEY, TARGET_MEAN_KEY)


def test_conv_vae_decoder_subset_matches_cuda_bf16() -> None:
    golden = _golden()
    decoder = ConvVideoDecoder(
        ConvVideoDecoderConfig(
            in_channels=golden["input"].shape[1],
            out_channels=golden["output"].shape[1],
            base_channels=4,
            patch_size=2,
            decoder_blocks=(
                DecoderBlockConfig(name="res_x_y", multiplier=2),
                DecoderBlockConfig(name="compress_all", multiplier=2),
            ),
        )
    )
    decoder.load_weights(
        [
            (_mlx_key(name.removeprefix(WEIGHT_PREFIX)), mx.array(value, dtype=mx.bfloat16))
            for name, value in golden.items()
            if name.startswith(WEIGHT_PREFIX)
        ]
    )
    result = decoder(mx.array(golden["input"], dtype=mx.bfloat16))
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
