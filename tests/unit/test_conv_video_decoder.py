import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.conv_decoder import ConvVideoDecoder, ConvVideoDecoderConfig, DecoderBlockConfig

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
BASE_CHANNELS = 4
PATCH_SIZE = 2


def _decoder() -> ConvVideoDecoder:
    return ConvVideoDecoder(
        ConvVideoDecoderConfig(
            in_channels=INPUT_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            base_channels=BASE_CHANNELS,
            patch_size=PATCH_SIZE,
            decoder_blocks=(
                DecoderBlockConfig(name="res_x_y", multiplier=2),
                DecoderBlockConfig(name="compress_all", multiplier=2),
            ),
        )
    )


def test_conv_video_decoder_runs_the_supported_non_attention_path() -> None:
    decoder = _decoder()
    result = decoder(mx.ones((1, INPUT_CHANNELS, 2, 2, 2), dtype=mx.float32))
    assert result.shape == (1, OUTPUT_CHANNELS, 3, 8, 8)
    np.testing.assert_array_equal(np.asarray(result), np.zeros(result.shape, dtype=np.float32))


def test_conv_video_decoder_rejects_unsupported_block() -> None:
    with pytest.raises(LaraError, match="LARA-TENSOR-013"):
        ConvVideoDecoder(
            ConvVideoDecoderConfig(
                in_channels=INPUT_CHANNELS,
                out_channels=OUTPUT_CHANNELS,
                base_channels=BASE_CHANNELS,
                decoder_blocks=(DecoderBlockConfig(name="attn"),),
            )
        )
