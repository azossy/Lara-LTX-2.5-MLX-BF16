from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.audio_vae.decoder import (
    AudioPerChannelStatistics,
    AudioVAEDecoder,
    AudioVAEDecoderConfig,
    HeightCausalConv2d,
)
from lara_ltx.errors import LaraError


def test_height_causal_conv_does_not_read_future_frames() -> None:
    conv = HeightCausalConv2d(1, 1, kernel_size=3)
    conv.conv.weight = mx.ones((1, 1, 3, 3), dtype=mx.float32)
    conv.conv.bias = mx.zeros((1,), dtype=mx.float32)
    value = mx.zeros((1, 1, 4, 3), dtype=mx.float32)
    value = value.at[:, :, 3:, :].add(1.0)

    output = conv(value)

    np.testing.assert_array_equal(np.asarray(output[:, :, :3]), 0.0)
    assert np.asarray(output[:, :, 3:]).max() > 0


def test_audio_statistics_follow_patchified_channel_frequency_order() -> None:
    statistics = AudioPerChannelStatistics(feature_width=4)
    statistics.mean_of_means = mx.array([10.0, 20.0, 30.0, 40.0])
    statistics.std_of_means = mx.ones((4,))
    latent = mx.zeros((1, 2, 1, 2))

    output = statistics.un_normalize(latent)

    np.testing.assert_array_equal(np.asarray(output), [[[[10.0, 20.0]], [[30.0, 40.0]]]])


def test_small_audio_decoder_restores_causal_time_and_frequency_shape() -> None:
    decoder = AudioVAEDecoder(
        AudioVAEDecoderConfig(
            base_channels=4,
            channel_multipliers=(1, 2),
            latent_channels=2,
            output_channels=2,
            residual_blocks_per_level=1,
            latent_downsample_factor=2,
            output_mel_bins=8,
        )
    )

    output = decoder(mx.ones((1, 2, 3, 4), dtype=mx.bfloat16))
    mx.eval(output)

    assert output.shape == (1, 2, 5, 8)
    np.testing.assert_array_equal(np.asarray(output.astype(mx.float32)), 0.0)


def test_audio_decoder_rejects_wrong_latent_channels() -> None:
    decoder = AudioVAEDecoder(
        AudioVAEDecoderConfig(
            base_channels=4,
            channel_multipliers=(1, 2),
            latent_channels=2,
            output_channels=2,
            residual_blocks_per_level=1,
            latent_downsample_factor=2,
            output_mel_bins=8,
        )
    )

    with pytest.raises(LaraError) as error:
        decoder(mx.zeros((1, 3, 2, 4)))

    assert error.value.code == "LARA-TENSOR-024"
