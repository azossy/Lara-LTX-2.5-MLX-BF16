from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.audio_vae.vocoder import HannUpSample1d, MelSTFT, Vocoder, VocoderConfig
from lara_ltx.errors import LaraError


def test_small_vocoder_restores_configured_temporal_scale() -> None:
    model = Vocoder(VocoderConfig(initial_channels=8, upsample_rates=(2,), upsample_kernels=(4,)))

    output = model(mx.zeros((1, 2, 3, 64), dtype=mx.float32))
    mx.eval(output)

    assert output.shape == (1, 2, 6)
    np.testing.assert_array_equal(np.asarray(output), 0.0)


def test_vocoder_rejects_non_stereo_mel_layout() -> None:
    model = Vocoder(VocoderConfig(initial_channels=8, upsample_rates=(2,), upsample_kernels=(4,)))

    with pytest.raises(LaraError) as error:
        model(mx.zeros((1, 1, 3, 64)))

    assert error.value.code == "LARA-TENSOR-025"


def test_causal_mel_stft_produces_expected_frame_count() -> None:
    transform = MelSTFT(hop_length=80, win_length=512)

    output = transform(mx.zeros((1, 2, 160), dtype=mx.float32))
    mx.eval(output)

    assert output.shape == (1, 2, 64, 2)
    np.testing.assert_allclose(np.asarray(output), np.log(1e-5), rtol=1e-6)


def test_hann_resampler_uses_exact_integer_output_ratio() -> None:
    resampler = HannUpSample1d(3)

    output = resampler(mx.ones((1, 2, 4), dtype=mx.float32))
    mx.eval(output)

    assert output.shape == (1, 2, 12)
    assert np.isfinite(np.asarray(output)).all()
