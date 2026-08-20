from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.sampling import Res2sLatentState
from lara_ltx.transformer import (
    AVTransformerBlock,
    AVTransformerInputPreprocessor,
    AVTransformerOutput,
    DenoiserModalityConditioning,
    ResidentAVDenoiser,
    TransformerInputConfig,
    TransformerOutputConfig,
    VideoTransformerConfig,
)

TOKEN_COUNT = 2
LATENT_CHANNELS = 4
HIDDEN_DIMENSION = 8
ATTENTION_HEADS = 2
EXPECTED_BLOCK_COUNT = 1


def _input_config() -> TransformerInputConfig:
    return TransformerInputConfig(
        input_channels=LATENT_CHANNELS,
        hidden_dimension=HIDDEN_DIMENSION,
        adaln_coefficient=6,
        attention_heads=ATTENTION_HEADS,
        max_positions=(16,),
    )


def _block() -> AVTransformerBlock:
    return AVTransformerBlock(
        video=VideoTransformerConfig(
            dim=HIDDEN_DIMENSION,
            heads=ATTENTION_HEADS,
            head_dim=HIDDEN_DIMENSION // ATTENTION_HEADS,
            context_dim=HIDDEN_DIMENSION,
            apply_gated_attention=False,
            cross_attention_adaln=False,
            ff_bias=True,
        )
    )


def _conditioning() -> DenoiserModalityConditioning:
    positions = mx.array([[[[0, 1], [1, 2]]]], dtype=mx.float32)
    return DenoiserModalityConditioning(
        positions=positions,
        context=mx.ones((1, 3, HIDDEN_DIMENSION), dtype=mx.float32),
    )


def _state() -> Res2sLatentState:
    latent = mx.arange(TOKEN_COUNT * LATENT_CHANNELS, dtype=mx.float32).reshape(
        1,
        TOKEN_COUNT,
        LATENT_CHANNELS,
    )
    return Res2sLatentState(
        latent=latent,
        denoise_mask=mx.array([[[1.0], [0.0]]], dtype=mx.float32),
        clean_latent=latent,
    )


def _denoiser(*, blocks: tuple[tuple[int, AVTransformerBlock], ...]) -> ResidentAVDenoiser:
    config = _input_config()
    return ResidentAVDenoiser(
        input_processor=AVTransformerInputPreprocessor(
            video=config,
            audio=config,
            cross_attention_dimension=HIDDEN_DIMENSION,
        ),
        blocks=blocks,
        expected_block_count=EXPECTED_BLOCK_COUNT,
        output_heads=AVTransformerOutput(
            video=TransformerOutputConfig(HIDDEN_DIMENSION, LATENT_CHANNELS),
            audio=TransformerOutputConfig(HIDDEN_DIMENSION, LATENT_CHANNELS),
        ),
        video_conditioning=_conditioning(),
        audio_conditioning=None,
    )


def test_resident_denoiser_runs_input_block_output_and_preserves_clean_token() -> None:
    state = _state()

    video, audio = _denoiser(blocks=((0, _block()),))(state, None, 0.5)

    assert video is not None
    assert audio is None
    assert video.shape == state.latent.shape
    assert np.isfinite(np.asarray(video)).all()
    np.testing.assert_array_equal(np.asarray(video[:, 1]), np.asarray(state.latent[:, 1]))


def test_resident_denoiser_rejects_exhausted_or_incomplete_block_provider() -> None:
    with pytest.raises(LaraError) as error:
        _denoiser(blocks=())(_state(), None, 0.5)

    assert error.value.code == "LARA-TENSOR-029"
    assert "block_count_mismatch" in error.value.details["reason"]


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), -0.1])
def test_resident_denoiser_rejects_invalid_sigma(sigma: float) -> None:
    with pytest.raises(LaraError) as error:
        _denoiser(blocks=((0, _block()),))(_state(), None, sigma)

    assert error.value.code == "LARA-TENSOR-029"
