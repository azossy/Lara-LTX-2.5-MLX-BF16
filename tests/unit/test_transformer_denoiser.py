from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.sampling import MultiModalGuider, MultiModalGuiderParams, Res2sLatentState
from lara_ltx.transformer import (
    AVTransformerBlock,
    AVTransformerInputPreprocessor,
    AVTransformerOutput,
    DenoiserModalityConditioning,
    GuidedDenoiserConditioning,
    GuidedResidentAVDenoiser,
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


def _guided_denoiser(params: MultiModalGuiderParams) -> GuidedResidentAVDenoiser:
    config = _input_config()
    conditioned = _conditioning()
    unconditioned = DenoiserModalityConditioning(
        positions=conditioned.positions,
        context=mx.zeros(conditioned.context.shape, dtype=conditioned.context.dtype),
    )
    return GuidedResidentAVDenoiser(
        input_processor=AVTransformerInputPreprocessor(
            video=config,
            audio=config,
            cross_attention_dimension=HIDDEN_DIMENSION,
        ),
        blocks=((0, _block()),),
        expected_block_count=EXPECTED_BLOCK_COUNT,
        output_heads=AVTransformerOutput(
            video=TransformerOutputConfig(HIDDEN_DIMENSION, LATENT_CHANNELS),
            audio=TransformerOutputConfig(HIDDEN_DIMENSION, LATENT_CHANNELS),
        ),
        video_conditioning=GuidedDenoiserConditioning(conditioned, unconditioned),
        audio_conditioning=None,
        video_guider=MultiModalGuider(params),
        audio_guider=MultiModalGuider(MultiModalGuiderParams()),
    )


def _guided_av_denoiser() -> GuidedResidentAVDenoiser:
    config = _input_config()
    conditioned = _conditioning()
    unconditioned = DenoiserModalityConditioning(
        positions=conditioned.positions,
        context=mx.zeros(conditioned.context.shape, dtype=conditioned.context.dtype),
    )
    block_config = VideoTransformerConfig(
        dim=HIDDEN_DIMENSION,
        heads=ATTENTION_HEADS,
        head_dim=HIDDEN_DIMENSION // ATTENTION_HEADS,
        context_dim=HIDDEN_DIMENSION,
        apply_gated_attention=False,
        cross_attention_adaln=False,
        ff_bias=True,
    )
    return GuidedResidentAVDenoiser(
        input_processor=AVTransformerInputPreprocessor(
            video=config,
            audio=config,
            cross_attention_dimension=HIDDEN_DIMENSION,
        ),
        blocks=((0, AVTransformerBlock(video=block_config, audio=block_config)),),
        expected_block_count=EXPECTED_BLOCK_COUNT,
        output_heads=AVTransformerOutput(
            video=TransformerOutputConfig(HIDDEN_DIMENSION, LATENT_CHANNELS),
            audio=TransformerOutputConfig(HIDDEN_DIMENSION, LATENT_CHANNELS),
        ),
        video_conditioning=GuidedDenoiserConditioning(conditioned, unconditioned),
        audio_conditioning=GuidedDenoiserConditioning(conditioned),
        video_guider=MultiModalGuider(MultiModalGuiderParams(cfg_scale=2.0, skip_step=1)),
        audio_guider=MultiModalGuider(MultiModalGuiderParams()),
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


def test_guided_denoiser_batches_cfg_and_matches_separate_predictions() -> None:
    cfg_scale = 2.0
    guided = _guided_denoiser(MultiModalGuiderParams(cfg_scale=cfg_scale))
    state = _state()
    guided_video, _ = guided(state, None, 0.5)
    assert guided.video_conditioning is not None

    def direct(conditioning: DenoiserModalityConditioning) -> mx.array:
        video, _ = ResidentAVDenoiser(
            input_processor=guided.input_processor,
            blocks=guided.blocks,
            expected_block_count=guided.expected_block_count,
            output_heads=guided.output_heads,
            video_conditioning=conditioning,
            audio_conditioning=None,
        )(state, None, 0.5)
        assert video is not None
        return video

    conditioned = direct(guided.video_conditioning.conditioned)
    unconditioned = direct(guided.video_conditioning.unconditioned)
    expected = conditioned + (cfg_scale - 1.0) * (conditioned - unconditioned)

    assert guided_video is not None
    np.testing.assert_allclose(np.asarray(guided_video), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_guided_denoiser_applies_stg_batch_and_honors_skip_step() -> None:
    guided = _guided_denoiser(
        MultiModalGuiderParams(
            cfg_scale=2.0,
            stg_scale=0.5,
            stg_blocks=(0,),
            skip_step=1,
        )
    )
    state = _state()
    guided_video, _ = guided(state, None, 0.5)
    assert guided_video is not None
    assert np.isfinite(np.asarray(guided_video)).all()

    guided.set_step_index(1)
    skipped_video, _ = guided(state, None, 0.5)

    assert skipped_video is not None
    np.testing.assert_array_equal(np.asarray(skipped_video), np.asarray(guided_video))


def test_guided_denoiser_reuses_only_skipped_modality_while_other_runs() -> None:
    guided = _guided_av_denoiser()
    state = _state()
    first_video, first_audio = guided(state, state, 0.5)
    assert first_video is not None and first_audio is not None

    guided.set_step_index(1)
    skipped_video, active_audio = guided(state, state, 0.5)

    assert skipped_video is not None and active_audio is not None
    np.testing.assert_array_equal(np.asarray(skipped_video), np.asarray(first_video))
    assert np.isfinite(np.asarray(active_audio)).all()
