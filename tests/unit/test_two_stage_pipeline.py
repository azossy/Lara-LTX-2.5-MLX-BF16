from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.pipeline import TwoStageContexts, TwoStageSamplingConfig, TwoStageSamplingRuntime
from lara_ltx.sampling import (
    AudioLatentLayout,
    MultiModalGuider,
    MultiModalGuiderParams,
    Res2sLatentState,
    Res2sSampler,
    VideoLatentLayout,
    apply_gaussian_noise,
    create_audio_state,
    create_video_state,
    unpatchify_audio,
)
from lara_ltx.transformer import (
    AVTransformerBlock,
    AVTransformerInputPreprocessor,
    AVTransformerOutput,
    TransformerInputConfig,
    TransformerOutputConfig,
    VideoTransformerConfig,
)

VIDEO_CHANNELS = 4
AUDIO_CHANNELS = 2
AUDIO_MEL_BINS = 2
HIDDEN_DIMENSION = 8
ATTENTION_HEADS = 2
EXPECTED_BLOCK_COUNT = 1
FRAME_RATE = 24.0
STAGE_ONE_LORA_STRENGTH = 0.25
STAGE_TWO_LORA_STRENGTH = 0.5


def _video_layout(*, height: int, width: int) -> VideoLatentLayout:
    return VideoLatentLayout(
        batch=1,
        channels=VIDEO_CHANNELS,
        frames=1,
        height=height,
        width=width,
        frame_rate=FRAME_RATE,
    )


def _audio_layout() -> AudioLatentLayout:
    return AudioLatentLayout(batch=1, channels=AUDIO_CHANNELS, frames=1, mel_bins=AUDIO_MEL_BINS)


def _input_config(channels: int, *, max_positions: tuple[int, ...]) -> TransformerInputConfig:
    return TransformerInputConfig(
        input_channels=channels,
        hidden_dimension=HIDDEN_DIMENSION,
        adaln_coefficient=6,
        attention_heads=ATTENTION_HEADS,
        max_positions=max_positions,
    )


def _block() -> AVTransformerBlock:
    config = VideoTransformerConfig(
        dim=HIDDEN_DIMENSION,
        heads=ATTENTION_HEADS,
        head_dim=HIDDEN_DIMENSION // ATTENTION_HEADS,
        context_dim=HIDDEN_DIMENSION,
        apply_gated_attention=False,
        cross_attention_adaln=False,
        ff_bias=True,
    )
    return AVTransformerBlock(video=config, audio=config)


def _module_loader(
    recorded_strengths: list[float],
) -> Callable[[float], tuple[AVTransformerInputPreprocessor, AVTransformerOutput]]:
    def load(strength: float) -> tuple[AVTransformerInputPreprocessor, AVTransformerOutput]:
        recorded_strengths.append(strength)
        return (
            AVTransformerInputPreprocessor(
                video=_input_config(VIDEO_CHANNELS, max_positions=(64, 64, 64)),
                audio=_input_config(AUDIO_CHANNELS * AUDIO_MEL_BINS, max_positions=(64,)),
                cross_attention_dimension=HIDDEN_DIMENSION,
            ),
            AVTransformerOutput(
                video=TransformerOutputConfig(HIDDEN_DIMENSION, VIDEO_CHANNELS),
                audio=TransformerOutputConfig(HIDDEN_DIMENSION, AUDIO_CHANNELS * AUDIO_MEL_BINS),
            ),
        )

    return load


def _initial_noiser(state: Res2sLatentState, scale: float) -> Res2sLatentState:
    return apply_gaussian_noise(state, mx.ones(state.latent.shape, dtype=state.latent.dtype), scale)


def _zero_sampler_noise(value: mx.array, _stream: str) -> mx.array:
    return mx.zeros(value.shape, dtype=value.dtype)


def test_two_stage_runtime_reuses_blocks_switches_lora_and_preserves_stage_one_audio() -> None:
    stage_one_layout = _video_layout(height=1, width=1)
    stage_two_layout = _video_layout(height=2, width=2)
    audio_layout = _audio_layout()
    loaded_strengths: list[float] = []
    applied_strengths: list[float] = []
    runtime = TwoStageSamplingRuntime(
        blocks=((0, _block()),),
        expected_block_count=EXPECTED_BLOCK_COUNT,
        stage_module_loader=_module_loader(loaded_strengths),
        set_lora_strength=applied_strengths.append,
        upscaler=lambda value: mx.repeat(mx.repeat(value, 2, axis=3), 2, axis=4),
        initial_noiser=_initial_noiser,
        stage_one_sampler=Res2sSampler(bongmath=False, noise_fn=_zero_sampler_noise),
        stage_two_sampler=Res2sSampler(bongmath=False, noise_fn=_zero_sampler_noise),
        config=TwoStageSamplingConfig(
            stage_one_lora_strength=STAGE_ONE_LORA_STRENGTH,
            stage_two_lora_strength=STAGE_TWO_LORA_STRENGTH,
        ),
    )
    context = mx.ones((1, 2, HIDDEN_DIMENSION), dtype=mx.float32)

    result = runtime.run(
        stage_one_video=create_video_state(stage_one_layout, dtype=mx.float32),
        stage_one_audio=create_audio_state(audio_layout, dtype=mx.float32),
        stage_one_video_layout=stage_one_layout,
        stage_two_video_layout=stage_two_layout,
        audio_layout=audio_layout,
        contexts=TwoStageContexts(video_positive=context, audio_positive=context),
        video_guider=MultiModalGuider(MultiModalGuiderParams()),
        audio_guider=MultiModalGuider(MultiModalGuiderParams()),
        stage_one_sigmas=mx.array([0.75, 0.0], dtype=mx.float32),
        stage_two_sigmas=mx.array([0.5, 0.0], dtype=mx.float32),
    )

    assert result.video.shape == stage_two_layout.latent_shape
    assert result.audio.shape == audio_layout.latent_shape
    assert np.isfinite(np.asarray(result.video)).all()
    assert np.isfinite(np.asarray(result.audio)).all()
    np.testing.assert_array_equal(
        np.asarray(result.audio),
        np.asarray(unpatchify_audio(result.stage_one_audio.latent, audio_layout)),
    )
    assert loaded_strengths == [STAGE_ONE_LORA_STRENGTH, STAGE_TWO_LORA_STRENGTH]
    assert applied_strengths == [STAGE_ONE_LORA_STRENGTH, STAGE_TWO_LORA_STRENGTH]


def test_two_stage_runtime_rejects_non_reusable_block_count() -> None:
    with pytest.raises(LaraError) as error:
        TwoStageSamplingRuntime(
            blocks=iter(()),
            expected_block_count=EXPECTED_BLOCK_COUNT,
            stage_module_loader=_module_loader([]),
            set_lora_strength=lambda _strength: None,
            upscaler=lambda value: value,
            initial_noiser=_initial_noiser,
            stage_one_sampler=Res2sSampler(bongmath=False, noise_fn=_zero_sampler_noise),
            stage_two_sampler=Res2sSampler(bongmath=False, noise_fn=_zero_sampler_noise),
            config=TwoStageSamplingConfig(0.0, 0.0),
        )

    assert error.value.code == "LARA-RUNTIME-008"
    assert error.value.details["reason"] == "invalid_stage_block_count"


@pytest.mark.parametrize("strength", [-0.1, float("nan"), float("inf")])
def test_two_stage_sampling_config_rejects_invalid_lora_strength(strength: float) -> None:
    with pytest.raises(LaraError) as error:
        TwoStageSamplingConfig(strength, STAGE_TWO_LORA_STRENGTH)

    assert error.value.code == "LARA-RUNTIME-008"
