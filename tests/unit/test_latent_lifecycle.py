from __future__ import annotations

import mlx.core as mx
import numpy as np
from lara_ltx.sampling import (
    AudioLatentLayout,
    VideoLatentLayout,
    apply_gaussian_noise,
    create_audio_state,
    create_video_state,
    patchify_audio,
    patchify_video,
    prepare_stage_two_states,
    unpatchify_audio,
    unpatchify_video,
)


def test_video_patchification_positions_and_first_frame_marker() -> None:
    layout = VideoLatentLayout(
        batch=1,
        channels=4,
        frames=3,
        height=2,
        width=3,
        frame_rate=24.0,
    )
    source = mx.arange(int(np.prod(layout.latent_shape)), dtype=mx.float32).reshape(layout.latent_shape)

    tokens = patchify_video(source, layout)
    restored = unpatchify_video(tokens, layout)
    bundle = create_video_state(layout, initial_latent=source)

    np.testing.assert_array_equal(np.asarray(restored), np.asarray(source))
    assert tokens.shape == (1, 18, 4)
    assert bundle.positions.shape == (1, 3, 18, 2)
    assert bundle.keyframes_mask is not None
    np.testing.assert_array_equal(
        np.asarray(bundle.keyframes_mask).reshape(-1),
        np.array([1.0] * 6 + [0.0] * 12),
    )
    np.testing.assert_allclose(np.asarray(bundle.positions)[0, 0, 0], [0.0, 1.0 / 24.0])


def test_audio_patchification_and_causal_timestamps() -> None:
    layout = AudioLatentLayout(batch=1, channels=2, frames=3, mel_bins=4)
    source = mx.arange(int(np.prod(layout.latent_shape)), dtype=mx.float32).reshape(layout.latent_shape)

    tokens = patchify_audio(source, layout)
    restored = unpatchify_audio(tokens, layout)
    bundle = create_audio_state(layout, initial_latent=source)

    np.testing.assert_array_equal(np.asarray(restored), np.asarray(source))
    assert tokens.shape == (1, 3, 8)
    np.testing.assert_allclose(
        np.asarray(bundle.positions)[0, 0],
        np.array([[0.0, 0.01], [0.01, 0.05], [0.05, 0.09]]),
        atol=1e-7,
    )


def test_gaussian_noise_blends_only_denoised_tokens() -> None:
    bundle = create_audio_state(AudioLatentLayout(batch=1, channels=1, frames=2, mel_bins=2))
    state = bundle.state
    state = type(state)(
        latent=mx.zeros((1, 2, 2), dtype=mx.float32),
        denoise_mask=mx.array([[[1.0], [0.0]]]),
        clean_latent=mx.full((1, 2, 2), 3.0),
    )

    result = apply_gaussian_noise(state, mx.full(state.latent.shape, 2.0), 0.5)

    np.testing.assert_array_equal(np.asarray(result.latent), np.array([[[1.0, 1.0], [3.0, 3.0]]]))


def test_stage_transition_upsamples_video_and_reuses_stage_one_audio() -> None:
    stage_one_layout = VideoLatentLayout(
        batch=1,
        channels=2,
        frames=1,
        height=2,
        width=2,
        frame_rate=24.0,
    )
    stage_two_layout = VideoLatentLayout(
        batch=1,
        channels=2,
        frames=1,
        height=4,
        width=4,
        frame_rate=24.0,
    )
    audio_layout = AudioLatentLayout(batch=1, channels=1, frames=2, mel_bins=2)
    stage_one_video = create_video_state(
        stage_one_layout,
        initial_latent=mx.ones(stage_one_layout.latent_shape),
    ).state
    stage_one_audio = create_audio_state(
        audio_layout,
        initial_latent=mx.full(audio_layout.latent_shape, 2.0),
    ).state

    video, audio = prepare_stage_two_states(
        stage_one_video,
        stage_one_audio,
        stage_one_video_layout=stage_one_layout,
        stage_two_video_layout=stage_two_layout,
        audio_layout=audio_layout,
        upscaler=lambda value: mx.repeat(mx.repeat(value, 2, axis=3), 2, axis=4),
        noiser=lambda state, scale: apply_gaussian_noise(state, mx.zeros_like(state.latent), scale),
        noise_scale=0.25,
    )

    assert video.state.latent.shape == (1, 16, 2)
    assert audio.state.latent.shape == (1, 2, 2)
    np.testing.assert_allclose(np.asarray(video.state.latent.astype(mx.float32)), 0.75, atol=1e-3)
    np.testing.assert_allclose(np.asarray(audio.state.latent.astype(mx.float32)), 1.5, atol=1e-3)
