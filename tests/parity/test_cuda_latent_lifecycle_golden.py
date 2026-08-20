from __future__ import annotations

from pathlib import Path

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
    unpatchify_audio,
    unpatchify_video,
)

REFERENCE = Path("golden/cuda_hq_boundary_trace_v2.npz")
FRAME_RATE = 24.0
STAGE_ONE_VIDEO_LAYOUT = VideoLatentLayout(
    batch=1,
    channels=128,
    frames=3,
    height=5,
    width=8,
    frame_rate=FRAME_RATE,
)
STAGE_TWO_VIDEO_LAYOUT = VideoLatentLayout(
    batch=1,
    channels=128,
    frames=3,
    height=10,
    width=16,
    frame_rate=FRAME_RATE,
)
AUDIO_LAYOUT = AudioLatentLayout(batch=1, channels=8, frames=18, mel_bins=16)


def test_stage_two_patchification_matches_cuda_boundaries() -> None:
    with np.load(REFERENCE) as archive:
        upscaled = mx.array(archive["upscaled_video_latent"], dtype=mx.bfloat16)
        audio = mx.array(archive["stage_1_audio_latent"], dtype=mx.bfloat16)
        expected_video = archive["stage_2_noiser_input_0"]
        expected_audio = archive["stage_2_noiser_input_1"]

    video_tokens = patchify_video(upscaled, STAGE_TWO_VIDEO_LAYOUT)
    audio_tokens = patchify_audio(audio, AUDIO_LAYOUT)

    np.testing.assert_array_equal(np.asarray(video_tokens.astype(mx.float32)), expected_video)
    np.testing.assert_array_equal(np.asarray(audio_tokens.astype(mx.float32)), expected_audio)


def test_stage_two_gaussian_lerp_order_reproduces_cuda_bf16_output() -> None:
    with np.load(REFERENCE) as archive:
        video_input = archive["stage_2_noiser_input_0"]
        video_output = archive["stage_2_noiser_output_0"]
        audio_input = archive["stage_2_noiser_input_1"]
        audio_output = archive["stage_2_noiser_output_1"]
        noise_scale = float(archive["stage_2_sigmas"][0])

    video_state = create_video_state(
        STAGE_TWO_VIDEO_LAYOUT,
        initial_latent=unpatchify_video(mx.array(video_input, dtype=mx.bfloat16), STAGE_TWO_VIDEO_LAYOUT),
    ).state
    audio_state = create_audio_state(
        AUDIO_LAYOUT,
        initial_latent=unpatchify_audio(mx.array(audio_input, dtype=mx.bfloat16), AUDIO_LAYOUT),
    ).state
    video_noise = (video_output - (1.0 - noise_scale) * video_input) / noise_scale
    audio_noise = (audio_output - (1.0 - noise_scale) * audio_input) / noise_scale

    video = apply_gaussian_noise(video_state, mx.array(video_noise), noise_scale)
    audio = apply_gaussian_noise(audio_state, mx.array(audio_noise), noise_scale)

    np.testing.assert_array_equal(np.asarray(video.latent.astype(mx.float32)), video_output)
    np.testing.assert_array_equal(np.asarray(audio.latent.astype(mx.float32)), audio_output)


def test_stage_layout_metadata_matches_captured_token_counts() -> None:
    stage_one = create_video_state(STAGE_ONE_VIDEO_LAYOUT)
    stage_two = create_video_state(STAGE_TWO_VIDEO_LAYOUT)
    audio = create_audio_state(AUDIO_LAYOUT)

    assert stage_one.state.latent.shape == (1, 120, 128)
    assert stage_two.state.latent.shape == (1, 480, 128)
    assert audio.state.latent.shape == (1, 18, 128)
    assert stage_one.positions.shape == (1, 3, 120, 2)
    assert stage_two.positions.shape == (1, 3, 480, 2)
    assert audio.positions.shape == (1, 1, 18, 2)
