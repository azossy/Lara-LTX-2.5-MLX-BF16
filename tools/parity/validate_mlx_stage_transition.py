#!/usr/bin/env python3
"""Validate the checkpoint-backed HQ stage-1 to stage-2 latent transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.parity.metrics import compare_tensors
from lara_ltx.sampling import (
    AudioLatentLayout,
    Res2sLatentState,
    VideoLatentLayout,
    apply_gaussian_noise,
    create_audio_state,
    create_video_state,
    prepare_stage_two_states,
)
from lara_ltx.video_vae import load_spatial_video_upscaler

STAGE_ONE_VIDEO_KEY = "stage_1_video_latent"
STAGE_ONE_AUDIO_KEY = "stage_1_audio_latent"
UPSCALED_VIDEO_KEY = "upscaled_video_latent"
STAGE_TWO_VIDEO_INPUT_KEY = "stage_2_noiser_input_0"
STAGE_TWO_AUDIO_INPUT_KEY = "stage_2_noiser_input_1"
STAGE_TWO_VIDEO_OUTPUT_KEY = "stage_2_noiser_output_0"
STAGE_TWO_AUDIO_OUTPUT_KEY = "stage_2_noiser_output_1"
STAGE_TWO_SIGMAS_KEY = "stage_2_sigmas"
MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upscaler-checkpoint", required=True, type=Path)
    parser.add_argument("--video-vae-checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--frame-rate", required=True, type=float)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


class _CapturedNoiseSequence:
    def __init__(self, noises: tuple[mx.array, ...]) -> None:
        self._noises = noises
        self._index = 0

    def __call__(self, state: Res2sLatentState, noise_scale: float) -> Res2sLatentState:
        noise = self._noises[self._index]
        self._index += 1
        return apply_gaussian_noise(state, noise, noise_scale)


def _video_layout(value: np.ndarray, frame_rate: float) -> VideoLatentLayout:
    batch, channels, frames, height, width = value.shape
    return VideoLatentLayout(
        batch=int(batch),
        channels=int(channels),
        frames=int(frames),
        height=int(height),
        width=int(width),
        frame_rate=frame_rate,
    )


def _audio_layout(value: np.ndarray) -> AudioLatentLayout:
    batch, channels, frames, mel_bins = value.shape
    return AudioLatentLayout(
        batch=int(batch),
        channels=int(channels),
        frames=int(frames),
        mel_bins=int(mel_bins),
    )


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    with np.load(arguments.cuda_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    stage_one_video = reference[STAGE_ONE_VIDEO_KEY]
    stage_one_audio = reference[STAGE_ONE_AUDIO_KEY]
    upscaled_video = reference[UPSCALED_VIDEO_KEY]
    stage_two_video_input = reference[STAGE_TWO_VIDEO_INPUT_KEY]
    stage_two_audio_input = reference[STAGE_TWO_AUDIO_INPUT_KEY]
    stage_two_video_output = reference[STAGE_TWO_VIDEO_OUTPUT_KEY]
    stage_two_audio_output = reference[STAGE_TWO_AUDIO_OUTPUT_KEY]
    noise_scale = float(reference[STAGE_TWO_SIGMAS_KEY][0])
    stage_one_layout = _video_layout(stage_one_video, arguments.frame_rate)
    stage_two_layout = _video_layout(upscaled_video, arguments.frame_rate)
    audio_layout = _audio_layout(stage_one_audio)
    video_state = create_video_state(
        stage_one_layout,
        initial_latent=mx.array(stage_one_video, dtype=mx.bfloat16),
    ).state
    audio_state = create_audio_state(
        audio_layout,
        initial_latent=mx.array(stage_one_audio, dtype=mx.bfloat16),
    ).state
    video_noise = (stage_two_video_output - (1.0 - noise_scale) * stage_two_video_input) / noise_scale
    audio_noise = (stage_two_audio_output - (1.0 - noise_scale) * stage_two_audio_input) / noise_scale
    captured_noiser = _CapturedNoiseSequence((mx.array(video_noise), mx.array(audio_noise)))
    mx.reset_peak_memory()
    upscaler = load_spatial_video_upscaler(
        upscaler_checkpoint=arguments.upscaler_checkpoint,
        video_vae_checkpoint=arguments.video_vae_checkpoint,
        mapping=mapping,
    )
    video, audio = prepare_stage_two_states(
        video_state,
        audio_state,
        stage_one_video_layout=stage_one_layout,
        stage_two_video_layout=stage_two_layout,
        audio_layout=audio_layout,
        upscaler=upscaler,
        noiser=captured_noiser,
        noise_scale=noise_scale,
    )
    mx.eval(video.state.clean_latent, video.state.latent, audio.state.clean_latent, audio.state.latent)
    candidates = {
        STAGE_TWO_VIDEO_INPUT_KEY: np.asarray(video.state.clean_latent.astype(mx.float32)),
        STAGE_TWO_AUDIO_INPUT_KEY: np.asarray(audio.state.clean_latent.astype(mx.float32)),
        STAGE_TWO_VIDEO_OUTPUT_KEY: np.asarray(video.state.latent.astype(mx.float32)),
        STAGE_TWO_AUDIO_OUTPUT_KEY: np.asarray(audio.state.latent.astype(mx.float32)),
    }
    metrics = {name: compare_tensors(name, reference[name], value) for name, value in candidates.items()}
    passed = all(
        metric.nan_count == 0
        and metric.inf_count == 0
        and metric.normalized_rmse <= MAX_NORMALIZED_RMSE
        and metric.cosine_similarity >= MIN_COSINE_SIMILARITY
        for metric in metrics.values()
    )
    report = {
        "schema_version": 1,
        "component": "hq_stage_latent_transition",
        "configuration": {
            "frame_rate": arguments.frame_rate,
            "noise_scale": noise_scale,
            "stage_one_video_shape": list(stage_one_layout.latent_shape),
            "stage_two_video_shape": list(stage_two_layout.latent_shape),
            "audio_shape": list(audio_layout.latent_shape),
        },
        "acceptance_criteria": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
        },
        "boundaries": {name: metric.to_dict() for name, metric in metrics.items()},
        "peak_bytes": int(mx.get_peak_memory()),
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
