from __future__ import annotations

import mlx.core as mx
import numpy as np
from lara_ltx.media import DecodedMediaResult
from lara_ltx.pipeline import (
    LocalGenerationRuntime,
    TwoStageContexts,
    TwoStageLatentResult,
    TwoStageSamplingRequest,
)
from lara_ltx.sampling import (
    AudioLatentLayout,
    MultiModalGuider,
    MultiModalGuiderParams,
    Res2sLatentState,
    VideoLatentLayout,
    create_audio_state,
    create_video_state,
)


def _request() -> TwoStageSamplingRequest:
    video_layout = VideoLatentLayout(batch=1, channels=4, frames=1, height=1, width=1, frame_rate=24.0)
    audio_layout = AudioLatentLayout(batch=1, channels=2, frames=1, mel_bins=2)
    context = mx.zeros((1, 1, 8), dtype=mx.float32)
    return TwoStageSamplingRequest(
        stage_one_video=create_video_state(video_layout, dtype=mx.float32),
        stage_one_audio=create_audio_state(audio_layout, dtype=mx.float32),
        stage_one_video_layout=video_layout,
        stage_two_video_layout=video_layout,
        audio_layout=audio_layout,
        contexts=TwoStageContexts(video_positive=context, audio_positive=context),
        video_guider=MultiModalGuider(MultiModalGuiderParams()),
        audio_guider=MultiModalGuider(MultiModalGuiderParams()),
        stage_one_sigmas=mx.array([1.0, 0.0]),
        stage_two_sigmas=mx.array([1.0, 0.0]),
    )


class _SamplingRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def run_request(self, request: TwoStageSamplingRequest) -> TwoStageLatentResult:
        self.events.append("sample")
        video_state = Res2sLatentState(
            latent=request.stage_one_video.state.latent,
            denoise_mask=request.stage_one_video.state.denoise_mask,
            clean_latent=request.stage_one_video.state.clean_latent,
        )
        audio_state = Res2sLatentState(
            latent=request.stage_one_audio.state.latent,
            denoise_mask=request.stage_one_audio.state.denoise_mask,
            clean_latent=request.stage_one_audio.state.clean_latent,
        )
        return TwoStageLatentResult(
            video=mx.zeros((1, 4, 1, 1, 1)),
            audio=mx.zeros((1, 2, 1, 2)),
            stage_one_video=video_state,
            stage_one_audio=audio_state,
            stage_two_video=video_state,
        )


class _DecodeRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def decode(self, *, video_latent: mx.array, audio_latent: mx.array) -> DecodedMediaResult:
        self.events.append("decode")
        assert video_latent.shape == (1, 4, 1, 1, 1)
        assert audio_latent.shape == (1, 2, 1, 2)
        return DecodedMediaResult(
            video=np.zeros((1, 3, 1, 2, 2), dtype=np.float32),
            audio=np.zeros((1, 2, 64), dtype=np.float32),
        )


def test_local_generation_runtime_sequences_sampling_before_decode() -> None:
    events: list[str] = []
    runtime = LocalGenerationRuntime(
        sampling_runtime_factory=lambda: _SamplingRuntime(events),
        decode_runtime_factory=lambda: _DecodeRuntime(events),
    )

    result = runtime.generate(_request())

    assert events == ["sample", "decode"]
    assert result.video.shape == (1, 3, 1, 2, 2)
    assert result.audio.shape == (1, 2, 64)
