"""Top-level in-process generation lifecycle composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import mlx.core as mx

from lara_ltx.media import DecodedMediaResult

from .profiling import RuntimePhase, RuntimeProfiler
from .two_stage import TwoStageLatentResult, TwoStageSamplingRequest


class SamplingRuntime(Protocol):
    def run_request(self, request: TwoStageSamplingRequest) -> TwoStageLatentResult: ...


class DecodeRuntime(Protocol):
    def decode(self, *, video_latent: mx.array, audio_latent: mx.array) -> DecodedMediaResult: ...


SamplingRuntimeFactory = Callable[[], SamplingRuntime]
DecodeRuntimeFactory = Callable[[], DecodeRuntime]


class LocalGenerationRuntime:
    """Sample latents, release transformer state, then decode local media."""

    def __init__(
        self,
        *,
        sampling_runtime_factory: SamplingRuntimeFactory,
        decode_runtime_factory: DecodeRuntimeFactory,
        profiler: RuntimeProfiler | None = None,
    ) -> None:
        self.sampling_runtime_factory = sampling_runtime_factory
        self.decode_runtime_factory = decode_runtime_factory
        self.profiler = profiler or RuntimeProfiler()

    def generate(self, request: TwoStageSamplingRequest) -> DecodedMediaResult:
        with self.profiler.measure(RuntimePhase.TRANSFORMER_LOAD):
            sampling_runtime = self.sampling_runtime_factory()
        latents = sampling_runtime.run_request(request)
        video_latent = latents.video
        audio_latent = latents.audio
        mx.eval(video_latent, audio_latent)
        del latents, sampling_runtime
        mx.clear_cache()

        decode_runtime = self.decode_runtime_factory()
        media = decode_runtime.decode(video_latent=video_latent, audio_latent=audio_latent)
        del decode_runtime, video_latent, audio_latent
        mx.clear_cache()
        return media
