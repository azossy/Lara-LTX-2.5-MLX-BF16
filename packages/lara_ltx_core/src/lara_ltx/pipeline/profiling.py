"""Optional phase-level MLX runtime profiling for local generation."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterator

import mlx.core as mx


class RuntimePhase(StrEnum):
    """Stable names for the major in-process generation phases."""

    TEXT_CONDITIONING = "text_conditioning"
    TRANSFORMER_LOAD = "transformer_load"
    SAMPLING_STAGE_ONE = "sampling_stage_one"
    LATENT_UPSCALE = "latent_upscale"
    SAMPLING_STAGE_TWO = "sampling_stage_two"
    VIDEO_DECODE = "video_decode"
    AUDIO_VAE_DECODE = "audio_vae_decode"
    VOCODER_DECODE = "vocoder_decode"


@dataclass(frozen=True)
class RuntimePhaseMetric:
    """Elapsed time and MLX allocator state for one completed phase."""

    phase: RuntimePhase
    elapsed_seconds: float
    active_bytes: int
    cache_bytes: int
    peak_bytes: int


class RuntimeProfiler:
    """Collect opt-in phase metrics without affecting the default hot path."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._metrics: list[RuntimePhaseMetric] = []

    @property
    def metrics(self) -> tuple[RuntimePhaseMetric, ...]:
        return tuple(self._metrics)

    @contextmanager
    def measure(self, phase: RuntimePhase) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        mx.reset_peak_memory()
        started = time.monotonic()
        try:
            yield
        finally:
            self._metrics.append(
                RuntimePhaseMetric(
                    phase=phase,
                    elapsed_seconds=time.monotonic() - started,
                    active_bytes=int(mx.get_active_memory()),
                    cache_bytes=int(mx.get_cache_memory()),
                    peak_bytes=int(mx.get_peak_memory()),
                )
            )
