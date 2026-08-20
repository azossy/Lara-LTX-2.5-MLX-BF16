from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.pipeline import CheckpointDecodeRuntime, DecodeRuntimeConfig
from lara_ltx.pipeline.profiling import RuntimePhase, RuntimeProfiler
from lara_ltx.video_vae import DiffusionVideoDecoder, DiffusionVideoDecoderConfig


def _video_decoder() -> DiffusionVideoDecoder:
    return DiffusionVideoDecoder(
        DiffusionVideoDecoderConfig(
            in_channels=4,
            out_channels=3,
            patch_size=2,
            head_dim=16,
            stage_channels=(16, 16, 16, 16, 16),
            stage_depths=(1, 1, 1, 1, 1),
            stage_kernels=((3, 3, 3),) * 5,
            upsamples=(((1, 1, 1), 1),) * 4,
            stage5_kernel=(3, 3, 3),
            stage5_channels=16,
            timestep_embedding_dim=16,
        )
    )


class _AudioDecoder:
    def __call__(self, latent: mx.array) -> mx.array:
        return mx.ones((latent.shape[0], 2, 4, 8), dtype=latent.dtype)


class _Vocoder:
    def __call__(self, spectrogram: mx.array) -> mx.array:
        return mx.zeros((spectrogram.shape[0], 2, 64), dtype=spectrogram.dtype)


def _recorded_loader(record: list[str], name: str, value: object) -> Callable[[], object]:
    def load() -> object:
        record.append(name)
        return value

    return load


def test_checkpoint_decode_runtime_runs_video_audio_and_vocoder_in_order() -> None:
    loads: list[str] = []
    profiler = RuntimeProfiler(enabled=True)
    runtime = CheckpointDecodeRuntime(
        video_decoder_loader=_recorded_loader(loads, "video", _video_decoder()),
        audio_decoder_loader=_recorded_loader(loads, "audio", _AudioDecoder()),
        vocoder_loader=_recorded_loader(loads, "vocoder", _Vocoder()),
        config=DecodeRuntimeConfig(
            video_noise_seed=42,
            video_timesteps=(1.0,),
            video_activation_budget_bytes=3_000_000,
        ),
        profiler=profiler,
    )

    result = runtime.decode(
        video_latent=mx.zeros((1, 4, 1, 1, 1), dtype=mx.bfloat16),
        audio_latent=mx.zeros((1, 2, 1, 2), dtype=mx.bfloat16),
    )

    assert loads == ["video", "audio", "vocoder"]
    assert result.video.shape == (1, 3, 1, 2, 2)
    assert result.audio.shape == (1, 2, 64)
    assert np.isfinite(result.video).all()
    assert np.isfinite(result.audio).all()
    assert [metric.phase for metric in profiler.metrics] == [
        RuntimePhase.VIDEO_DECODE,
        RuntimePhase.AUDIO_VAE_DECODE,
        RuntimePhase.VOCODER_DECODE,
    ]
    assert all(metric.elapsed_seconds >= 0.0 and metric.peak_bytes >= 0 for metric in profiler.metrics)


@pytest.mark.parametrize(
    "config",
    [
        DecodeRuntimeConfig(video_noise_seed=0, video_timesteps=(1.0,), video_activation_budget_bytes=1),
    ],
)
def test_checkpoint_decode_runtime_rejects_invalid_latents(config: DecodeRuntimeConfig) -> None:
    runtime = CheckpointDecodeRuntime(
        video_decoder_loader=_video_decoder,
        audio_decoder_loader=_AudioDecoder,
        vocoder_loader=_Vocoder,
        config=config,
    )

    with pytest.raises(LaraError) as error:
        runtime.decode(
            video_latent=mx.zeros((1, 4, 1, 1), dtype=mx.bfloat16),
            audio_latent=mx.zeros((1, 2, 1, 2), dtype=mx.bfloat16),
        )

    assert error.value.code == "LARA-RUNTIME-009"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"video_noise_seed": True, "video_timesteps": (1.0,), "video_activation_budget_bytes": 1},
        {"video_noise_seed": 0, "video_timesteps": (), "video_activation_budget_bytes": 1},
        {"video_noise_seed": 0, "video_timesteps": (0.5, 0.75), "video_activation_budget_bytes": 1},
        {"video_noise_seed": 0, "video_timesteps": (1.0,), "video_activation_budget_bytes": 0},
    ],
)
def test_decode_runtime_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(LaraError) as error:
        DecodeRuntimeConfig(**kwargs)

    assert error.value.code == "LARA-RUNTIME-009"
