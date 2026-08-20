"""Sequential checkpoint decoder lifecycle for bounded local media output."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import mlx.core as mx
import numpy as np

from lara_ltx.audio_vae import AudioVAEDecoder, VocoderWithBWE
from lara_ltx.errors import LaraError
from lara_ltx.media import DecodedMediaResult
from lara_ltx.video_vae import DiffusionVideoDecoder

from .profiling import RuntimePhase, RuntimeProfiler

VideoDecoderLoader = Callable[[], DiffusionVideoDecoder]
AudioDecoderLoader = Callable[[], AudioVAEDecoder]
VocoderLoader = Callable[[], VocoderWithBWE]


@dataclass(frozen=True)
class DecodeRuntimeConfig:
    video_noise_seed: int
    video_timesteps: tuple[float, ...]
    video_activation_budget_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.video_noise_seed, int)
            or isinstance(self.video_noise_seed, bool)
            or not self.video_timesteps
            or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in self.video_timesteps)
            or any(left < right for left, right in pairwise(self.video_timesteps))
            or not isinstance(self.video_activation_budget_bytes, int)
            or isinstance(self.video_activation_budget_bytes, bool)
            or self.video_activation_budget_bytes <= 0
        ):
            raise LaraError("LARA-RUNTIME-009", details={"reason": "invalid_decode_runtime_configuration"})


class CheckpointDecodeRuntime:
    """Decode video and audio sequentially so inactive components are released."""

    def __init__(
        self,
        *,
        video_decoder_loader: VideoDecoderLoader,
        audio_decoder_loader: AudioDecoderLoader,
        vocoder_loader: VocoderLoader,
        config: DecodeRuntimeConfig,
        profiler: RuntimeProfiler | None = None,
    ) -> None:
        self.video_decoder_loader = video_decoder_loader
        self.audio_decoder_loader = audio_decoder_loader
        self.vocoder_loader = vocoder_loader
        self.config = config
        self.profiler = profiler or RuntimeProfiler()

    def decode(self, *, video_latent: mx.array, audio_latent: mx.array) -> DecodedMediaResult:
        if video_latent.ndim != 5 or audio_latent.ndim != 4 or video_latent.shape[0] != audio_latent.shape[0]:
            raise LaraError("LARA-RUNTIME-009", details={"reason": "invalid_decode_latents"})
        with self.profiler.measure(RuntimePhase.VIDEO_DECODE):
            video_decoder = self.video_decoder_loader()
            output_shape = video_decoder.output_shape_for_latent(tuple(video_latent.shape[2:5]))
            initial_noise = mx.random.normal(
                (video_latent.shape[0], video_decoder.out_channels, *output_shape),
                key=mx.random.key(self.config.video_noise_seed),
                dtype=video_latent.dtype,
            )
            tiling = video_decoder.recommend_tiling(
                tuple(video_latent.shape[2:5]),
                activation_budget_bytes=self.config.video_activation_budget_bytes,
                batch_size=video_latent.shape[0],
            )
            timesteps = mx.broadcast_to(
                mx.array(self.config.video_timesteps, dtype=mx.float32)[None],
                (video_latent.shape[0], len(self.config.video_timesteps)),
            )
            pixels = video_decoder.decode(
                video_latent,
                initial_noise,
                timesteps=timesteps,
                stage4_tiling=tiling.stage4,
                stage5_tiling=tiling.stage5,
            )
            mx.eval(pixels)
            video = np.asarray(pixels.astype(mx.float32))
            del pixels, initial_noise, video_decoder
            mx.clear_cache()

        with self.profiler.measure(RuntimePhase.AUDIO_VAE_DECODE):
            audio_decoder = self.audio_decoder_loader()
            spectrogram = audio_decoder(audio_latent)
            mx.eval(spectrogram)
            del audio_decoder
            mx.clear_cache()

        with self.profiler.measure(RuntimePhase.VOCODER_DECODE):
            vocoder = self.vocoder_loader()
            waveform = vocoder(spectrogram)
            mx.eval(waveform)
            audio = np.asarray(waveform.astype(mx.float32))
            del waveform, spectrogram, vocoder
            mx.clear_cache()
        if not np.isfinite(video).all() or not np.isfinite(audio).all():
            raise LaraError("LARA-RUNTIME-009", details={"reason": "non_finite_decoded_media"})
        return DecodedMediaResult(video=video, audio=audio)
