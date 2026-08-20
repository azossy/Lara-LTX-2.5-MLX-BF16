"""Atomic local MP4 encoding for decoded MLX video and synchronized audio."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lara_ltx.errors import LaraError

DEFAULT_FFMPEG_BINARY = "ffmpeg"
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_AUDIO_BITRATE = "192k"
DEFAULT_AUDIO_SAMPLE_RATE = 48_000
DEFAULT_CRF = 18
DEFAULT_TIMEOUT_SECONDS = 300
RGB_CHANNEL_COUNT = 3
STEREO_CHANNEL_COUNT = 2
MODEL_PIXEL_MINIMUM = -1.0
MODEL_PIXEL_MAXIMUM = 1.0
BYTE_PIXEL_MAXIMUM = 255.0


def _ffmpeg_default() -> str:
    return os.environ.get("LARA_FFMPEG_BINARY", DEFAULT_FFMPEG_BINARY)


@dataclass(frozen=True)
class MediaEncodingConfig:
    frame_rate: float
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    video_codec: str = DEFAULT_VIDEO_CODEC
    audio_codec: str = DEFAULT_AUDIO_CODEC
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    audio_bitrate: str = DEFAULT_AUDIO_BITRATE
    crf: int = DEFAULT_CRF
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    ffmpeg_binary: str = field(default_factory=_ffmpeg_default)

    def __post_init__(self) -> None:
        strings = (
            self.video_codec,
            self.audio_codec,
            self.pixel_format,
            self.audio_bitrate,
            self.ffmpeg_binary,
        )
        if (
            not math.isfinite(self.frame_rate)
            or self.frame_rate <= 0
            or not isinstance(self.audio_sample_rate, int)
            or isinstance(self.audio_sample_rate, bool)
            or self.audio_sample_rate <= 0
            or not isinstance(self.crf, int)
            or isinstance(self.crf, bool)
            or not 0 <= self.crf <= 51
            or not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds <= 0
            or any(not isinstance(value, str) or not value.strip() for value in strings)
        ):
            raise LaraError("LARA-MEDIA-001", details={"reason": "invalid_encoding_configuration"})


def _video_frames(value: np.ndarray) -> np.ndarray:
    video = np.asarray(value, dtype=np.float32)
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != RGB_CHANNEL_COUNT:
        raise LaraError("LARA-MEDIA-001", details={"reason": "invalid_video_shape"})
    if not np.isfinite(video).all():
        raise LaraError("LARA-MEDIA-001", details={"reason": "non_finite_video"})
    frames = np.transpose(video[0], (1, 2, 3, 0))
    normalized = (np.clip(frames, MODEL_PIXEL_MINIMUM, MODEL_PIXEL_MAXIMUM) - MODEL_PIXEL_MINIMUM) / (
        MODEL_PIXEL_MAXIMUM - MODEL_PIXEL_MINIMUM
    )
    return np.rint(normalized * BYTE_PIXEL_MAXIMUM).astype(np.uint8)


def _audio_samples(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != STEREO_CHANNEL_COUNT:
        raise LaraError("LARA-MEDIA-001", details={"reason": "invalid_audio_shape"})
    if not np.isfinite(audio).all():
        raise LaraError("LARA-MEDIA-001", details={"reason": "non_finite_audio"})
    return np.ascontiguousarray(np.transpose(np.clip(audio[0], -1.0, 1.0), (1, 0)), dtype=np.float32)


def _resolve_binary(value: str) -> str:
    if os.path.isabs(value):
        if Path(value).is_file():
            return value
    else:
        resolved = shutil.which(value)
        if resolved is not None:
            return resolved
    raise LaraError("LARA-MEDIA-002", details={"reason": "ffmpeg_not_found"})


@dataclass(frozen=True)
class DecodedMediaResult:
    video: np.ndarray
    audio: np.ndarray

    def save(self, output_path: Path | str, *, config: MediaEncodingConfig) -> Path:
        """Encode one local MP4 atomically without a server or persistent intermediates."""

        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            raise LaraError("LARA-MEDIA-001", details={"reason": "output_must_be_mp4"})
        frames = _video_frames(self.video)
        samples = _audio_samples(self.audio)
        frame_count, height, width, _ = frames.shape
        if frame_count <= 0 or samples.shape[0] <= 0:
            raise LaraError("LARA-MEDIA-001", details={"reason": "empty_media"})
        ffmpeg = _resolve_binary(config.ffmpeg_binary)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".lara-media-", dir=output.parent) as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_input = temporary_root / "video.rgb"
            audio_input = temporary_root / "audio.f32"
            temporary_output = temporary_root / "encoded.mp4"
            frames.tofile(video_input)
            samples.tofile(audio_input)
            command = [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(config.frame_rate),
                "-i",
                str(video_input),
                "-f",
                "f32le",
                "-ar",
                str(config.audio_sample_rate),
                "-ac",
                str(STEREO_CHANNEL_COUNT),
                "-i",
                str(audio_input),
                "-c:v",
                config.video_codec,
                "-pix_fmt",
                config.pixel_format,
                "-crf",
                str(config.crf),
                "-c:a",
                config.audio_codec,
                "-b:a",
                config.audio_bitrate,
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=config.timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise LaraError("LARA-MEDIA-002", details={"reason": "ffmpeg_execution_failed"}) from error
            if completed.returncode != 0 or not temporary_output.is_file() or temporary_output.stat().st_size <= 0:
                raise LaraError("LARA-MEDIA-002", details={"reason": "ffmpeg_encode_failed"})
            temporary_output.replace(output)
        return output
