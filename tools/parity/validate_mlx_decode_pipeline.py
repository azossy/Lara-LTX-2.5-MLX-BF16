#!/usr/bin/env python3
"""Decode fixed video/audio latents sequentially and save one local MLX MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.audio_vae import load_audio_vae_decoder, load_vocoder_with_bwe
from lara_ltx.errors import LaraError
from lara_ltx.media import MediaEncodingConfig
from lara_ltx.pipeline import CheckpointDecodeRuntime, DecodeRuntimeConfig
from lara_ltx.video_vae import load_diffusion_video_decoder

CONFIG_SCHEMA_VERSION = 1
HASH_BLOCK_BYTES = 1024 * 1024
REQUIRED_STREAM_TYPES = frozenset({"audio", "video"})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-checkpoint", required=True, type=Path)
    parser.add_argument("--audio-checkpoint", required=True, type=Path)
    parser.add_argument("--video-mapping", required=True, type=Path)
    parser.add_argument("--audio-mapping", required=True, type=Path)
    parser.add_argument("--vocoder-mapping", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-RUNTIME-009", details={"reason": "unreadable_decode_smoke_input"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-RUNTIME-009", details={"reason": "invalid_decode_smoke_input"})
    return value


def _required(config: dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    value = config.get(key)
    if not isinstance(value, expected) or isinstance(value, bool):
        raise LaraError("LARA-RUNTIME-009", details={"reason": f"invalid_decode_config_{key}"})
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-009", details={"reason": "unsupported_decode_smoke_schema"})
    video_mapping = _read_json(arguments.video_mapping)
    audio_mapping = _read_json(arguments.audio_mapping)
    vocoder_mapping = _read_json(arguments.vocoder_mapping)
    video_latent_key = _required(config, "video_latent_key", str)
    audio_latent_key = _required(config, "audio_latent_key", str)
    timestep_values = _required(config, "video_timesteps", list)
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in timestep_values):
        raise LaraError("LARA-RUNTIME-009", details={"reason": "invalid_decode_config_video_timesteps"})
    runtime_config = DecodeRuntimeConfig(
        video_noise_seed=_required(config, "video_noise_seed", int),
        video_timesteps=tuple(float(value) for value in timestep_values),
        video_activation_budget_bytes=_required(config, "video_activation_budget_bytes", int),
    )
    encoding_config = MediaEncodingConfig(
        frame_rate=float(_required(config, "frame_rate", (int, float))),
        audio_sample_rate=_required(config, "audio_sample_rate", int),
        video_codec=_required(config, "video_codec", str),
        audio_codec=_required(config, "audio_codec", str),
        pixel_format=_required(config, "pixel_format", str),
        audio_bitrate=_required(config, "audio_bitrate", str),
        crf=_required(config, "crf", int),
        timeout_seconds=_required(config, "timeout_seconds", int),
        ffmpeg_binary=_required(config, "ffmpeg_binary", str),
    )
    ffprobe_binary = _required(config, "ffprobe_binary", str)
    with np.load(arguments.reference) as archive:
        try:
            video_latent = mx.array(archive[video_latent_key], dtype=mx.bfloat16)
            audio_latent = mx.array(archive[audio_latent_key], dtype=mx.bfloat16)
        except KeyError as error:
            raise LaraError("LARA-RUNTIME-009", details={"reason": "missing_decode_smoke_latent"}) from error
    runtime = CheckpointDecodeRuntime(
        video_decoder_loader=partial(
            load_diffusion_video_decoder,
            checkpoint=arguments.video_checkpoint,
            mapping=video_mapping,
        ),
        audio_decoder_loader=partial(
            load_audio_vae_decoder,
            checkpoint=arguments.audio_checkpoint,
            mapping=audio_mapping,
        ),
        vocoder_loader=partial(
            load_vocoder_with_bwe,
            checkpoint=arguments.audio_checkpoint,
            mapping=vocoder_mapping,
        ),
        config=runtime_config,
    )
    mx.reset_peak_memory()
    started = time.monotonic()
    media = runtime.decode(video_latent=video_latent, audio_latent=audio_latent)
    output = media.save(arguments.output, config=encoding_config)
    elapsed_seconds = time.monotonic() - started
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels,nb_frames",
        "-of",
        "json",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=encoding_config.timeout_seconds,
        )
        probe = json.loads(completed.stdout)
        duration = float(probe["format"]["duration"])
        stream_types = {stream["codec_type"] for stream in probe["streams"]}
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        raise LaraError("LARA-MEDIA-002", details={"reason": "ffprobe_validation_failed"}) from error
    passed = duration > 0 and REQUIRED_STREAM_TYPES <= stream_types
    report = {
        "schema_version": 1,
        "component": "checkpoint_sequential_decode_pipeline",
        "configuration": config,
        "outputs": {
            "video_shape": list(media.video.shape),
            "audio_shape": list(media.audio.shape),
            "media_path": str(output),
            "media_size_bytes": output.stat().st_size,
            "media_sha256": _sha256(output),
        },
        "ffprobe": probe,
        "elapsed_seconds": elapsed_seconds,
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
