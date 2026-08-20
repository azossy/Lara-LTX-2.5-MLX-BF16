#!/usr/bin/env python3
"""Encode and validate an MP4 through Lara's local decoded-media result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.media import DecodedMediaResult, MediaEncodingConfig

CONFIG_SCHEMA_VERSION = 1
HASH_BLOCK_BYTES = 1024 * 1024
REQUIRED_STREAM_TYPES = frozenset({"audio", "video"})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-MEDIA-001", details={"reason": "unreadable_mux_config"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-MEDIA-001", details={"reason": "invalid_mux_config"})
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _required(config: dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    value = config.get(key)
    if not isinstance(value, expected) or isinstance(value, bool):
        raise LaraError("LARA-MEDIA-001", details={"reason": f"invalid_mux_config_{key}"})
    return value


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-MEDIA-001", details={"reason": "unsupported_mux_config_schema"})
    video_key = _required(config, "video_key", str)
    audio_key = _required(config, "audio_key", str)
    encoding = MediaEncodingConfig(
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
            source_frames = archive[video_key]
            source_audio = archive[audio_key]
        except KeyError as error:
            raise LaraError("LARA-MEDIA-001", details={"reason": "missing_mux_boundary"}) from error
    if source_frames.ndim != 4 or source_audio.ndim != 2:
        raise LaraError("LARA-MEDIA-001", details={"reason": "invalid_mux_boundary"})
    video = np.transpose(source_frames, (3, 0, 1, 2))[None]
    audio = source_audio[None]
    output = DecodedMediaResult(video=video, audio=audio).save(arguments.output, config=encoding)
    probe_command = [
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
            probe_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=encoding.timeout_seconds,
        )
        probe = json.loads(completed.stdout)
        streams = probe["streams"]
        duration = float(probe["format"]["duration"])
        stream_types = {stream["codec_type"] for stream in streams}
        video_stream = next(stream for stream in streams if stream["codec_type"] == "video")
        audio_stream = next(stream for stream in streams if stream["codec_type"] == "audio")
    except (
        OSError,
        StopIteration,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        raise LaraError("LARA-MEDIA-002", details={"reason": "ffprobe_validation_failed"}) from error
    passed = (
        duration > 0
        and REQUIRED_STREAM_TYPES <= stream_types
        and int(video_stream["nb_frames"]) == source_frames.shape[0]
        and int(video_stream["height"]) == source_frames.shape[1]
        and int(video_stream["width"]) == source_frames.shape[2]
        and int(audio_stream["channels"]) == source_audio.shape[0]
        and int(audio_stream["sample_rate"]) == encoding.audio_sample_rate
    )
    report = {
        "schema_version": 1,
        "component": "local_media_mux_smoke",
        "configuration": config,
        "media": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "duration_seconds": duration,
            "stream_types": sorted(stream_types),
        },
        "ffprobe": probe,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
