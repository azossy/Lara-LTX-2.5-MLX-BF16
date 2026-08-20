#!/usr/bin/env python3
"""Validate an official CUDA media artifact and record its stream metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

ERROR_CODE = "LARA-RUNTIME-006"
HASH_BLOCK_BYTES = 1024 * 1024
REPORT_SCHEMA_VERSION = 1
REQUIRED_STREAM_TYPES = frozenset({"audio", "video"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _validated_probe_payload(payload: dict[str, Any]) -> tuple[float, set[str]]:
    try:
        duration = float(payload["format"]["duration"])
        stream_types = {str(stream["codec_type"]) for stream in payload["streams"]}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"[{ERROR_CODE}] ffprobe returned incomplete media metadata. "
            "Inspect the reference pipeline log and regenerate the output."
        ) from error
    if duration <= 0:
        raise ValueError(
            f"[{ERROR_CODE}] Media duration must be positive, received {duration}. "
            "Inspect the reference pipeline log and regenerate the output."
        )
    missing_types = REQUIRED_STREAM_TYPES - stream_types
    if missing_types:
        raise ValueError(
            f"[{ERROR_CODE}] Media is missing required stream types: {sorted(missing_types)}. "
            "Verify CUDA video/audio decoding and regenerate the output."
        )
    return duration, stream_types


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    if not arguments.media.is_file() or arguments.media.stat().st_size == 0:
        raise ValueError(
            f"[{ERROR_CODE}] Media artifact is missing or empty: {arguments.media}. "
            "Inspect the reference pipeline log and regenerate the output."
        )
    command = [
        arguments.ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(arguments.media),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        probe_payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"[{ERROR_CODE}] ffprobe could not inspect {arguments.media}. "
            "Install ffprobe or inspect the CUDA pipeline log, then retry."
        ) from error
    duration, stream_types = _validated_probe_payload(probe_payload)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "media": {
            "path": str(arguments.media),
            "size_bytes": arguments.media.stat().st_size,
            "sha256": _sha256(arguments.media),
            "duration_seconds": duration,
            "stream_types": sorted(stream_types),
        },
        "ffprobe": probe_payload,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
