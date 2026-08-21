#!/usr/bin/env python3
"""Build transparent 4K presentation exports from matched native CUDA/MLX videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.quality.corpus import ERROR_CODE, message, write_json_atomic
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from corpus import ERROR_CODE, message, write_json_atomic

REPORT_SCHEMA_VERSION = 1
HASH_BLOCK_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 1800
BACKENDS = ("cuda", "mlx")


class DemoBuildError(RuntimeError):
    """Stable, user-actionable presentation-build failure."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cuda-video", required=True, type=Path)
    parser.add_argument("--mlx-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_BLOCK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path, executable: str, timeout_seconds: int) -> dict[str, Any]:
    command = [
        executable,
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    if completed.returncode:
        raise DemoBuildError(f"probe_failed:{path.name}")
    try:
        return dict(json.loads(completed.stdout))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise DemoBuildError(f"invalid_probe:{path.name}") from error


def _video_stream(probe: dict[str, Any], path: Path) -> dict[str, Any]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return dict(stream)
    raise DemoBuildError(f"missing_video_stream:{path.name}")


def _has_audio(probe: dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def _validate_source(path: Path, probe: dict[str, Any], case: dict[str, Any]) -> None:
    stream = _video_stream(probe, path)
    observed_frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    expected = (int(case["width"]), int(case["height"]), int(case["num_frames"]))
    observed = (int(stream.get("width", 0)), int(stream.get("height", 0)), observed_frames)
    if observed != expected:
        raise DemoBuildError(f"source_grid_mismatch:{path.name}:{observed}")
    if not _has_audio(probe):
        raise DemoBuildError(f"missing_audio_stream:{path.name}")


def _scale_filter(width: int, height: int, scaler: str) -> str:
    return (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease:flags={scaler},"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )


def _run(command: list[str], timeout_seconds: int) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    if completed.returncode:
        summary = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown_ffmpeg_error"
        raise DemoBuildError(f"ffmpeg_failed:{summary}")
    return time.perf_counter() - started


def _individual_command(
    *, executable: str, source: Path, output: Path, presentation: dict[str, Any]
) -> list[str]:
    return [
        executable,
        "-y",
        "-i",
        str(source),
        "-vf",
        _scale_filter(
            int(presentation["target_width"]),
            int(presentation["target_height"]),
            str(presentation["scaler"]),
        ),
        "-c:v",
        str(presentation["video_codec"]),
        "-crf",
        str(presentation["crf"]),
        "-pix_fmt",
        str(presentation["pixel_format"]),
        "-c:a",
        str(presentation["audio_codec"]),
        "-b:a",
        str(presentation["audio_bitrate"]),
        "-movflags",
        "+faststart",
        str(output),
    ]


def _side_by_side_command(
    *, executable: str, cuda_video: Path, mlx_video: Path, output: Path, presentation: dict[str, Any]
) -> list[str]:
    target_width = int(presentation["target_width"])
    target_height = int(presentation["target_height"])
    if target_width % 2:
        raise DemoBuildError("side_by_side_width_must_be_even")
    half_width = target_width // 2
    scaler = str(presentation["scaler"])
    left = _scale_filter(half_width, target_height, scaler)
    right = _scale_filter(half_width, target_height, scaler)
    filter_graph = (
        f"[0:v]{left}[cuda];[1:v]{right}[mlx];"
        "[cuda][mlx]hstack=inputs=2:shortest=1[video];"
        "[0:a]apad[cuda_audio];[1:a]apad[mlx_audio]"
    )
    return [
        executable,
        "-y",
        "-i",
        str(cuda_video),
        "-i",
        str(mlx_video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[video]",
        "-map",
        "[cuda_audio]",
        "-map",
        "[mlx_audio]",
        "-metadata:s:a:0",
        "title=CUDA audio",
        "-metadata:s:a:1",
        "title=MLX Metal audio",
        "-c:v",
        str(presentation["video_codec"]),
        "-crf",
        str(presentation["crf"]),
        "-pix_fmt",
        str(presentation["pixel_format"]),
        "-c:a",
        str(presentation["audio_codec"]),
        "-b:a",
        str(presentation["audio_bitrate"]),
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.timeout_seconds <= 0:
            raise DemoBuildError("invalid_timeout")
        if shutil.which(arguments.ffmpeg) is None or shutil.which(arguments.ffprobe) is None:
            raise DemoBuildError("ffmpeg_or_ffprobe_unavailable")
        manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
        cases = list(manifest["cases"])
        if len(cases) != 1:
            raise DemoBuildError("manifest_requires_exactly_one_case")
        case = dict(cases[0])
        presentation = dict(manifest["presentation"])
        sources = {"cuda": arguments.cuda_video, "mlx": arguments.mlx_video}
        source_probes: dict[str, dict[str, Any]] = {}
        for backend, path in sources.items():
            if not path.is_file() or path.stat().st_size <= 0:
                raise DemoBuildError(f"missing_source:{backend}")
            source_probes[backend] = _probe(path, arguments.ffprobe, arguments.timeout_seconds)
            _validate_source(path, source_probes[backend], case)

        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {
            backend: arguments.output_dir / f"{case['id']}_{backend}_4k.mp4" for backend in BACKENDS
        }
        elapsed: dict[str, float] = {}
        commands: dict[str, list[str]] = {}
        for backend in BACKENDS:
            command = _individual_command(
                executable=arguments.ffmpeg,
                source=sources[backend],
                output=outputs[backend],
                presentation=presentation,
            )
            commands[backend] = command
            elapsed[backend] = _run(command, arguments.timeout_seconds)

        comparison = arguments.output_dir / f"{case['id']}_cuda_mlx_side_by_side_4k.mp4"
        comparison_command = _side_by_side_command(
            executable=arguments.ffmpeg,
            cuda_video=sources["cuda"],
            mlx_video=sources["mlx"],
            output=comparison,
            presentation=presentation,
        )
        commands["side_by_side"] = comparison_command
        elapsed["side_by_side"] = _run(comparison_command, arguments.timeout_seconds)
        outputs["side_by_side"] = comparison

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "component": "cuda_mlx_cinematic_4k_presentation",
            "case": case,
            "presentation": presentation,
            "sources": {
                backend: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "probe": source_probes[backend],
                }
                for backend, path in sources.items()
            },
            "outputs": {
                name: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "elapsed_seconds": elapsed[name],
                    "probe": _probe(path, arguments.ffprobe, arguments.timeout_seconds),
                }
                for name, path in outputs.items()
            },
            "commands": commands,
            "passed": True,
            "scope": "identical_deterministic_presentation_upscale_not_native_4k_generation",
        }
        write_json_atomic(arguments.report, report)
    except (DemoBuildError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"[{ERROR_CODE}] {message('demo_build_failed', reason=str(error))}", file=sys.stderr)
        return 2
    print(message("demo_build_completed", report=arguments.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
