#!/usr/bin/env python3
"""Compare decoded CUDA/MLX video frames, motion deltas and synchronized audio."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors

REPORT_SCHEMA_VERSION = 1
RGB_CHANNELS = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--max-audio-lag-ms", required=True, type=float)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _run(command: list[str], *, text: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=text)
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaraError("LARA-PARITY-005", details={"reason": "media_decode_failed"}) from error
    return completed.stdout


def _probe(path: Path, binary: str) -> dict[str, Any]:
    output = _run(
        [
            binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate,sample_rate,channels,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    try:
        return json.loads(output)
    except (TypeError, json.JSONDecodeError) as error:
        raise LaraError("LARA-PARITY-005", details={"reason": "invalid_probe_output"}) from error


def _stream(probe: dict[str, Any], kind: str) -> dict[str, Any]:
    try:
        return next(item for item in probe["streams"] if item["codec_type"] == kind)
    except (KeyError, StopIteration, TypeError) as error:
        raise LaraError("LARA-PARITY-005", details={"reason": f"missing_{kind}_stream"}) from error


def _video(path: Path, binary: str, *, width: int, height: int) -> np.ndarray:
    payload = _run([binary, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    values = np.frombuffer(payload, dtype=np.uint8)
    frame_elements = width * height * RGB_CHANNELS
    if not values.size or values.size % frame_elements:
        raise LaraError("LARA-PARITY-005", details={"reason": "invalid_video_payload"})
    return values.reshape(-1, height, width, RGB_CHANNELS).astype(np.float32) / np.float32(255.0)


def _audio(path: Path, binary: str, *, sample_rate: int, channels: int) -> np.ndarray:
    payload = _run(
        [
            binary,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "pipe:1",
        ]
    )
    values = np.frombuffer(payload, dtype=np.float32)
    if not values.size or values.size % channels:
        raise LaraError("LARA-PARITY-005", details={"reason": "invalid_audio_payload"})
    return values.reshape(-1, channels)


def _audio_alignment(
    reference: np.ndarray, candidate: np.ndarray, *, sample_rate: int, max_lag_ms: float
) -> dict[str, float]:
    left = reference.mean(axis=1, dtype=np.float64)
    right = candidate.mean(axis=1, dtype=np.float64)
    left -= left.mean()
    right -= right.mean()
    size = left.size + right.size - 1
    fft_size = 1 << size.bit_length()
    circular = np.fft.irfft(np.fft.rfft(left, fft_size) * np.conj(np.fft.rfft(right, fft_size)), fft_size)
    correlation = np.concatenate((circular[-(right.size - 1) :], circular[: left.size]))
    lags = np.arange(-(right.size - 1), left.size)
    maximum_lag = round(max_lag_ms * sample_rate / 1000.0)
    allowed = np.abs(lags) <= maximum_lag
    selected_index = int(np.argmax(np.abs(correlation[allowed])))
    lag = int(lags[allowed][selected_index])
    if lag >= 0:
        aligned_left = left[lag:]
        aligned_right = right[: aligned_left.size]
    else:
        aligned_right = right[-lag:]
        aligned_left = left[: aligned_right.size]
    norm = float(np.linalg.norm(aligned_left) * np.linalg.norm(aligned_right))
    cosine = float(np.dot(aligned_left, aligned_right) / norm) if norm else 1.0
    return {"lag_samples": lag, "lag_ms": lag * 1000.0 / sample_rate, "aligned_cosine_similarity": cosine}


def _quality_metrics(reference: np.ndarray, candidate: np.ndarray, name: str) -> dict[str, Any]:
    metrics = compare_tensors(name, reference, candidate).to_dict()
    rmse = float(metrics["rmse"])
    metrics["psnr_db"] = float("inf") if rmse == 0 else 20.0 * math.log10(1.0 / rmse)
    return metrics


def _threshold_result(report: dict[str, Any], path: Path | None) -> tuple[dict[str, Any] | None, bool]:
    if path is None:
        return None, True
    try:
        thresholds = json.loads(path.read_text(encoding="utf-8"))
        minimum_frame_cosine = float(thresholds["minimum_frame_cosine"])
        minimum_temporal_cosine = float(thresholds["minimum_temporal_cosine"])
        minimum_audio_cosine = float(thresholds["minimum_audio_aligned_cosine"])
        maximum_audio_lag_ms = float(thresholds["maximum_audio_lag_ms"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise LaraError("LARA-PARITY-005", details={"reason": "invalid_quality_thresholds"}) from error
    checks = {
        "frame_cosine": report["video"]["frame_metrics"]["cosine_similarity"] >= minimum_frame_cosine,
        "temporal_cosine": report["video"]["temporal_delta_metrics"]["cosine_similarity"] >= minimum_temporal_cosine,
        "audio_cosine": report["audio"]["metrics"]["cosine_similarity"] >= minimum_audio_cosine,
        "audio_lag": abs(report["audio"]["alignment"]["lag_ms"]) <= maximum_audio_lag_ms,
    }
    return {"thresholds": thresholds, "checks": checks}, all(checks.values())


def main() -> int:
    arguments = parse_arguments()
    reference_probe = _probe(arguments.reference, arguments.ffprobe)
    candidate_probe = _probe(arguments.candidate, arguments.ffprobe)
    reference_video_stream = _stream(reference_probe, "video")
    candidate_video_stream = _stream(candidate_probe, "video")
    dimensions = (
        int(reference_video_stream["width"]),
        int(reference_video_stream["height"]),
    )
    if dimensions != (int(candidate_video_stream["width"]), int(candidate_video_stream["height"])):
        raise LaraError("LARA-PARITY-005", details={"reason": "video_dimension_mismatch"})
    reference_frames = _video(arguments.reference, arguments.ffmpeg, width=dimensions[0], height=dimensions[1])
    candidate_frames = _video(arguments.candidate, arguments.ffmpeg, width=dimensions[0], height=dimensions[1])
    frame_count = min(reference_frames.shape[0], candidate_frames.shape[0])
    reference_frames = reference_frames[:frame_count]
    candidate_frames = candidate_frames[:frame_count]
    reference_audio_stream = _stream(reference_probe, "audio")
    candidate_audio_stream = _stream(candidate_probe, "audio")
    audio_layout = (int(reference_audio_stream["sample_rate"]), int(reference_audio_stream["channels"]))
    if audio_layout != (int(candidate_audio_stream["sample_rate"]), int(candidate_audio_stream["channels"])):
        raise LaraError("LARA-PARITY-005", details={"reason": "audio_layout_mismatch"})
    reference_audio = _audio(
        arguments.reference, arguments.ffmpeg, sample_rate=audio_layout[0], channels=audio_layout[1]
    )
    candidate_audio = _audio(
        arguments.candidate, arguments.ffmpeg, sample_rate=audio_layout[0], channels=audio_layout[1]
    )
    sample_count = min(reference_audio.shape[0], candidate_audio.shape[0])
    reference_audio = reference_audio[:sample_count]
    candidate_audio = candidate_audio[:sample_count]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": "cuda_mlx_media_quality",
        "reference": str(arguments.reference),
        "candidate": str(arguments.candidate),
        "video": {
            "frame_count": frame_count,
            "width": dimensions[0],
            "height": dimensions[1],
            "frame_metrics": _quality_metrics(reference_frames, candidate_frames, "decoded_rgb_frames"),
            "temporal_delta_metrics": _quality_metrics(
                np.diff(reference_frames, axis=0),
                np.diff(candidate_frames, axis=0),
                "decoded_rgb_temporal_delta",
            ),
        },
        "audio": {
            "sample_count": sample_count,
            "sample_rate": audio_layout[0],
            "channels": audio_layout[1],
            "metrics": _quality_metrics(reference_audio, candidate_audio, "decoded_audio"),
            "alignment": _audio_alignment(
                reference_audio,
                candidate_audio,
                sample_rate=audio_layout[0],
                max_lag_ms=arguments.max_audio_lag_ms,
            ),
        },
    }
    threshold_result, passed = _threshold_result(report, arguments.thresholds)
    report["acceptance"] = threshold_result
    report["passed"] = passed
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
