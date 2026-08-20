#!/usr/bin/env python3
"""Measure same-process public pipeline determinism and released MLX memory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx import LTXPipeline
from lara_ltx.errors import LaraError

HASH_BLOCK_BYTES = 1024 * 1024


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-PERF-001", details={"reason": "unreadable_config"}) from error
    required = {
        "schema_version",
        "model",
        "prompt",
        "seed",
        "height",
        "width",
        "num_frames",
        "frame_rate",
        "steps",
        "runs",
        "maximum_released_memory_growth_bytes",
        "require_identical_media",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload["schema_version"] != 1:
        raise LaraError("LARA-PERF-001", details={"reason": "invalid_config"})
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _memory() -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def main() -> int:
    arguments = parse_arguments()
    config = _config(arguments.config)
    runs = int(config["runs"])
    maximum_growth = int(config["maximum_released_memory_growth_bytes"])
    if runs < 2 or maximum_growth < 0:
        raise LaraError("LARA-PERF-001", details={"reason": "invalid_run_limits"})
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    load_started = time.monotonic()
    pipeline = LTXPipeline.from_pretrained(str(config["model"]))
    pipeline_resolution_seconds = time.monotonic() - load_started
    records: list[dict[str, object]] = []
    for run_index in range(runs):
        mx.reset_peak_memory()
        started = time.monotonic()
        generated = pipeline(
            prompt=str(config["prompt"]),
            seed=int(config["seed"]),
            height=int(config["height"]),
            width=int(config["width"]),
            num_frames=int(config["num_frames"]),
            frame_rate=float(config["frame_rate"]),
            num_inference_steps=int(config["steps"]),
            profile=True,
        )
        output = generated.save(arguments.output_directory / f"run_{run_index + 1:02d}.mp4")
        elapsed = time.monotonic() - started
        evaluated_memory = _memory()
        video_shape = list(generated.video.shape)
        audio_shape = list(generated.audio.shape)
        finite = bool(np.isfinite(generated.video).all() and np.isfinite(generated.audio).all())
        phase_metrics = [asdict(metric) for metric in generated.metrics]
        generation_peak_bytes = max(int(metric["peak_bytes"]) for metric in phase_metrics)
        del generated
        gc.collect()
        mx.clear_cache()
        released_memory = _memory()
        records.append(
            {
                "run": run_index + 1,
                "elapsed_seconds": elapsed,
                "output": str(output),
                "output_sha256": _sha256(output),
                "video_shape": video_shape,
                "audio_shape": audio_shape,
                "finite": finite,
                "phase_metrics": phase_metrics,
                "generation_peak_bytes": generation_peak_bytes,
                "evaluated_memory": evaluated_memory,
                "released_memory": released_memory,
            }
        )
    baseline = records[0]["released_memory"]
    assert isinstance(baseline, dict)
    released_growth = {
        key: max(int(record["released_memory"][key]) - int(baseline[key]) for record in records[1:])
        for key in ("active_bytes", "cache_bytes")
    }
    hashes = {str(record["output_sha256"]) for record in records}
    identical = len(hashes) == 1
    passed = (
        all(bool(record["finite"]) for record in records)
        and all(growth <= maximum_growth for growth in released_growth.values())
        and (identical or not bool(config["require_identical_media"]))
    )
    report = {
        "schema_version": 1,
        "component": "repeated_public_pipeline",
        "configuration": config,
        "pipeline_resolution_seconds": pipeline_resolution_seconds,
        "runs": records,
        "released_memory_growth_bytes": released_growth,
        "identical_media": identical,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
