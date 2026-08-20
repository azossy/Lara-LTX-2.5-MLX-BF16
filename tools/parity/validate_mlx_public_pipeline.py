#!/usr/bin/env python3
"""Generate and validate one prompt-to-MP4 artifact through the public API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx import LTXPipeline

HASH_BLOCK_BYTES = 1024 * 1024


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--num-frames", required=True, type=int)
    parser.add_argument("--frame-rate", required=True, type=float)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    arguments = parse_arguments()
    mx.reset_peak_memory()
    started = time.monotonic()
    pipeline = LTXPipeline.from_pretrained(arguments.model, profile_path=arguments.profile)
    loaded_seconds = time.monotonic() - started
    generated = pipeline(
        prompt=arguments.prompt,
        negative_prompt=arguments.negative_prompt,
        seed=arguments.seed,
        height=arguments.height,
        width=arguments.width,
        num_frames=arguments.num_frames,
        frame_rate=arguments.frame_rate,
        num_inference_steps=arguments.steps,
    )
    generated_seconds = time.monotonic() - started - loaded_seconds
    output = generated.save(arguments.output)
    total_seconds = time.monotonic() - started
    video_shape = list(generated.video.shape)
    audio_shape = list(generated.audio.shape)
    passed = (
        output.is_file()
        and output.stat().st_size > 0
        and video_shape == [1, 3, arguments.num_frames, arguments.height, arguments.width]
        and audio_shape[0:2] == [1, 2]
        and bool(np.isfinite(generated.video).all())
        and bool(np.isfinite(generated.audio).all())
    )
    report = {
        "schema_version": 1,
        "component": "public_prompt_to_mp4_pipeline",
        "configuration": {
            "model": arguments.model,
            "prompt": arguments.prompt,
            "negative_prompt": arguments.negative_prompt,
            "seed": arguments.seed,
            "height": arguments.height,
            "width": arguments.width,
            "num_frames": arguments.num_frames,
            "frame_rate": arguments.frame_rate,
            "steps": arguments.steps,
        },
        "outputs": {
            "video_shape": video_shape,
            "audio_shape": audio_shape,
            "media_path": str(output),
            "media_size_bytes": output.stat().st_size,
            "media_sha256": _sha256(output),
        },
        "timing_seconds": {
            "resolve": loaded_seconds,
            "generate": generated_seconds,
            "total": total_seconds,
        },
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
