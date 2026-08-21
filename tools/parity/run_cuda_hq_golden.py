#!/usr/bin/env python3
"""Run the pinned upstream HQ CUDA pipeline and record a reproducible result.

This tool intentionally invokes the official CUDA reference only.  It is a
golden-artifact producer, not part of the MLX inference runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PIPELINE_MODULE = "ltx_pipelines.ti2vid_two_stages_hq"
HASH_BLOCK_BYTES = 1024 * 1024
REPORT_SCHEMA_VERSION = 1
DEFAULT_SEED = 25_081_900
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 320
DEFAULT_NUM_FRAMES = 17
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_INFERENCE_STEPS = 15
DEFAULT_DIFFVAE_OPTIMIZATION = "chunked_eager"
RUNTIME_PROBE_TIMEOUT_SECONDS = 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _existing_file(path: Path, argument_name: str) -> Path:
    if not path.is_file():
        raise ValueError(
            f"[LARA-RUNTIME-005] {argument_name} does not name a readable file: {path}. "
            "Download and verify the pinned BF16 model pack, then retry."
        )
    return path


def _probe_runtime(python_executable: str, working_directory: Path, module: str = PIPELINE_MODULE) -> None:
    """Fail before generation when the configured Python cannot import the official pipeline."""
    try:
        completed = subprocess.run(
            [python_executable, "-c", f"import importlib; importlib.import_module({module!r})"],
            cwd=working_directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=RUNTIME_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            "[LARA-RUNTIME-005] The configured CUDA Python runtime could not be started or timed out. "
            "Install the pinned official ltx-pipelines environment and retry."
        ) from error
    if completed.returncode:
        raise ValueError(
            f"[LARA-RUNTIME-005] The configured CUDA Python runtime cannot import {module}. "
            "Install the pinned official ltx-pipelines package and its locked dependencies, then retry."
        )


def build_command(arguments: argparse.Namespace) -> list[str]:
    """Build the upstream invocation without shell interpolation."""
    return [
        arguments.python_executable,
        "-m",
        PIPELINE_MODULE,
        "--transformer-path",
        str(arguments.transformer_path),
        "--text-encoder-path",
        str(arguments.text_encoder_path),
        "--video-vae-path",
        str(arguments.video_vae_path),
        "--audio-vae-path",
        str(arguments.audio_vae_path),
        "--duration-head-path",
        str(arguments.duration_head_path),
        "--distilled-lora",
        str(arguments.distilled_lora_path),
        "--spatial-upsampler-path",
        str(arguments.spatial_upsampler_path),
        "--prompt",
        arguments.prompt,
        "--output-path",
        str(arguments.output_video),
        "--seed",
        str(arguments.seed),
        "--width",
        str(arguments.width),
        "--height",
        str(arguments.height),
        "--num-frames",
        str(arguments.num_frames),
        "--frame-rate",
        str(arguments.frame_rate),
        "--num-inference-steps",
        str(arguments.num_inference_steps),
        "--diffvae-optimization",
        arguments.diffvae_optimization,
    ]


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--text-encoder-path", type=Path, required=True)
    parser.add_argument("--video-vae-path", type=Path, required=True)
    parser.add_argument("--audio-vae-path", type=Path, required=True)
    parser.add_argument("--duration-head-path", type=Path, required=True)
    parser.add_argument("--distilled-lora-path", type=Path, required=True)
    parser.add_argument("--spatial-upsampler-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--diffvae-optimization", default=DEFAULT_DIFFVAE_OPTIMIZATION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    if not arguments.upstream_root.is_dir():
        raise ValueError(
            f"[LARA-RUNTIME-005] Upstream root is unavailable: {arguments.upstream_root}. "
            "Checkout the pinned official source revision and retry."
        )
    _probe_runtime(arguments.python_executable, arguments.upstream_root)
    for attribute, name in (
        ("transformer_path", "--transformer-path"),
        ("text_encoder_path", "--text-encoder-path"),
        ("video_vae_path", "--video-vae-path"),
        ("audio_vae_path", "--audio-vae-path"),
        ("duration_head_path", "--duration-head-path"),
        ("distilled_lora_path", "--distilled-lora-path"),
        ("spatial_upsampler_path", "--spatial-upsampler-path"),
    ):
        setattr(arguments, attribute, _existing_file(getattr(arguments, attribute), name))
    if arguments.width <= 0 or arguments.height <= 0 or arguments.width % 64 or arguments.height % 64:
        raise ValueError(
            "[LARA-RUNTIME-005] Width and height must be positive multiples of 64. "
            "Use a supported LTX-2.5 resolution and retry."
        )
    if arguments.num_frames <= 0 or (arguments.num_frames - 1) % 8:
        raise ValueError(
            "[LARA-RUNTIME-005] num-frames must follow 8 * k + 1. Use a positive LTX-2.5 temporal-grid value and retry."
        )

    arguments.output_video.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.log.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(arguments)
    started_at = datetime.now(timezone.utc)
    with arguments.log.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=arguments.upstream_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"[LARA-RUNTIME-005] Official CUDA HQ pipeline failed with exit code {completed.returncode}. "
            f"Inspect {arguments.log}, verify the pinned pack, and retry."
        )
    if not arguments.output_video.is_file() or arguments.output_video.stat().st_size == 0:
        raise RuntimeError(
            f"[LARA-RUNTIME-005] Official CUDA HQ pipeline did not produce a playable output at "
            f"{arguments.output_video}. Inspect {arguments.log} and retry."
        )
    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = (finished_at - started_at).total_seconds()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "captured_at_utc": finished_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "generated_frames_per_second": arguments.num_frames / elapsed_seconds,
        "pipeline_module": PIPELINE_MODULE,
        "command": command,
        "parameters": {
            "seed": arguments.seed,
            "width": arguments.width,
            "height": arguments.height,
            "num_frames": arguments.num_frames,
            "frame_rate": arguments.frame_rate,
            "num_inference_steps": arguments.num_inference_steps,
            "diffvae_optimization": arguments.diffvae_optimization,
        },
        "output_video": {
            "path": str(arguments.output_video),
            "size_bytes": arguments.output_video.stat().st_size,
            "sha256": _sha256(arguments.output_video),
        },
        "log": str(arguments.log),
    }
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0


def cli() -> int:
    try:
        return main()
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
