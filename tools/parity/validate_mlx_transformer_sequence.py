#!/usr/bin/env python3
"""Validate the bounded checkpoint-backed AV transformer block sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.parity.metrics import compare_tensors
from lara_ltx.transformer import CheckpointTransformerBlocks, TransformerStream, run_transformer_block_sequence

MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--block-count", type=int, default=1)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--lora-strength", type=float, default=0.0)
    return parser.parse_args()


def _array(reference: dict[str, np.ndarray], name: str) -> mx.array:
    return mx.array(reference[name], dtype=mx.bfloat16)


def _stream(reference: dict[str, np.ndarray], prefix: str) -> TransformerStream:
    return TransformerStream(
        x=_array(reference, f"{prefix}_input"),
        context=_array(reference, f"{prefix}_context"),
        timesteps=_array(reference, f"{prefix}_timesteps"),
        prompt_timestep=_array(reference, f"{prefix}_prompt_timestep"),
        cross_scale_shift_timestep=_array(reference, f"{prefix}_cross_scale_shift_timestep"),
        cross_gate_timestep=_array(reference, f"{prefix}_cross_gate_timestep"),
    )


def _summary(name: str, value: mx.array) -> dict[str, object]:
    array = np.asarray(value.astype(mx.float32))
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def main() -> int:
    arguments = parse_arguments()
    with np.load(arguments.cuda_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    provider = CheckpointTransformerBlocks(
        checkpoint=arguments.transformer_checkpoint,
        block_count=arguments.block_count,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=arguments.lora_strength,
    )
    result = run_transformer_block_sequence(_stream(reference, "video"), _stream(reference, "audio"), provider)
    assert result.video is not None and result.audio is not None
    summaries = {
        "video": _summary("video_output", result.video.x),
        "audio": _summary("audio_output", result.audio.x),
    }
    parity: dict[str, object] | None = None
    passed = all(bool(summary["finite"]) for summary in summaries.values())
    if arguments.block_count == 1 and arguments.lora_checkpoint is None:
        video_metrics = compare_tensors(
            "video_output",
            reference["video_output"],
            np.asarray(result.video.x.astype(mx.float32)),
        )
        audio_metrics = compare_tensors(
            "audio_output",
            reference["audio_output"],
            np.asarray(result.audio.x.astype(mx.float32)),
        )
        parity = {"video": video_metrics.to_dict(), "audio": audio_metrics.to_dict()}
        passed = passed and all(
            metrics.normalized_rmse <= MAX_NORMALIZED_RMSE and metrics.cosine_similarity >= MIN_COSINE_SIMILARITY
            for metrics in (video_metrics, audio_metrics)
        )
    report = {
        "schema_version": 1,
        "component": "sequential_av_transformer_blocks",
        "block_count": result.completed_block_count,
        "lora": {
            "enabled": arguments.lora_checkpoint is not None,
            "strength": arguments.lora_strength,
        },
        "outputs": summaries,
        "parity": parity,
        "block_records": [asdict(record) for record in provider.records],
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
