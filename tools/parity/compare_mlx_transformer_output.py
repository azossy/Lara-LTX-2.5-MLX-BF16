#!/usr/bin/env python3
"""Compare mapped MLX transformer output heads with CUDA BF16 golden data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.models import iter_component_weight_batches
from lara_ltx.models.transformer_output import transformer_output_target_shapes
from lara_ltx.parity.metrics import compare_tensors
from lara_ltx.transformer.output import AVTransformerOutput, TransformerOutputConfig

ARTIFACT_SCHEMA_VERSION = 1
VIDEO_HIDDEN_DIMENSION = 4096
AUDIO_HIDDEN_DIMENSION = 2048
OUTPUT_CHANNELS = 128
MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    rules = mapping.get("rules")
    if not isinstance(rules, list):
        raise LaraError("LARA-MODEL-014", details={"source_key": str(arguments.mapping)})
    output = AVTransformerOutput(
        video=TransformerOutputConfig(VIDEO_HIDDEN_DIMENSION, OUTPUT_CHANNELS),
        audio=TransformerOutputConfig(AUDIO_HIDDEN_DIMENSION, OUTPUT_CHANNELS),
    )
    active_memory_before_loading = int(mx.get_active_memory())
    mx.reset_peak_memory()
    for batch in iter_component_weight_batches(
        (arguments.transformer_checkpoint,),
        rules,
        expected_target_shapes=transformer_output_target_shapes(),
    ):
        output.load_weights(batch)
        mx.eval(*[value for _, value in batch])
    weight_load_memory = {
        "active_before_bytes": active_memory_before_loading,
        "active_after_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
    with np.load(arguments.cuda_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    video, audio = output(
        mx.array(reference["video_hidden"], dtype=mx.bfloat16),
        mx.array(reference["video_timestep"], dtype=mx.bfloat16),
        mx.array(reference["audio_hidden"], dtype=mx.bfloat16),
        mx.array(reference["audio_timestep"], dtype=mx.bfloat16),
    )
    mx.eval(video, audio)
    metrics = {
        "video": compare_tensors("video_output", reference["video_output"], np.asarray(video.astype(mx.float32))),
        "audio": compare_tensors("audio_output", reference["audio_output"], np.asarray(audio.astype(mx.float32))),
    }
    passed = all(
        result.nan_count == 0
        and result.inf_count == 0
        and result.normalized_rmse <= MAX_NORMALIZED_RMSE
        and result.cosine_similarity >= MIN_COSINE_SIMILARITY
        for result in metrics.values()
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "acceptance_criteria": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "requires_finite_output": True,
        },
        "streams": {name: result.to_dict() for name, result in metrics.items()},
        "weight_load_memory": weight_load_memory,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
