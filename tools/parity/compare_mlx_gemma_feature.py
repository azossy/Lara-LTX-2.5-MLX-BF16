#!/usr/bin/env python3
"""Compare mapped MLX Gemma V2 LTX feature projections with CUDA golden data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.models import gemma_feature_target_shapes, iter_component_weight_batches
from lara_ltx.models.gemma_feature_runtime import GemmaFeatureProjector
from lara_ltx.parity.metrics import compare_tensors

MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    rules = mapping["rules"]
    if not isinstance(rules, list):
        raise LaraError("LARA-MODEL-014", details={"source_key": str(arguments.mapping)})
    projector = GemmaFeatureProjector()
    active_memory_before_loading = int(mx.get_active_memory())
    mx.reset_peak_memory()
    for batch in iter_component_weight_batches(
        (arguments.gemma_checkpoint,),
        rules,
        expected_target_shapes=gemma_feature_target_shapes(),
    ):
        projector.load_weights(batch)
        mx.eval(*[value for _, value in batch])
    weight_load_memory = {
        "active_before_bytes": active_memory_before_loading,
        "active_after_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
    with np.load(arguments.cuda_reference) as reference:
        hidden_states = mx.array(reference["hidden_states"], dtype=mx.bfloat16)
        attention_mask = mx.array(reference["attention_mask"])
        video_reference = reference["video_features"]
        audio_reference = reference["audio_features"]
    video_features, audio_features = projector(hidden_states, attention_mask)
    mx.eval(video_features, audio_features)
    video = compare_tensors("video_features", video_reference, np.asarray(video_features.astype(mx.float32)))
    audio = compare_tensors("audio_features", audio_reference, np.asarray(audio_features.astype(mx.float32)))
    stream_metrics = (video, audio)
    passed = all(
        result.nan_count == 0
        and result.inf_count == 0
        and result.normalized_rmse <= MAX_NORMALIZED_RMSE
        and result.cosine_similarity >= MIN_COSINE_SIMILARITY
        for result in stream_metrics
    )
    report = {
        "schema_version": 1,
        "acceptance_criteria": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "requires_finite_output": True,
        },
        "video": video.to_dict(),
        "audio": audio.to_dict(),
        "weight_load_memory": weight_load_memory,
        "passed": passed,
    }
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
