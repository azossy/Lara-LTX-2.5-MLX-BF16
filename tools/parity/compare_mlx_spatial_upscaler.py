#!/usr/bin/env python3
"""Compare the complete MLX x2 latent upscaler with CUDA BF16 boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity.metrics import compare_tensors
from lara_ltx.video_vae import load_spatial_video_upscaler

ARTIFACT_SCHEMA_VERSION = 1
MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999
EXPECTED_UPSCALER_TENSORS = 72
EXPECTED_STATISTICS_TENSORS = 2
INPUT_KEY = "normalized_input_latent"
UNNORMALIZED_INPUT_KEY = "unnormalized_input_latent"
UNNORMALIZED_OUTPUT_KEY = "upscaled_unnormalized_latent"
NORMALIZED_OUTPUT_KEY = "upscaled_normalized_latent"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upscaler-checkpoint", required=True, type=Path)
    parser.add_argument("--video-vae-checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    required = (INPUT_KEY, UNNORMALIZED_INPUT_KEY, UNNORMALIZED_OUTPUT_KEY, NORMALIZED_OUTPUT_KEY)
    try:
        with np.load(path) as archive:
            if any(name not in archive.files for name in required):
                missing = next(name for name in required if name not in archive.files)
                raise LaraError("LARA-PARITY-002", details={"key": missing})
            return {name: archive[name] for name in required}
    except OSError as exc:
        raise LaraError("LARA-PARITY-002", details={"key": str(path)}) from exc


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    rules = mapping.get("rules")
    if not isinstance(rules, list) or len(rules) != EXPECTED_UPSCALER_TENSORS:
        raise LaraError("LARA-MODEL-030", details={"key": "mapping_rule_count"})

    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    owner = load_spatial_video_upscaler(
        upscaler_checkpoint=arguments.upscaler_checkpoint,
        video_vae_checkpoint=arguments.video_vae_checkpoint,
        mapping=mapping,
    )
    weight_load_memory = {
        "active_before_bytes": active_before,
        "active_after_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }

    reference = _load_reference(arguments.cuda_reference)
    normalized_input = mx.array(reference[INPUT_KEY], dtype=mx.bfloat16)
    unnormalized_input = owner.per_channel_statistics.un_normalize(normalized_input)
    upscaled_unnormalized = owner.upscaler(unnormalized_input)
    upscaled_normalized = owner.per_channel_statistics.normalize(upscaled_unnormalized)
    mx.eval(unnormalized_input, upscaled_unnormalized, upscaled_normalized)
    candidates = {
        UNNORMALIZED_INPUT_KEY: np.asarray(unnormalized_input.astype(mx.float32)),
        UNNORMALIZED_OUTPUT_KEY: np.asarray(upscaled_unnormalized.astype(mx.float32)),
        NORMALIZED_OUTPUT_KEY: np.asarray(upscaled_normalized.astype(mx.float32)),
    }
    metrics = {name: compare_tensors(name, reference[name], candidate) for name, candidate in candidates.items()}
    passed = all(
        result.nan_count == 0
        and result.inf_count == 0
        and result.normalized_rmse <= MAX_NORMALIZED_RMSE
        and result.cosine_similarity >= MIN_COSINE_SIMILARITY
        for result in metrics.values()
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "latent_spatial_upscaler_x2",
        "loaded_tensor_count": EXPECTED_UPSCALER_TENSORS + EXPECTED_STATISTICS_TENSORS,
        "acceptance_criteria": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "requires_finite_output": True,
        },
        "boundaries": {name: result.to_dict() for name, result in metrics.items()},
        "weight_load_memory": weight_load_memory,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
