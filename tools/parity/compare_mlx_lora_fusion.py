#!/usr/bin/env python3
"""Compare MLX stage-local BF16 LoRA fusion with the fixed CUDA reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.models.lora_fusion import fuse_lora_weight
from lara_ltx.parity.metrics import compare_tensors

ARTIFACT_SCHEMA_VERSION = 1
MAX_ABSOLUTE_TOLERANCE = 2e-2
REQUIRED_ARRAY_KEYS = (
    "base_bf16_bits",
    "a_bf16_bits",
    "b_bf16_bits",
    "fused_stage_1_bf16_bits",
    "fused_stage_2_bf16_bits",
)
STAGE_NAMES = ("stage_1", "stage_2")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--cuda-report", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _from_bf16_bits(bits: np.ndarray) -> mx.array:
    if bits.dtype != np.uint16:
        raise LaraError("LARA-PARITY-003", details={"key": f"dtype={bits.dtype}"})
    return mx.array(bits, dtype=mx.uint16).view(mx.bfloat16)


def main() -> int:
    arguments = parse_arguments()
    metadata = json.loads(arguments.cuda_report.read_text(encoding="utf-8"))
    strengths = metadata.get("stage_strengths")
    invalid_strengths = not isinstance(strengths, dict) or any(
        not isinstance(strengths.get(stage), (int, float)) for stage in STAGE_NAMES
    )
    if invalid_strengths:
        raise LaraError("LARA-PARITY-003", details={"key": "stage_strengths"})
    with np.load(arguments.cuda_reference) as archive:
        missing = [key for key in REQUIRED_ARRAY_KEYS if key not in archive.files]
        if missing:
            raise LaraError("LARA-PARITY-003", details={"key": missing[0]})
        arrays = {key: archive[key] for key in REQUIRED_ARRAY_KEYS}

    base = _from_bf16_bits(arrays["base_bf16_bits"])
    a = _from_bf16_bits(arrays["a_bf16_bits"])
    b = _from_bf16_bits(arrays["b_bf16_bits"])
    mx.eval(base, a, b)
    active_memory_before_fusion = int(mx.get_active_memory())
    mx.reset_peak_memory()
    candidates = {stage: fuse_lora_weight(base, a, b, float(strengths[stage])) for stage in STAGE_NAMES}
    mx.eval(*candidates.values())
    metrics = {
        stage: compare_tensors(
            f"fused_{stage}",
            np.asarray(_from_bf16_bits(arrays[f"fused_{stage}_bf16_bits"]).astype(mx.float32)),
            np.asarray(candidate.astype(mx.float32)),
        )
        for stage, candidate in candidates.items()
    }
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "max_absolute_tolerance": MAX_ABSOLUTE_TOLERANCE,
        "stages": {stage: result.to_dict() for stage, result in metrics.items()},
        "fusion_memory": {
            "active_before_bytes": active_memory_before_fusion,
            "active_after_bytes": int(mx.get_active_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
        },
        "passed": max(result.max_abs_error for result in metrics.values()) <= MAX_ABSOLUTE_TOLERANCE,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
