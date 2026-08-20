#!/usr/bin/env python3
"""Compare the checkpoint-backed MLX primary vocoder with CUDA FP32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.audio_vae import load_primary_vocoder
from lara_ltx.errors import LaraError
from lara_ltx.parity.metrics import compare_tensors

ARTIFACT_SCHEMA_VERSION = 1
MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999
INPUT_KEY = "primary_vocoder_input"
OUTPUT_KEY = "primary_vocoder_waveform"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path) as archive:
            missing = next((key for key in (INPUT_KEY, OUTPUT_KEY) if key not in archive.files), None)
            if missing is not None:
                raise LaraError("LARA-PARITY-002", details={"key": missing})
            return archive[INPUT_KEY], archive[OUTPUT_KEY]
    except OSError as exc:
        raise LaraError("LARA-PARITY-002", details={"key": str(path)}) from exc


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    mx.reset_peak_memory()
    model = load_primary_vocoder(checkpoint=arguments.checkpoint, mapping=mapping)
    loaded_memory = int(mx.get_active_memory())
    source, expected = _load_reference(arguments.cuda_reference)
    output = model(mx.array(source, dtype=mx.float32))
    mx.eval(output)
    metrics = compare_tensors(OUTPUT_KEY, expected, np.asarray(output))
    passed = (
        metrics.nan_count == 0
        and metrics.inf_count == 0
        and metrics.normalized_rmse <= MAX_NORMALIZED_RMSE
        and metrics.cosine_similarity >= MIN_COSINE_SIMILARITY
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "primary_audio_vocoder",
        "acceptance_criteria": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "requires_finite_output": True,
        },
        "waveform": metrics.to_dict(),
        "memory": {
            "active_after_load_bytes": loaded_memory,
            "peak_bytes": int(mx.get_peak_memory()),
        },
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
