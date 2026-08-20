#!/usr/bin/env python3
"""Compare the checkpoint-backed MLX Audio VAE decoder with CUDA BF16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.audio_vae import load_audio_vae_decoder
from lara_ltx.errors import LaraError
from lara_ltx.parity.metrics import compare_tensors

ARTIFACT_SCHEMA_VERSION = 1
MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999
EXPECTED_DECODER_TENSORS = 56
EXPECTED_STATISTICS_TENSORS = 2
INPUT_KEY = "audio_vae_input_latent"
OUTPUT_KEY = "audio_vae_decoded_spectrogram"


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
            if INPUT_KEY not in archive.files or OUTPUT_KEY not in archive.files:
                missing = INPUT_KEY if INPUT_KEY not in archive.files else OUTPUT_KEY
                raise LaraError("LARA-PARITY-002", details={"key": missing})
            return archive[INPUT_KEY], archive[OUTPUT_KEY]
    except OSError as exc:
        raise LaraError("LARA-PARITY-002", details={"key": str(path)}) from exc


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    rules = mapping.get("rules")
    if not isinstance(rules, list):
        raise LaraError("LARA-MODEL-031", details={"key": "mapping_rules"})

    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    decoder = load_audio_vae_decoder(checkpoint=arguments.checkpoint, mapping=mapping)
    weight_load_memory = {
        "active_before_bytes": active_before,
        "active_after_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
    input_latent, expected = _load_reference(arguments.cuda_reference)
    output = decoder(mx.array(input_latent, dtype=mx.bfloat16))
    mx.eval(output)
    metrics = compare_tensors(OUTPUT_KEY, expected, np.asarray(output.astype(mx.float32)))
    passed = (
        metrics.nan_count == 0
        and metrics.inf_count == 0
        and metrics.normalized_rmse <= MAX_NORMALIZED_RMSE
        and metrics.cosine_similarity >= MIN_COSINE_SIMILARITY
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "audio_vae_decoder",
        "loaded_tensor_count": EXPECTED_DECODER_TENSORS + EXPECTED_STATISTICS_TENSORS,
        "acceptance_criteria": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "requires_finite_output": True,
        },
        "decoded_spectrogram": metrics.to_dict(),
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
