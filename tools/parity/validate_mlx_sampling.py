#!/usr/bin/env python3
"""Validate MLX scheduler and guidance against frozen HQ sampling contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.sampling import LTX2Scheduler, MultiModalGuider, MultiModalGuiderParams, get_res2s_coefficients

ARTIFACT_SCHEMA_VERSION = 1
HQ_STAGE_1_STEPS = 15
HQ_STAGE_1_TOKEN_COUNT = 120
MAX_SIGMA_ABSOLUTE_ERROR = 2e-7


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    with np.load(arguments.cuda_reference) as archive:
        expected_sigmas = archive["stage_1_sigmas"]
    actual_sigmas = np.asarray(LTX2Scheduler().execute(steps=HQ_STAGE_1_STEPS, token_count=HQ_STAGE_1_TOKEN_COUNT))
    sigma_error = float(np.max(np.abs(expected_sigmas - actual_sigmas)))

    guider = MultiModalGuider(MultiModalGuiderParams(cfg_scale=3.0, stg_scale=0.5, modality_scale=2.0))
    guided = guider.calculate(
        mx.full((1, 2, 3), 4.0),
        uncond=mx.full((1, 2, 3), 1.0),
        perturbed=mx.full((1, 2, 3), 2.0),
        isolated=mx.full((1, 2, 3), 3.0),
    )
    mx.eval(guided)
    guidance_error = float(np.max(np.abs(np.asarray(guided) - 12.0)))
    coefficients = get_res2s_coefficients(0.25)
    passed = sigma_error <= MAX_SIGMA_ABSOLUTE_ERROR and guidance_error == 0.0 and all(np.isfinite(coefficients))
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "sampling",
        "cuda_reference": arguments.cuda_reference.name,
        "stage_1_sigma_count": len(actual_sigmas),
        "max_sigma_absolute_error": sigma_error,
        "max_guidance_absolute_error": guidance_error,
        "res2s_coefficients_h_0_25": list(coefficients),
        "acceptance_criteria": {
            "max_sigma_absolute_error": MAX_SIGMA_ABSOLUTE_ERROR,
            "max_guidance_absolute_error": 0.0,
            "requires_finite_res2s_coefficients": True,
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
