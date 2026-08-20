#!/usr/bin/env python3
"""Compare complete packed-Gemma prompt contexts with the fixed CUDA HQ trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.parity import compare_tensors
from lara_ltx.pipeline.api import _load_text_contexts
from lara_ltx.pipeline.configuration import load_pipeline_profile


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--connector-checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    profile = load_pipeline_profile(arguments.profile)
    contexts = _load_text_contexts(
        arguments.checkpoint,
        arguments.connector_checkpoint,
        prompt=arguments.prompt,
        negative_prompt=arguments.negative_prompt or profile.generation.negative_prompt,
        max_length=arguments.max_length,
    )
    candidates = {
        "prompt_video_positive": np.asarray(contexts.video_positive.astype(mx.float32)),
        "prompt_video_negative": np.asarray(contexts.video_negative.astype(mx.float32)),
        "prompt_audio_positive": np.asarray(contexts.audio_positive.astype(mx.float32)),
        "prompt_audio_negative": np.asarray(contexts.audio_negative.astype(mx.float32)),
    }
    with np.load(arguments.reference) as archive:
        results = {
            key: compare_tensors(key, archive[key].astype(np.float32), candidate).to_dict()
            for key, candidate in candidates.items()
        }
    report = {
        "schema_version": 1,
        "component": "full_gemma_prompt_context",
        "results": results,
        "passed": all(result["normalized_rmse"] <= 0.02 for result in results.values()),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
