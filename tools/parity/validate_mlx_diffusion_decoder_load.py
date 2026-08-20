#!/usr/bin/env python3
"""Strict-load the official 407-target Diffusion Video VAE decoder on MLX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from lara_ltx.video_vae import load_diffusion_video_decoder
from mlx.utils import tree_flatten

ARTIFACT_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    decoder = load_diffusion_video_decoder(checkpoint=arguments.checkpoint, mapping=mapping)
    parameters = dict(tree_flatten(decoder.parameters()))
    dtype_counts: dict[str, int] = {}
    for value in parameters.values():
        dtype = str(value.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    mapped_count = len(mapping.get("rules", ()))
    passed = len(parameters) == mapped_count and dtype_counts == {str(mx.bfloat16): mapped_count}
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "diffusion_video_decoder_checkpoint_load",
        "mapped_target_count": mapped_count,
        "loaded_parameter_count": len(parameters),
        "dtype_counts": dtype_counts,
        "weight_load_memory": {
            "active_before_bytes": active_before,
            "active_after_bytes": int(mx.get_active_memory()),
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
