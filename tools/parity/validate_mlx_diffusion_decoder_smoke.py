#!/usr/bin/env python3
"""Decode the fixed CUDA latent through the official MLX Diffusion Video VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.video_vae import load_diffusion_video_decoder

CONFIG_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-TENSOR-016", details={"reason": "unreadable_decoder_smoke_config"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_decoder_smoke_config"})
    return value


def _required(config: dict[str, Any], key: str, expected: type) -> Any:
    value = config.get(key)
    if not isinstance(value, expected) or (expected in (int, float) and isinstance(value, bool)):
        raise LaraError("LARA-TENSOR-016", details={"reason": f"invalid_decoder_config_{key}"})
    return value


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-TENSOR-016", details={"reason": "unsupported_decoder_smoke_schema"})
    mapping = _read_json(arguments.mapping)
    latent_key = _required(config, "latent_key", str)
    expected_frames_key = _required(config, "expected_frames_key", str)
    noise_seed = _required(config, "noise_seed", int)
    activation_budget_bytes = _required(config, "activation_budget_bytes", int)
    timestep_values = _required(config, "timesteps", list)
    try:
        timesteps = mx.array([timestep_values], dtype=mx.float32)
    except (TypeError, ValueError) as error:
        raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_decoder_smoke_timesteps"}) from error
    with np.load(arguments.cuda_reference) as archive:
        try:
            latent_source = archive[latent_key]
            expected_frames = archive[expected_frames_key]
        except KeyError as error:
            raise LaraError("LARA-TENSOR-016", details={"reason": "missing_decoder_smoke_boundary"}) from error
    if latent_source.ndim != 5 or expected_frames.ndim != 4:
        raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_decoder_smoke_boundary"})

    decoder = load_diffusion_video_decoder(checkpoint=arguments.checkpoint, mapping=mapping)
    latent = mx.array(latent_source, dtype=mx.bfloat16)
    output_shape = decoder.output_shape_for_latent(tuple(latent.shape[2:5]))
    noise_key = mx.random.key(noise_seed)
    initial_noise = mx.random.normal(
        (latent.shape[0], decoder.out_channels, *output_shape),
        key=noise_key,
        dtype=mx.bfloat16,
    )
    tiling = decoder.recommend_tiling(
        tuple(latent.shape[2:5]),
        activation_budget_bytes=activation_budget_bytes,
        batch_size=latent.shape[0],
    )
    mx.reset_peak_memory()
    started = time.monotonic()
    pixels = decoder.decode(
        latent,
        initial_noise,
        timesteps=timesteps,
        stage4_tiling=tiling.stage4,
        stage5_tiling=tiling.stage5,
    )
    finite = mx.all(mx.isfinite(pixels))
    mx.eval(pixels, finite)
    elapsed_seconds = time.monotonic() - started
    output = np.asarray(pixels.astype(mx.float32))
    expected_ncthw_shape = (
        latent_source.shape[0],
        expected_frames.shape[3],
        expected_frames.shape[0],
        expected_frames.shape[1],
        expected_frames.shape[2],
    )
    passed = bool(finite.item()) and output.shape == expected_ncthw_shape
    report = {
        "schema_version": 1,
        "component": "diffusion_video_decoder_fixed_latent_smoke",
        "configuration": config,
        "input_shape": list(latent.shape),
        "expected_output_shape": list(expected_ncthw_shape),
        "output": {
            "shape": list(output.shape),
            "finite": bool(finite.item()),
            "minimum": float(output.min()),
            "maximum": float(output.max()),
            "mean": float(output.mean()),
            "standard_deviation": float(output.std()),
            "sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        },
        "tiling": {
            "stage4": list(tiling.stage4.tile_shape),
            "stage5": list(tiling.stage5.tile_shape),
            "resident_stage3_bytes": tiling.resident_stage3_bytes,
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_bytes": int(mx.get_peak_memory()),
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
