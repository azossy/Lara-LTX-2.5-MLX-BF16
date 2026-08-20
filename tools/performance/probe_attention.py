#!/usr/bin/env python3
"""Measure canonical LTX fused-SDPA shapes on the target Apple Silicon GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from lara_ltx.config import ProjectConfig, load_project_config
from lara_ltx.errors import LaraError
from lara_ltx.runtime.memory import AttentionShape, bf16_attention_io_bytes, stage_video_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.toml"
DEFAULT_BATCH_SIZE = 1
PROBE_RANDOM_SEED = 42
PROBE_RANDOM_LOW = -1.0
PROBE_RANDOM_HIGH = 1.0
STAGE_ONE_SCALE_DIVISOR = 2
STAGE_CHOICES = ("stage1", "stage2", "all")


def _physical_memory_bytes() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    page_count = int(os.sysconf("SC_PHYS_PAGES"))
    return page_size * page_count


def _memory_budget(config: ProjectConfig, device_info: dict[str, Any]) -> int:
    physical = _physical_memory_bytes()
    fraction_budget = int(physical * config.memory.maximum_peak_fraction)
    reserve_budget = physical - config.memory.minimum_system_reserve_bytes
    recommended = int(device_info.get("max_recommended_working_set_size", physical))
    return min(fraction_budget, reserve_budget, recommended)


def _tokens(config: ProjectConfig, stage: str) -> int:
    shape = config.canonical_hq
    divisor = STAGE_ONE_SCALE_DIVISOR if stage == "stage1" else 1
    return stage_video_tokens(
        frames=shape.num_frames,
        height=shape.output_height // divisor,
        width=shape.output_width // divisor,
        time_scale=shape.vae_time_scale,
        spatial_scale=shape.vae_spatial_scale,
    )


def _probe_stage(config: ProjectConfig, stage: str, budget_bytes: int) -> dict[str, int | float | str]:
    tokens = _tokens(config, stage)
    shape = AttentionShape(
        batch_size=DEFAULT_BATCH_SIZE,
        heads=config.canonical_hq.video_attention_heads,
        tokens=tokens,
        head_dim=config.canonical_hq.video_attention_head_dim,
    )
    estimated_bytes = bf16_attention_io_bytes(shape)
    if estimated_bytes > budget_bytes:
        raise LaraError(
            "LARA-RUNTIME-004",
            details={"estimated_bytes": estimated_bytes, "budget_bytes": budget_bytes},
        )

    mx.clear_cache()
    tensor_shape = (shape.batch_size, shape.heads, shape.tokens, shape.head_dim)
    random_keys = mx.random.split(mx.random.key(PROBE_RANDOM_SEED), num=3)
    query, key, value = (
        mx.random.uniform(
            low=PROBE_RANDOM_LOW,
            high=PROBE_RANDOM_HIGH,
            shape=tensor_shape,
            dtype=mx.bfloat16,
            key=random_key,
        )
        for random_key in random_keys
    )
    mx.eval(query, key, value)
    active_before = int(mx.get_active_memory())
    cache_before = int(mx.get_cache_memory())
    mx.reset_peak_memory()

    started = time.perf_counter()
    output = mx.fast.scaled_dot_product_attention(query, key, value, scale=shape.head_dim**-0.5)
    mx.eval(output)
    elapsed = time.perf_counter() - started

    result: dict[str, int | float | str] = {
        "stage": stage,
        "tokens": tokens,
        "heads": shape.heads,
        "head_dim": shape.head_dim,
        "dtype": "bfloat16",
        "estimated_qkv_output_bytes": estimated_bytes,
        "active_memory_before_bytes": active_before,
        "cache_memory_before_bytes": cache_before,
        "peak_evaluation_memory_bytes": int(mx.get_peak_memory()),
        "active_memory_after_bytes": int(mx.get_active_memory()),
        "cache_memory_after_bytes": int(mx.get_cache_memory()),
        "elapsed_seconds": elapsed,
        "output_probe": float(output[0, 0, 0, 0].item()),
    }
    del output, value, key, query
    mx.clear_cache()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage", choices=STAGE_CHOICES, default="all")
    parser.add_argument("--output", type=Path)
    return parser


def run(config_path: Path, stage: str) -> dict[str, Any]:
    if not mx.metal.is_available():
        raise LaraError("LARA-RUNTIME-003")
    config = load_project_config(config_path)
    device_info = dict(mx.device_info())
    budget = _memory_budget(config, device_info)
    stages = ("stage1", "stage2") if stage == "all" else (stage,)
    return {
        "schema_version": 1,
        "mlx_version": importlib.metadata.version("mlx"),
        "device": device_info,
        "memory_budget_bytes": budget,
        "results": [_probe_stage(config, name, budget) for name in stages],
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run(args.config, args.stage)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(f"{rendered}\n", encoding="utf-8")
        print(rendered)
        return 0
    except LaraError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
