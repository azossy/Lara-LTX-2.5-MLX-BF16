#!/usr/bin/env python3
"""Measure a canonical score-free LTX self-attention and FFN block workload."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from lara_ltx.config import ProjectConfig, load_project_config
from lara_ltx.errors import LaraError
from lara_ltx.runtime.memory import bf16_transformer_block_probe_bytes, stage_video_tokens
from lara_ltx.transformer.layers import gelu_approx, rms_norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.toml"
DEFAULT_BATCH_SIZE = 1
PROBE_RANDOM_SEED = 43
INPUT_RANDOM_LOW = -1.0
INPUT_RANDOM_HIGH = 1.0
WEIGHT_RANDOM_LOW = -0.02
WEIGHT_RANDOM_HIGH = 0.02
PROBE_ARRAY_COUNT = 7


def _memory_budget(config: ProjectConfig, device_info: dict[str, Any]) -> int:
    physical = int(device_info.get("memory_size", os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))
    fraction_budget = int(physical * config.memory.maximum_peak_fraction)
    reserve_budget = physical - config.memory.minimum_system_reserve_bytes
    recommended = int(device_info.get("max_recommended_working_set_size", physical))
    return min(fraction_budget, reserve_budget, recommended)


def _linear(x: mx.array, weight: mx.array) -> mx.array:
    return mx.matmul(x, mx.swapaxes(weight, -1, -2))


def _random_array(shape: tuple[int, ...], key: mx.array, *, is_weight: bool) -> mx.array:
    low = WEIGHT_RANDOM_LOW if is_weight else INPUT_RANDOM_LOW
    high = WEIGHT_RANDOM_HIGH if is_weight else INPUT_RANDOM_HIGH
    return mx.random.uniform(low=low, high=high, shape=shape, dtype=mx.bfloat16, key=key)


def run(config_path: Path) -> dict[str, Any]:
    if not mx.metal.is_available():
        raise LaraError("LARA-RUNTIME-003")
    config = load_project_config(config_path)
    canonical = config.canonical_hq
    tokens = stage_video_tokens(
        frames=canonical.num_frames,
        height=canonical.output_height,
        width=canonical.output_width,
        time_scale=canonical.vae_time_scale,
        spatial_scale=canonical.vae_spatial_scale,
    )
    estimated_bytes = bf16_transformer_block_probe_bytes(
        tokens=tokens,
        hidden_size=canonical.video_hidden_size,
        feed_forward_multiplier=canonical.feed_forward_multiplier,
    )
    device_info = dict(mx.device_info())
    budget_bytes = _memory_budget(config, device_info)
    if estimated_bytes > budget_bytes:
        raise LaraError(
            "LARA-RUNTIME-004",
            details={"estimated_bytes": estimated_bytes, "budget_bytes": budget_bytes},
        )

    hidden = canonical.video_hidden_size
    expanded = hidden * canonical.feed_forward_multiplier
    random_keys = mx.random.split(mx.random.key(PROBE_RANDOM_SEED), num=PROBE_ARRAY_COUNT)
    x = _random_array((DEFAULT_BATCH_SIZE, tokens, hidden), random_keys[0], is_weight=False)
    q_weight = _random_array((hidden, hidden), random_keys[1], is_weight=True)
    k_weight = _random_array((hidden, hidden), random_keys[2], is_weight=True)
    v_weight = _random_array((hidden, hidden), random_keys[3], is_weight=True)
    out_weight = _random_array((hidden, hidden), random_keys[4], is_weight=True)
    ff_in_weight = _random_array((expanded, hidden), random_keys[5], is_weight=True)
    ff_out_weight = _random_array((hidden, expanded), random_keys[6], is_weight=True)
    mx.eval(x, q_weight, k_weight, v_weight, out_weight, ff_in_weight, ff_out_weight)
    mx.clear_cache()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()

    heads = canonical.video_attention_heads
    head_dim = canonical.video_attention_head_dim
    if heads * head_dim != hidden:
        raise LaraError(
            "LARA-CONFIG-003",
            details={"key": "video_attention_shape", "value": (heads, head_dim, hidden), "path": str(config_path)},
        )

    block_started = time.perf_counter()
    attention_started = block_started
    normalized = rms_norm(x)
    query = mx.swapaxes(_linear(normalized, q_weight).reshape(DEFAULT_BATCH_SIZE, tokens, heads, head_dim), 1, 2)
    key = mx.swapaxes(_linear(normalized, k_weight).reshape(DEFAULT_BATCH_SIZE, tokens, heads, head_dim), 1, 2)
    value = mx.swapaxes(_linear(normalized, v_weight).reshape(DEFAULT_BATCH_SIZE, tokens, heads, head_dim), 1, 2)
    attended = mx.fast.scaled_dot_product_attention(query, key, value, scale=1.0 / math.sqrt(head_dim))
    attended = mx.swapaxes(attended, 1, 2).reshape(DEFAULT_BATCH_SIZE, tokens, hidden)
    residual = x + _linear(attended, out_weight)
    mx.eval(residual)
    attention_elapsed = time.perf_counter() - attention_started
    del value, key, query, normalized, attended

    feed_forward_started = time.perf_counter()
    expanded_state = gelu_approx(_linear(rms_norm(residual), ff_in_weight))
    output = residual + _linear(expanded_state, ff_out_weight)
    mx.eval(output)
    feed_forward_elapsed = time.perf_counter() - feed_forward_started
    block_elapsed = time.perf_counter() - block_started
    del expanded_state, residual

    return {
        "schema_version": 1,
        "mlx_version": importlib.metadata.version("mlx"),
        "device": device_info,
        "memory_budget_bytes": budget_bytes,
        "shape": {
            "tokens": tokens,
            "hidden_size": hidden,
            "heads": heads,
            "head_dim": head_dim,
            "feed_forward_size": expanded,
            "dtype": "bfloat16",
        },
        "estimated_score_free_bytes": estimated_bytes,
        "active_memory_before_bytes": active_before,
        "peak_evaluation_memory_bytes": int(mx.get_peak_memory()),
        "active_memory_after_bytes": int(mx.get_active_memory()),
        "cache_memory_after_bytes": int(mx.get_cache_memory()),
        "attention_seconds": attention_elapsed,
        "feed_forward_seconds": feed_forward_elapsed,
        "block_seconds": block_elapsed,
        "output_probe": float(output[0, 0, 0].item()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run(args.config)
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
