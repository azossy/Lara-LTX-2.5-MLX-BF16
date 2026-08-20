#!/usr/bin/env python3
"""Strict-load and execute the packed Gemma 4 LTX text conditioner on MLX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.models import (
    build_packed_gemma_tokenizer,
    gemma_feature_target_shapes,
    gemma_text_target_shapes,
    iter_component_weight_batches,
    load_packed_gemma_assets,
    load_packed_gemma_config,
    tokenize_prompts,
    validate_gemma_feature_mapping,
    validate_gemma_text_mapping,
)
from lara_ltx.text_encoder import LTXGemma4TextEncoder, build_gemma4_text_args
from mlx.utils import tree_flatten

ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 16


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-checkpoint", required=True, type=Path)
    parser.add_argument("--text-mapping", required=True, type=Path)
    parser.add_argument("--feature-mapping", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaraError("LARA-MODEL-014", details={"source_key": str(path)}) from exc
    if not isinstance(mapping, dict):
        raise LaraError("LARA-MODEL-014", details={"source_key": str(path)})
    return mapping


def main() -> int:
    arguments = parse_arguments()
    if arguments.sequence_length <= 0:
        raise LaraError("LARA-TENSOR-022", details={"reason": "invalid_sequence_length"})
    packed_config = load_packed_gemma_config(arguments.gemma_checkpoint)
    text_rules = validate_gemma_text_mapping(_load_mapping(arguments.text_mapping), packed_config)
    feature_rules = validate_gemma_feature_mapping(_load_mapping(arguments.feature_mapping))
    encoder = LTXGemma4TextEncoder(build_gemma4_text_args(packed_config))
    expected_keys = set(gemma_text_target_shapes(packed_config)) | set(gemma_feature_target_shapes())
    actual_keys = {name for name, _ in tree_flatten(encoder.parameters())}
    if actual_keys != expected_keys:
        raise LaraError(
            "LARA-MODEL-017",
            details={
                "missing": ",".join(sorted(expected_keys - actual_keys)[:3]),
                "unexpected": ",".join(sorted(actual_keys - expected_keys)[:3]),
            },
        )

    active_memory_before_loading = int(mx.get_active_memory())
    mx.reset_peak_memory()
    for rules, shapes in (
        (text_rules, gemma_text_target_shapes(packed_config)),
        (feature_rules, gemma_feature_target_shapes()),
    ):
        for batch in iter_component_weight_batches(
            (arguments.gemma_checkpoint,),
            rules,
            expected_target_shapes=shapes,
        ):
            encoder.load_weights(batch, strict=False)
            mx.eval(*[value for _, value in batch])
    weight_load_memory = {
        "active_before_bytes": active_memory_before_loading,
        "active_after_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }

    assets = load_packed_gemma_assets(arguments.gemma_checkpoint)
    tokenizer = build_packed_gemma_tokenizer(assets, max_length=arguments.sequence_length)
    token_ids, attention_mask = tokenize_prompts(
        tokenizer,
        [arguments.prompt],
        max_length=arguments.sequence_length,
    )
    video, audio = encoder.encode_features(
        mx.array(token_ids, dtype=mx.int32),
        mx.array(attention_mask, dtype=mx.int32),
    )
    finite = (mx.all(mx.isfinite(video)), mx.all(mx.isfinite(audio)))
    mx.eval(video, audio, *finite)
    passed = all(bool(np.asarray(value)) for value in finite)
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "gemma4_ltx_text_conditioner",
        "checkpoint": arguments.gemma_checkpoint.name,
        "text_tensor_count": len(text_rules),
        "feature_tensor_count": len(feature_rules),
        "tokenizer_asset_count": 1 + len(assets.sidecars),
        "sequence_length": arguments.sequence_length,
        "valid_token_count": int(attention_mask.sum()),
        "hidden_state_count": packed_config["text_config"]["num_hidden_layers"] + 1,
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "weight_load_memory": weight_load_memory,
        "requires_finite_output": True,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
