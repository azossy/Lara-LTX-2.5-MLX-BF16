"""Reviewed mapping contract for the LTX Gemma V2 feature extractor."""

from __future__ import annotations

import json
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

GEMMA_HIDDEN_DIMENSION = 3840
GEMMA_HIDDEN_STATE_COUNT = 49
GEMMA_FEATURE_INPUT_DIMENSION = GEMMA_HIDDEN_DIMENSION * GEMMA_HIDDEN_STATE_COUNT
GEMMA_VIDEO_FEATURE_DIMENSION = 4096
GEMMA_AUDIO_FEATURE_DIMENSION = 2048
GEMMA_FEATURE_SOURCE_PREFIX = "text_embedding_projection."
GEMMA_FEATURE_TARGET_PREFIX = "feature_extractor."


def gemma_feature_target_shapes() -> dict[str, tuple[int, ...]]:
    """Return the four projection tensors consumed after Gemma hidden states."""

    return {
        "feature_extractor.video_aggregate_embed.weight": (
            GEMMA_VIDEO_FEATURE_DIMENSION,
            GEMMA_FEATURE_INPUT_DIMENSION,
        ),
        "feature_extractor.video_aggregate_embed.bias": (GEMMA_VIDEO_FEATURE_DIMENSION,),
        "feature_extractor.audio_aggregate_embed.weight": (
            GEMMA_AUDIO_FEATURE_DIMENSION,
            GEMMA_FEATURE_INPUT_DIMENSION,
        ),
        "feature_extractor.audio_aggregate_embed.bias": (GEMMA_AUDIO_FEATURE_DIMENSION,),
    }


def build_gemma_feature_mapping(shard_path: Path) -> dict[str, object]:
    """Map only the BF16 LTX feature-extractor projection from the Gemma shard.

    Gemma language-model weights and serialized tokenizer assets belong to
    their own loader path. This mapping deliberately excludes both, avoiding
    accidental treatment of U8 assets as model parameters.
    """

    source = inspect_safetensors(shard_path)
    target_shapes = gemma_feature_target_shapes()
    by_source = {descriptor.name: descriptor for descriptor in source.tensors}
    rules: list[dict[str, object]] = []
    for target_key in sorted(target_shapes):
        source_key = f"{GEMMA_FEATURE_SOURCE_PREFIX}{target_key.removeprefix(GEMMA_FEATURE_TARGET_PREFIX)}"
        descriptor = by_source.get(source_key)
        if descriptor is None:
            raise LaraError("LARA-MODEL-029", details={"key": source_key})
        rules.append(
            {
                "source_file": shard_path.name,
                "source_key": source_key,
                "target_key": target_key,
                "transform": IDENTITY_TRANSFORM,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
            }
        )
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "gemma_feature_extractor_v2",
        "shards": [{"file": shard_path.name, "tensor_count": len(rules)}],
        "rules": rules,
    }
    validate_gemma_feature_mapping(mapping)
    return mapping


def validate_gemma_feature_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Require exact BF16 shape-compatible Gemma V2 projection rules."""

    target_shapes = gemma_feature_target_shapes()
    rules = validate_mapping(mapping, expected_target_keys=target_shapes)
    for rule in rules:
        target_key = rule["target_key"]
        source_key = rule["source_key"]
        source_shape = rule.get("shape")
        if not isinstance(target_key, str) or not isinstance(source_key, str) or not isinstance(source_shape, list):
            raise LaraError("LARA-MODEL-014", details={"source_key": "gemma_feature_mapping_rule"})
        if (
            rule["transform"] != IDENTITY_TRANSFORM
            or rule.get("dtype") != BF16_DTYPE
            or tuple(source_shape) != target_shapes[target_key]
        ):
            raise LaraError("LARA-MODEL-029", details={"key": source_key})
    return rules


def write_gemma_feature_mapping(output: Path, shard_path: Path) -> None:
    """Atomically write the reviewed Gemma V2 feature-extractor mapping."""

    mapping = build_gemma_feature_mapping(shard_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc
