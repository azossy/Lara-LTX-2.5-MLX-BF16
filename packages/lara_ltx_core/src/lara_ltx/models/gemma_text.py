"""Exact Gemma 4 text-core mapping for the packed LTX-2.5 encoder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

GEMMA_CONFIG_METADATA_KEY = "gemma_config"
SLIDING_ATTENTION_TYPE = "sliding_attention"
FULL_ATTENTION_TYPE = "full_attention"
SUPPORTED_ATTENTION_TYPES = frozenset((SLIDING_ATTENTION_TYPE, FULL_ATTENTION_TYPE))
EXPECTED_UNIFIED_MODEL_TYPE = "gemma4_unified"
EXPECTED_TEXT_MODEL_TYPE = "gemma4_unified_text"


def load_packed_gemma_config(checkpoint: Path) -> dict[str, Any]:
    """Load and validate the JSON configuration embedded in the packed shard."""

    source = inspect_safetensors(checkpoint)
    raw = source.metadata.get(GEMMA_CONFIG_METADATA_KEY)
    try:
        config = json.loads(raw) if raw is not None else None
    except json.JSONDecodeError as exc:
        raise LaraError("LARA-MODEL-035", details={"reason": "invalid_config_json"}) from exc
    if not isinstance(config, dict) or not isinstance(config.get("text_config"), dict):
        raise LaraError("LARA-MODEL-035", details={"reason": "missing_text_config"})
    if (
        config.get("model_type") != EXPECTED_UNIFIED_MODEL_TYPE
        or config["text_config"].get("model_type") != EXPECTED_TEXT_MODEL_TYPE
    ):
        raise LaraError("LARA-MODEL-035", details={"reason": "unsupported_model_type"})
    return config


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LaraError("LARA-MODEL-035", details={"reason": f"invalid_{key}"})
    return value


def gemma_text_target_shapes(packed_config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Derive the complete logits-free text-core parameter contract."""

    raw_text = packed_config.get("text_config")
    if not isinstance(raw_text, dict):
        raise LaraError("LARA-MODEL-035", details={"reason": "missing_text_config"})
    hidden = _positive_int(raw_text, "hidden_size")
    layer_count = _positive_int(raw_text, "num_hidden_layers")
    intermediate = _positive_int(raw_text, "intermediate_size")
    attention_heads = _positive_int(raw_text, "num_attention_heads")
    head_dimension = _positive_int(raw_text, "head_dim")
    global_head_dimension = _positive_int(raw_text, "global_head_dim")
    key_value_heads = _positive_int(raw_text, "num_key_value_heads")
    global_key_value_heads = _positive_int(raw_text, "num_global_key_value_heads")
    vocabulary = _positive_int(raw_text, "vocab_size")
    layer_types = raw_text.get("layer_types")
    unsupported_features = (
        raw_text.get("enable_moe_block") is not False
        or raw_text.get("hidden_size_per_layer_input") != 0
        or raw_text.get("num_kv_shared_layers") != 0
        or raw_text.get("use_double_wide_mlp") is not False
        or not isinstance(layer_types, list)
        or len(layer_types) != layer_count
        or any(value not in SUPPORTED_ATTENTION_TYPES for value in layer_types)
    )
    if unsupported_features:
        raise LaraError("LARA-MODEL-035", details={"reason": "unsupported_text_architecture"})

    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocabulary, hidden),
        "model.norm.weight": (hidden,),
    }
    for layer_index, layer_type in enumerate(layer_types):
        prefix = f"model.layers.{layer_index}"
        is_full_attention = layer_type == FULL_ATTENTION_TYPE
        layer_head_dimension = global_head_dimension if is_full_attention else head_dimension
        layer_key_value_heads = global_key_value_heads if is_full_attention else key_value_heads
        attention_width = attention_heads * layer_head_dimension
        key_value_width = layer_key_value_heads * layer_head_dimension
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.layer_scalar": (1,),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.post_feedforward_layernorm.weight": (hidden,),
                f"{prefix}.pre_feedforward_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.k_norm.weight": (layer_head_dimension,),
                f"{prefix}.self_attn.k_proj.weight": (key_value_width, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, attention_width),
                f"{prefix}.self_attn.q_norm.weight": (layer_head_dimension,),
                f"{prefix}.self_attn.q_proj.weight": (attention_width, hidden),
            }
        )
        if not (is_full_attention and raw_text.get("attention_k_eq_v") is True):
            shapes[f"{prefix}.self_attn.v_proj.weight"] = (key_value_width, hidden)
    return shapes


def build_gemma_text_mapping(checkpoint: Path) -> dict[str, object]:
    packed_config = load_packed_gemma_config(checkpoint)
    target_shapes = gemma_text_target_shapes(packed_config)
    source = inspect_safetensors(checkpoint)
    descriptors = {descriptor.name: descriptor for descriptor in source.tensors}
    rules: list[dict[str, object]] = []
    for target_key in sorted(target_shapes):
        descriptor = descriptors.get(target_key)
        if descriptor is None:
            raise LaraError("LARA-MODEL-035", details={"reason": f"missing_{target_key}"})
        rules.append(
            {
                "source_file": checkpoint.name,
                "source_key": target_key,
                "target_key": target_key,
                "transform": IDENTITY_TRANSFORM,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
            }
        )
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "gemma4_text_core",
        "shards": [{"file": checkpoint.name, "tensor_count": len(rules)}],
        "rules": rules,
    }
    validate_gemma_text_mapping(mapping, packed_config)
    return mapping


def validate_gemma_text_mapping(
    mapping: dict[str, object],
    packed_config: dict[str, Any],
) -> tuple[dict[str, object], ...]:
    target_shapes = gemma_text_target_shapes(packed_config)
    rules = validate_mapping(mapping, expected_target_keys=target_shapes)
    for rule in rules:
        target_key = rule.get("target_key")
        if (
            not isinstance(target_key, str)
            or rule.get("source_key") != target_key
            or rule.get("transform") != IDENTITY_TRANSFORM
            or rule.get("dtype") != BF16_DTYPE
            or tuple(rule.get("shape", ())) != target_shapes[target_key]
        ):
            raise LaraError("LARA-MODEL-035", details={"reason": str(rule.get("source_key"))})
    return rules


def write_gemma_text_mapping(output: Path, checkpoint: Path) -> None:
    mapping = build_gemma_text_mapping(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc
