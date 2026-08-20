"""Reviewed mapping contract for LTX-2.5 video/audio prompt connectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

CHECKPOINT_METADATA_KEY = "config"
SOURCE_ROOT = "model.diffusion_model."
CONNECTOR_SUFFIX = "_embeddings_connector."
MODALITIES = ("video", "audio")
FEED_FORWARD_MULTIPLIER = 4


@dataclass(frozen=True)
class PromptConnectorConfig:
    video_dimensions: int
    audio_dimensions: int
    video_attention_heads: int
    audio_attention_heads: int
    video_attention_head_dim: int
    audio_attention_head_dim: int
    layer_count: int
    register_count: int
    positional_theta: float
    positional_maximum: int
    apply_gated_attention: bool
    double_precision_frequencies: bool

    def dimensions(self, modality: str) -> int:
        return self.video_dimensions if modality == "video" else self.audio_dimensions

    def heads(self, modality: str) -> int:
        return self.video_attention_heads if modality == "video" else self.audio_attention_heads


def load_prompt_connector_config(checkpoint: Path) -> PromptConnectorConfig:
    source = inspect_safetensors(checkpoint)
    try:
        metadata = json.loads(source.metadata[CHECKPOINT_METADATA_KEY])
        transformer = metadata["transformer"]
        video_heads = int(transformer["connector_num_attention_heads"])
        audio_heads = int(transformer["audio_connector_num_attention_heads"])
        video_head_dim = int(transformer["connector_attention_head_dim"])
        audio_head_dim = int(transformer["audio_connector_attention_head_dim"])
        maximums = transformer["connector_positional_embedding_max_pos"]
        config = PromptConnectorConfig(
            video_dimensions=video_heads * video_head_dim,
            audio_dimensions=audio_heads * audio_head_dim,
            video_attention_heads=video_heads,
            audio_attention_heads=audio_heads,
            video_attention_head_dim=video_head_dim,
            audio_attention_head_dim=audio_head_dim,
            layer_count=int(transformer["connector_num_layers"]),
            register_count=int(transformer["connector_num_learnable_registers"]),
            positional_theta=float(transformer.get("positional_embedding_theta", 10_000.0)),
            positional_maximum=int(maximums[0]),
            apply_gated_attention=bool(transformer["connector_apply_gated_attention"]),
            double_precision_frequencies=transformer["frequencies_precision"] == "float64",
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, IndexError) as error:
        raise LaraError("LARA-MODEL-038", details={"key": "connector_config"}) from error
    if (
        not transformer.get("use_embeddings_connector")
        or config.layer_count <= 0
        or config.register_count <= 0
        or config.positional_maximum <= 0
        or not config.apply_gated_attention
    ):
        raise LaraError("LARA-MODEL-038", details={"key": "unsupported_connector_config"})
    return config


def prompt_connector_target_shapes(config: PromptConnectorConfig) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for modality in MODALITIES:
        dimensions = config.dimensions(modality)
        heads = config.heads(modality)
        root = f"{modality}_connector"
        shapes[f"{root}.learnable_registers"] = (config.register_count, dimensions)
        for index in range(config.layer_count):
            prefix = f"{root}.transformer_1d_blocks.{index}"
            shapes.update(
                {
                    f"{prefix}.attn1.q_norm.weight": (dimensions,),
                    f"{prefix}.attn1.k_norm.weight": (dimensions,),
                    f"{prefix}.attn1.to_q.weight": (dimensions, dimensions),
                    f"{prefix}.attn1.to_q.bias": (dimensions,),
                    f"{prefix}.attn1.to_k.weight": (dimensions, dimensions),
                    f"{prefix}.attn1.to_k.bias": (dimensions,),
                    f"{prefix}.attn1.to_v.weight": (dimensions, dimensions),
                    f"{prefix}.attn1.to_v.bias": (dimensions,),
                    f"{prefix}.attn1.to_gate_logits.weight": (heads, dimensions),
                    f"{prefix}.attn1.to_gate_logits.bias": (heads,),
                    f"{prefix}.attn1.to_out.0.weight": (dimensions, dimensions),
                    f"{prefix}.attn1.to_out.0.bias": (dimensions,),
                    f"{prefix}.ff.net.0.proj.weight": (
                        dimensions * FEED_FORWARD_MULTIPLIER,
                        dimensions,
                    ),
                    f"{prefix}.ff.net.0.proj.bias": (dimensions * FEED_FORWARD_MULTIPLIER,),
                    f"{prefix}.ff.net.2.weight": (
                        dimensions,
                        dimensions * FEED_FORWARD_MULTIPLIER,
                    ),
                    f"{prefix}.ff.net.2.bias": (dimensions,),
                }
            )
    return shapes


def build_prompt_connector_mapping(checkpoint: Path) -> dict[str, object]:
    config = load_prompt_connector_config(checkpoint)
    shapes = prompt_connector_target_shapes(config)
    descriptors = {descriptor.name: descriptor for descriptor in inspect_safetensors(checkpoint).tensors}
    rules: list[dict[str, object]] = []
    for target_key in sorted(shapes):
        modality, suffix = target_key.split("_connector.", maxsplit=1)
        source_key = f"{SOURCE_ROOT}{modality}{CONNECTOR_SUFFIX}{suffix}"
        descriptor = descriptors.get(source_key)
        if descriptor is None:
            raise LaraError("LARA-MODEL-038", details={"key": source_key})
        rules.append(
            {
                "source_file": checkpoint.name,
                "source_key": source_key,
                "target_key": target_key,
                "transform": IDENTITY_TRANSFORM,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
            }
        )
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "gemma_prompt_connectors",
        "shards": [{"file": checkpoint.name, "tensor_count": len(rules)}],
        "rules": rules,
    }
    validate_prompt_connector_mapping(mapping, config)
    return mapping


def validate_prompt_connector_mapping(
    mapping: dict[str, object],
    config: PromptConnectorConfig,
) -> tuple[dict[str, object], ...]:
    shapes = prompt_connector_target_shapes(config)
    rules = validate_mapping(mapping, expected_target_keys=shapes)
    for rule in rules:
        target_key = rule.get("target_key")
        if (
            not isinstance(target_key, str)
            or rule.get("transform") != IDENTITY_TRANSFORM
            or rule.get("dtype") != BF16_DTYPE
            or tuple(rule.get("shape", ())) != shapes[target_key]
        ):
            raise LaraError("LARA-MODEL-038", details={"key": str(target_key)})
    return rules
