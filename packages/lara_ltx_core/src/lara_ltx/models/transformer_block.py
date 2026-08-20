"""Exact checkpoint contract for one full LTX-2.5 AV transformer block."""

from __future__ import annotations

import json
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

VIDEO_DIMENSION = 4096
VIDEO_HEAD_COUNT = 32
AUDIO_DIMENSION = 2048
AUDIO_HEAD_COUNT = 32
VIDEO_FEED_FORWARD_DIMENSION = 16384
AUDIO_FEED_FORWARD_DIMENSION = 8192
TRANSFORMER_BLOCK_SOURCE_PREFIX = "model.diffusion_model.transformer_blocks."
TRANSFORMER_BLOCK_F32_TARGETS = frozenset(
    {
        "scale_shift_table",
        "audio_scale_shift_table",
        "prompt_scale_shift_table",
        "audio_prompt_scale_shift_table",
        "scale_shift_table_a2v_ca_video",
        "scale_shift_table_a2v_ca_audio",
    }
)


def transformer_block_target_shapes() -> dict[str, tuple[int, ...]]:
    """Return the 84 MLX parameter shapes for a production AV block.

    The names intentionally match ``AVTransformerBlock``'s MLX module layout,
    including the PyTorch-compatible ``to_out.0`` and ``net.0.proj`` paths.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    _add_attention_shapes(
        shapes,
        prefix="attn1",
        query_dim=VIDEO_DIMENSION,
        context_dim=VIDEO_DIMENSION,
        output_dim=VIDEO_DIMENSION,
        gate_input_dim=VIDEO_DIMENSION,
    )
    _add_attention_shapes(
        shapes,
        prefix="attn2",
        query_dim=VIDEO_DIMENSION,
        context_dim=VIDEO_DIMENSION,
        output_dim=VIDEO_DIMENSION,
        gate_input_dim=VIDEO_DIMENSION,
    )
    _add_attention_shapes(
        shapes,
        prefix="audio_attn1",
        query_dim=AUDIO_DIMENSION,
        context_dim=AUDIO_DIMENSION,
        output_dim=AUDIO_DIMENSION,
        gate_input_dim=AUDIO_DIMENSION,
    )
    _add_attention_shapes(
        shapes,
        prefix="audio_attn2",
        query_dim=AUDIO_DIMENSION,
        context_dim=AUDIO_DIMENSION,
        output_dim=AUDIO_DIMENSION,
        gate_input_dim=AUDIO_DIMENSION,
    )
    _add_attention_shapes(
        shapes,
        prefix="audio_to_video_attn",
        query_dim=VIDEO_DIMENSION,
        context_dim=AUDIO_DIMENSION,
        output_dim=VIDEO_DIMENSION,
        gate_input_dim=VIDEO_DIMENSION,
    )
    _add_attention_shapes(
        shapes,
        prefix="video_to_audio_attn",
        query_dim=AUDIO_DIMENSION,
        context_dim=VIDEO_DIMENSION,
        output_dim=AUDIO_DIMENSION,
        gate_input_dim=AUDIO_DIMENSION,
    )
    shapes.update(
        {
            "ff.net.0.proj.weight": (VIDEO_FEED_FORWARD_DIMENSION, VIDEO_DIMENSION),
            "ff.net.2.weight": (VIDEO_DIMENSION, VIDEO_FEED_FORWARD_DIMENSION),
            "audio_ff.net.0.proj.weight": (AUDIO_FEED_FORWARD_DIMENSION, AUDIO_DIMENSION),
            "audio_ff.net.0.proj.bias": (AUDIO_FEED_FORWARD_DIMENSION,),
            "audio_ff.net.2.weight": (AUDIO_DIMENSION, AUDIO_FEED_FORWARD_DIMENSION),
            "audio_ff.net.2.bias": (AUDIO_DIMENSION,),
            "scale_shift_table": (9, VIDEO_DIMENSION),
            "audio_scale_shift_table": (9, AUDIO_DIMENSION),
            "prompt_scale_shift_table": (2, VIDEO_DIMENSION),
            "audio_prompt_scale_shift_table": (2, AUDIO_DIMENSION),
            "scale_shift_table_a2v_ca_video": (5, VIDEO_DIMENSION),
            "scale_shift_table_a2v_ca_audio": (5, AUDIO_DIMENSION),
        }
    )
    return shapes


def transformer_block_target_dtypes() -> dict[str, str]:
    """Return the reviewed source dtype for every production-block target.

    LTX keeps its AdaLN modulation tables as FP32 auxiliary parameters while
    operational projection weights remain BF16. MLX preserves those FP32 tables
    instead of coercing them during loading.
    """

    return {
        target_key: "F32" if target_key in TRANSFORMER_BLOCK_F32_TARGETS else BF16_DTYPE
        for target_key in transformer_block_target_shapes()
    }


def build_transformer_block_mapping(shard_path: Path, block_index: int) -> dict[str, object]:
    """Build and validate the direct BF16 mapping for one production AV block."""

    if block_index < 0:
        raise LaraError("LARA-MODEL-028", details={"key": f"block_index={block_index}"})
    source = inspect_safetensors(shard_path)
    prefix = f"{TRANSFORMER_BLOCK_SOURCE_PREFIX}{block_index}."
    rules: list[dict[str, object]] = []
    for descriptor in source.tensors:
        if not descriptor.name.startswith(prefix):
            continue
        target_key = descriptor.name.removeprefix(prefix)
        rules.append(
            {
                "source_file": shard_path.name,
                "source_key": descriptor.name,
                "target_key": target_key,
                "transform": IDENTITY_TRANSFORM,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
            }
        )
    if not rules:
        raise LaraError("LARA-MODEL-028", details={"key": prefix})
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "transformer_block",
        "block_index": block_index,
        "shards": [{"file": shard_path.name, "tensor_count": len(rules)}],
        "rules": sorted(rules, key=lambda rule: str(rule["target_key"])),
    }
    validate_transformer_block_mapping(mapping)
    return mapping


def validate_transformer_block_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Require the full 84-key production AV block with reviewed BF16 shapes."""

    target_shapes = transformer_block_target_shapes()
    target_dtypes = transformer_block_target_dtypes()
    rules = validate_mapping(mapping, expected_target_keys=target_shapes)
    for rule in rules:
        target_key = rule["target_key"]
        source_key = rule["source_key"]
        source_shape = rule.get("shape")
        if not isinstance(target_key, str) or not isinstance(source_key, str) or not isinstance(source_shape, list):
            raise LaraError("LARA-MODEL-014", details={"source_key": "transformer_block_mapping_rule"})
        if rule["transform"] != IDENTITY_TRANSFORM or rule.get("dtype") != target_dtypes[target_key]:
            raise LaraError("LARA-MODEL-028", details={"key": source_key})
        if tuple(source_shape) != target_shapes[target_key]:
            raise LaraError("LARA-MODEL-028", details={"key": source_key})
    return rules


def write_transformer_block_mapping(output: Path, shard_path: Path, block_index: int) -> None:
    """Atomically write a full production-block mapping manifest."""

    mapping = build_transformer_block_mapping(shard_path, block_index)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def _add_attention_shapes(
    shapes: dict[str, tuple[int, ...]],
    *,
    prefix: str,
    query_dim: int,
    context_dim: int,
    output_dim: int,
    gate_input_dim: int,
) -> None:
    inner_dim = AUDIO_DIMENSION if prefix in {"audio_to_video_attn", "video_to_audio_attn"} else output_dim
    shapes.update(
        {
            f"{prefix}.q_norm.weight": (inner_dim,),
            f"{prefix}.k_norm.weight": (inner_dim,),
            f"{prefix}.to_q.weight": (inner_dim, query_dim),
            f"{prefix}.to_q.bias": (inner_dim,),
            f"{prefix}.to_k.weight": (inner_dim, context_dim),
            f"{prefix}.to_k.bias": (inner_dim,),
            f"{prefix}.to_v.weight": (inner_dim, context_dim),
            f"{prefix}.to_v.bias": (inner_dim,),
            f"{prefix}.to_out.0.weight": (output_dim, inner_dim),
            f"{prefix}.to_out.0.bias": (output_dim,),
            f"{prefix}.to_gate_logits.weight": (VIDEO_HEAD_COUNT, gate_input_dim),
            f"{prefix}.to_gate_logits.bias": (VIDEO_HEAD_COUNT,),
        }
    )
