"""Exact checkpoint mapping for transformer input and AV conditioning modules."""

from __future__ import annotations

import json
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

VIDEO_HIDDEN_DIMENSION = 4_096
AUDIO_HIDDEN_DIMENSION = 2_048
INPUT_CHANNELS = 128
TIMESTEP_PROJECTION_CHANNELS = 256
SOURCE_PREFIX = "model.diffusion_model."
KEYFRAME_SOURCE = "keyframes_abs_pos_embedding"
KEYFRAME_TARGET = "video.keyframes_abs_pos_embedding"
SOURCE_MODULE_TO_TARGET = {
    "patchify_proj": "video.patchify_proj",
    "audio_patchify_proj": "audio.patchify_proj",
    "adaln_single": "video.adaln_single",
    "audio_adaln_single": "audio.adaln_single",
    "prompt_adaln_single": "video.prompt_adaln_single",
    "audio_prompt_adaln_single": "audio.prompt_adaln_single",
    "av_ca_video_scale_shift_adaln_single": "video_cross_scale_shift",
    "av_ca_audio_scale_shift_adaln_single": "audio_cross_scale_shift",
    "av_ca_a2v_gate_adaln_single": "video_cross_gate",
    "av_ca_v2a_gate_adaln_single": "audio_cross_gate",
}


def _add_adaln_shapes(
    shapes: dict[str, tuple[int, ...]],
    *,
    prefix: str,
    hidden_dimension: int,
    coefficient: int,
) -> None:
    shapes.update(
        {
            f"{prefix}.emb.timestep_embedder.linear_1.weight": (
                hidden_dimension,
                TIMESTEP_PROJECTION_CHANNELS,
            ),
            f"{prefix}.emb.timestep_embedder.linear_1.bias": (hidden_dimension,),
            f"{prefix}.emb.timestep_embedder.linear_2.weight": (hidden_dimension, hidden_dimension),
            f"{prefix}.emb.timestep_embedder.linear_2.bias": (hidden_dimension,),
            f"{prefix}.linear.weight": (coefficient * hidden_dimension, hidden_dimension),
            f"{prefix}.linear.bias": (coefficient * hidden_dimension,),
        }
    )


def transformer_input_target_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "video.patchify_proj.weight": (VIDEO_HIDDEN_DIMENSION, INPUT_CHANNELS),
        "video.patchify_proj.bias": (VIDEO_HIDDEN_DIMENSION,),
        "video.keyframes_abs_pos_embedding": (1, VIDEO_HIDDEN_DIMENSION),
        "audio.patchify_proj.weight": (AUDIO_HIDDEN_DIMENSION, INPUT_CHANNELS),
        "audio.patchify_proj.bias": (AUDIO_HIDDEN_DIMENSION,),
    }
    for prefix, hidden_dimension, coefficient in (
        ("video.adaln_single", VIDEO_HIDDEN_DIMENSION, 9),
        ("audio.adaln_single", AUDIO_HIDDEN_DIMENSION, 9),
        ("video.prompt_adaln_single", VIDEO_HIDDEN_DIMENSION, 2),
        ("audio.prompt_adaln_single", AUDIO_HIDDEN_DIMENSION, 2),
        ("video_cross_scale_shift", VIDEO_HIDDEN_DIMENSION, 4),
        ("audio_cross_scale_shift", AUDIO_HIDDEN_DIMENSION, 4),
        ("video_cross_gate", VIDEO_HIDDEN_DIMENSION, 1),
        ("audio_cross_gate", AUDIO_HIDDEN_DIMENSION, 1),
    ):
        _add_adaln_shapes(
            shapes,
            prefix=prefix,
            hidden_dimension=hidden_dimension,
            coefficient=coefficient,
        )
    return shapes


def build_transformer_input_mapping(checkpoint: Path) -> dict[str, object]:
    source = inspect_safetensors(checkpoint)
    rules: list[dict[str, object]] = []
    for descriptor in source.tensors:
        target_key: str | None = None
        if descriptor.name == f"{SOURCE_PREFIX}{KEYFRAME_SOURCE}":
            target_key = KEYFRAME_TARGET
        else:
            for source_module, target_module in SOURCE_MODULE_TO_TARGET.items():
                module_prefix = f"{SOURCE_PREFIX}{source_module}."
                if descriptor.name.startswith(module_prefix):
                    target_key = f"{target_module}.{descriptor.name.removeprefix(module_prefix)}"
                    break
        if target_key is None:
            continue
        rules.append(
            {
                "source_file": checkpoint.name,
                "source_key": descriptor.name,
                "target_key": target_key,
                "transform": IDENTITY_TRANSFORM,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
            }
        )
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "transformer_input",
        "shards": [{"file": checkpoint.name, "tensor_count": len(rules)}],
        "rules": sorted(rules, key=lambda rule: str(rule["target_key"])),
    }
    validate_transformer_input_mapping(mapping)
    return mapping


def validate_transformer_input_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    shapes = transformer_input_target_shapes()
    rules = validate_mapping(mapping, expected_target_keys=shapes)
    for rule in rules:
        target = rule.get("target_key")
        if not isinstance(target, str):
            raise LaraError("LARA-MODEL-014", details={"source_key": "transformer_input_mapping_rule"})
        if (
            rule.get("transform") != IDENTITY_TRANSFORM
            or rule.get("dtype") != BF16_DTYPE
            or tuple(rule.get("shape", ())) != shapes[target]
        ):
            raise LaraError("LARA-MODEL-034", details={"key": str(rule.get("source_key"))})
    return rules


def write_transformer_input_mapping(output: Path, checkpoint: Path) -> None:
    mapping = build_transformer_input_mapping(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc
