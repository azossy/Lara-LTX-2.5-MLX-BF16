"""Exact checkpoint mapping for transformer input and AV conditioning modules."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

VIDEO_HIDDEN_DIMENSION = 4_096
AUDIO_HIDDEN_DIMENSION = 2_048
INPUT_CHANNELS = 128
TIMESTEP_PROJECTION_CHANNELS = 256
SOURCE_PREFIX = "model.diffusion_model."
CHECKPOINT_CONFIG_METADATA_KEY = "config"
TRANSFORMER_CONFIG_KEY = "transformer"
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


@dataclass(frozen=True)
class TransformerInputArchitecture:
    timestep_scale_multiplier: float
    av_cross_timestep_scale_multiplier: float


def _checkpoint_architecture(metadata: dict[str, str]) -> TransformerInputArchitecture:
    raw = metadata.get(CHECKPOINT_CONFIG_METADATA_KEY)
    try:
        payload = json.loads(raw) if raw is not None else None
    except json.JSONDecodeError as error:
        raise LaraError("LARA-MODEL-034", details={"key": CHECKPOINT_CONFIG_METADATA_KEY}) from error
    transformer = payload.get(TRANSFORMER_CONFIG_KEY) if isinstance(payload, dict) else None
    if not isinstance(transformer, dict):
        raise LaraError("LARA-MODEL-034", details={"key": TRANSFORMER_CONFIG_KEY})
    return _validate_architecture_values(transformer)


def _validate_architecture_values(value: dict[str, object]) -> TransformerInputArchitecture:
    timestep = value.get("timestep_scale_multiplier")
    av_cross = value.get("av_ca_timestep_scale_multiplier")
    if (
        not isinstance(timestep, (int, float))
        or isinstance(timestep, bool)
        or not isinstance(av_cross, (int, float))
        or isinstance(av_cross, bool)
        or not math.isfinite(float(timestep))
        or not math.isfinite(float(av_cross))
        or float(timestep) <= 0
        or float(av_cross) <= 0
    ):
        raise LaraError("LARA-MODEL-034", details={"key": "transformer_input_architecture"})
    return TransformerInputArchitecture(
        timestep_scale_multiplier=float(timestep),
        av_cross_timestep_scale_multiplier=float(av_cross),
    )


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
    architecture = _checkpoint_architecture(source.metadata)
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
        "architecture": {
            "timestep_scale_multiplier": architecture.timestep_scale_multiplier,
            "av_ca_timestep_scale_multiplier": architecture.av_cross_timestep_scale_multiplier,
        },
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


def validate_transformer_input_architecture(mapping: dict[str, object]) -> TransformerInputArchitecture:
    value = mapping.get("architecture")
    if not isinstance(value, dict):
        raise LaraError("LARA-MODEL-034", details={"key": "transformer_input_architecture"})
    return _validate_architecture_values(value)


def write_transformer_input_mapping(output: Path, checkpoint: Path) -> None:
    mapping = build_transformer_input_mapping(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc
