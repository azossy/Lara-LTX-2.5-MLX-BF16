"""Exact checkpoint mapping for the LTX video/audio transformer output heads."""

from __future__ import annotations

import json
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

VIDEO_HIDDEN_DIMENSION = 4096
AUDIO_HIDDEN_DIMENSION = 2048
OUTPUT_CHANNELS = 128
SOURCE_PREFIX = "model.diffusion_model."
OUTPUT_SOURCE_TO_TARGET = {
    "scale_shift_table": "video.scale_shift_table",
    "proj_out.weight": "video.proj_out.weight",
    "proj_out.bias": "video.proj_out.bias",
    "audio_scale_shift_table": "audio.scale_shift_table",
    "audio_proj_out.weight": "audio.proj_out.weight",
    "audio_proj_out.bias": "audio.proj_out.bias",
}


def transformer_output_target_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "video.scale_shift_table": (2, VIDEO_HIDDEN_DIMENSION),
        "video.proj_out.weight": (OUTPUT_CHANNELS, VIDEO_HIDDEN_DIMENSION),
        "video.proj_out.bias": (OUTPUT_CHANNELS,),
        "audio.scale_shift_table": (2, AUDIO_HIDDEN_DIMENSION),
        "audio.proj_out.weight": (OUTPUT_CHANNELS, AUDIO_HIDDEN_DIMENSION),
        "audio.proj_out.bias": (OUTPUT_CHANNELS,),
    }


def transformer_output_target_dtypes() -> dict[str, str]:
    return {
        target: "F32" if target.endswith("scale_shift_table") else BF16_DTYPE
        for target in transformer_output_target_shapes()
    }


def build_transformer_output_mapping(checkpoint: Path) -> dict[str, object]:
    source = inspect_safetensors(checkpoint)
    descriptors = {descriptor.name: descriptor for descriptor in source.tensors}
    rules: list[dict[str, object]] = []
    for suffix, target in OUTPUT_SOURCE_TO_TARGET.items():
        source_key = f"{SOURCE_PREFIX}{suffix}"
        descriptor = descriptors.get(source_key)
        if descriptor is None:
            raise LaraError("LARA-MODEL-033", details={"key": source_key})
        rules.append(
            {
                "source_file": checkpoint.name,
                "source_key": source_key,
                "target_key": target,
                "transform": IDENTITY_TRANSFORM,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
            }
        )
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "transformer_output",
        "shards": [{"file": checkpoint.name, "tensor_count": len(rules)}],
        "rules": sorted(rules, key=lambda rule: str(rule["target_key"])),
    }
    validate_transformer_output_mapping(mapping)
    return mapping


def validate_transformer_output_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    shapes = transformer_output_target_shapes()
    dtypes = transformer_output_target_dtypes()
    rules = validate_mapping(mapping, expected_target_keys=shapes)
    for rule in rules:
        target = rule["target_key"]
        if not isinstance(target, str):
            raise LaraError("LARA-MODEL-014", details={"source_key": "transformer_output_mapping_rule"})
        if (
            rule.get("transform") != IDENTITY_TRANSFORM
            or rule.get("dtype") != dtypes[target]
            or tuple(rule.get("shape", ())) != shapes[target]
        ):
            raise LaraError("LARA-MODEL-033", details={"key": str(rule.get("source_key"))})
    return rules


def write_transformer_output_mapping(output: Path, checkpoint: Path) -> None:
    mapping = build_transformer_output_mapping(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc
