"""Reviewed mapping contract for the LTX 2.5 latent spatial upscaler."""

from __future__ import annotations

import json
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

LATENT_CHANNELS = 128
UPSAMPLER_CHANNELS = 1024
RESIDUAL_BLOCK_COUNT = 4
KERNEL_SIZE = 3
UPSAMPLER_EXPANSION = 4
COMPONENT_NAME = "latent_spatial_upscaler_x2"


def spatial_upscaler_target_shapes() -> dict[str, tuple[int, ...]]:
    """Return the exact source-layout MLX parameter contract for x2 upscaling."""

    kernel_3d = (KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE)
    shapes = {
        "initial_conv.weight": (UPSAMPLER_CHANNELS, LATENT_CHANNELS, *kernel_3d),
        "initial_conv.bias": (UPSAMPLER_CHANNELS,),
        "initial_norm.weight": (UPSAMPLER_CHANNELS,),
        "initial_norm.bias": (UPSAMPLER_CHANNELS,),
        "upsampler.0.weight": (
            UPSAMPLER_EXPANSION * UPSAMPLER_CHANNELS,
            UPSAMPLER_CHANNELS,
            KERNEL_SIZE,
            KERNEL_SIZE,
        ),
        "upsampler.0.bias": (UPSAMPLER_EXPANSION * UPSAMPLER_CHANNELS,),
        "final_conv.weight": (LATENT_CHANNELS, UPSAMPLER_CHANNELS, *kernel_3d),
        "final_conv.bias": (LATENT_CHANNELS,),
    }
    for block_collection in ("res_blocks", "post_upsample_res_blocks"):
        for block_index in range(RESIDUAL_BLOCK_COUNT):
            prefix = f"{block_collection}.{block_index}"
            shapes.update(
                {
                    f"{prefix}.conv1.weight": (UPSAMPLER_CHANNELS, UPSAMPLER_CHANNELS, *kernel_3d),
                    f"{prefix}.conv1.bias": (UPSAMPLER_CHANNELS,),
                    f"{prefix}.norm1.weight": (UPSAMPLER_CHANNELS,),
                    f"{prefix}.norm1.bias": (UPSAMPLER_CHANNELS,),
                    f"{prefix}.conv2.weight": (UPSAMPLER_CHANNELS, UPSAMPLER_CHANNELS, *kernel_3d),
                    f"{prefix}.conv2.bias": (UPSAMPLER_CHANNELS,),
                    f"{prefix}.norm2.weight": (UPSAMPLER_CHANNELS,),
                    f"{prefix}.norm2.bias": (UPSAMPLER_CHANNELS,),
                }
            )
    return shapes


def build_spatial_upscaler_mapping(shard_path: Path) -> dict[str, object]:
    """Map the complete x2 upscaler using its source-compatible MLX keys."""

    source = inspect_safetensors(shard_path)
    target_shapes = spatial_upscaler_target_shapes()
    descriptors = {descriptor.name: descriptor for descriptor in source.tensors}
    _require_exact_source_keys(set(descriptors), set(target_shapes))
    rules = [
        {
            "source_file": shard_path.name,
            "source_key": key,
            "target_key": key,
            "transform": IDENTITY_TRANSFORM,
            "dtype": descriptors[key].dtype,
            "shape": list(descriptors[key].shape),
        }
        for key in sorted(target_shapes)
    ]
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": COMPONENT_NAME,
        "shards": [
            {
                "file": shard_path.name,
                "tensor_count": source.tensor_count,
                "tensor_bytes": source.total_tensor_bytes,
            }
        ],
        "rules": rules,
    }
    validate_spatial_upscaler_mapping(mapping)
    return mapping


def validate_spatial_upscaler_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Require every x2 upscaler parameter to retain BF16 layout and shape."""

    target_shapes = spatial_upscaler_target_shapes()
    rules = validate_mapping(mapping, expected_target_keys=target_shapes)
    for rule in rules:
        target_key = rule["target_key"]
        source_key = rule["source_key"]
        source_shape = rule.get("shape")
        if not isinstance(target_key, str) or not isinstance(source_key, str) or not isinstance(source_shape, list):
            raise LaraError("LARA-MODEL-014", details={"source_key": "spatial_upscaler_mapping_rule"})
        if (
            source_key != target_key
            or rule["transform"] != IDENTITY_TRANSFORM
            or rule.get("dtype") != BF16_DTYPE
            or tuple(source_shape) != target_shapes[target_key]
        ):
            raise LaraError("LARA-MODEL-030", details={"key": source_key})
    return rules


def write_spatial_upscaler_mapping(output: Path, shard_path: Path) -> None:
    """Atomically write the reviewed latent spatial-upscaler mapping."""

    mapping = build_spatial_upscaler_mapping(shard_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def _require_exact_source_keys(actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        key = missing[0] if missing else unexpected[0]
        raise LaraError("LARA-MODEL-030", details={"key": key})
