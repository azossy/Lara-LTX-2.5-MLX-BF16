"""Versioned, header-derived checkpoint-key mapping templates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from lara_ltx.duration_head_contract import duration_head_target_keys, duration_head_target_shapes
from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors

MAPPING_SCHEMA_VERSION = 2
IDENTITY_TRANSFORM = "identity"
TRANSPOSE_2D_TRANSFORM = "transpose_2d"
SPLIT_QKV_Q_TRANSFORM = "split_qkv_q"
SPLIT_QKV_K_TRANSFORM = "split_qkv_k"
SPLIT_QKV_V_TRANSFORM = "split_qkv_v"
FOLD_GATE_TRANSFORM = "fold_gate"
QKV_SPLIT_TRANSFORMS = frozenset(
    {
        SPLIT_QKV_Q_TRANSFORM,
        SPLIT_QKV_K_TRANSFORM,
        SPLIT_QKV_V_TRANSFORM,
    }
)
QKV_PART_COUNT = len(QKV_SPLIT_TRANSFORMS)
DURATION_HEAD_SOURCE_PREFIX = "duration_head."
DURATION_HEAD_QKV_SOURCE_SUFFIXES = frozenset(
    {
        "attention_pooler.cross_attn.in_proj_weight",
        "attention_pooler.cross_attn.in_proj_bias",
    }
)
SUPPORTED_TRANSFORMS = frozenset(
    {IDENTITY_TRANSFORM, TRANSPOSE_2D_TRANSFORM, FOLD_GATE_TRANSFORM, *QKV_SPLIT_TRANSFORMS}
)
MAPPABLE_SOURCE_DTYPES = frozenset({BF16_DTYPE, "F32"})


def build_mapping_template(shard_paths: Iterable[Path]) -> dict[str, object]:
    """Create a deterministic mapping-review template from validated BF16 headers.

    Target keys remain null until an implementation-specific MLX parameter map is
    reviewed. This avoids silently treating source checkpoint names as MLX names.
    """

    seen: dict[str, Path] = {}
    shards: list[dict[str, object]] = []
    rules: list[dict[str, object]] = []
    for path in sorted((Path(item) for item in shard_paths), key=lambda item: item.name):
        source = inspect_safetensors(path)
        source.require_dtype(BF16_DTYPE)
        shards.append(
            {
                "file": path.name,
                "tensor_count": source.tensor_count,
                "tensor_bytes": source.total_tensor_bytes,
            }
        )
        for descriptor in source.tensors:
            previous = seen.get(descriptor.name)
            if previous is not None:
                raise LaraError(
                    "LARA-MODEL-012",
                    details={"name": descriptor.name, "first": str(previous), "second": str(path)},
                )
            seen[descriptor.name] = path
            rules.append(
                {
                    "source_file": path.name,
                    "source_key": descriptor.name,
                    "target_key": None,
                    "transform": IDENTITY_TRANSFORM,
                    "dtype": descriptor.dtype,
                    "shape": list(descriptor.shape),
                }
            )
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "shards": shards,
        "rules": rules,
    }


def write_mapping_template(output: Path, shard_paths: Iterable[Path]) -> None:
    """Atomically write a reviewable mapping template without creating parents."""

    template = build_mapping_template(shard_paths)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def build_diffusion_vae_decoder_mapping(shard_path: Path) -> dict[str, object]:
    """Build the pinned Diffusion VAE decoder mapping from its verified header.

    The official standalone VAE packs encoder and decoder tensors together. This
    manifest deliberately selects only the decoder plus latent statistics, splits
    fused QKV tensors, renames the timestep MLP, and records the single upstream
    ``type_emb`` load artifact as an audited omission.
    """

    source = inspect_safetensors(shard_path)
    source.require_dtype(BF16_DTYPE)
    rules: list[dict[str, object]] = []
    ignored_sources: list[dict[str, str]] = []
    for descriptor in source.tensors:
        source_key = descriptor.name
        if source_key == "decoder.type_emb":
            ignored_sources.append({"source_key": source_key, "reason": "upstream_load_artifact"})
            continue
        if source_key.startswith("decoder."):
            target_key = source_key.removeprefix("decoder.")
            target_key = target_key.replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
            target_key = target_key.replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
            if source_key.endswith((".qkv.weight", ".qkv.bias")):
                leaf = "weight" if source_key.endswith(".weight") else "bias"
                target_prefix = target_key[: -len(leaf)]
                for part in ("q", "k", "v"):
                    rules.append(
                        {
                            "source_file": shard_path.name,
                            "source_key": source_key,
                            "target_key": f"{target_prefix}to_{part}.{leaf}",
                            "transform": f"split_qkv_{part}",
                            "dtype": descriptor.dtype,
                            "shape": list(descriptor.shape),
                        }
                    )
                continue
        elif source_key.startswith("per_channel_statistics."):
            target_key = source_key.replace("std-of-means", "std_of_means").replace("mean-of-means", "mean_of_means")
        else:
            continue
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
    if ignored_sources != [{"source_key": "decoder.type_emb", "reason": "upstream_load_artifact"}]:
        raise LaraError("LARA-MODEL-014", details={"source_key": "decoder.type_emb"})
    if not rules:
        raise LaraError("LARA-MODEL-014", details={"source_key": "diffusion_vae_decoder"})
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "diffusion_video_decoder",
        "shards": [
            {
                "file": shard_path.name,
                "tensor_count": source.tensor_count,
                "tensor_bytes": source.total_tensor_bytes,
            }
        ],
        "ignored_sources": ignored_sources,
        "rules": rules,
    }


def write_diffusion_vae_decoder_mapping(output: Path, shard_path: Path) -> None:
    """Atomically write the checked Diffusion VAE decoder transform manifest."""

    mapping = build_diffusion_vae_decoder_mapping(shard_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def build_duration_head_mapping(shard_path: Path) -> dict[str, object]:
    """Build the reviewed DurationHead mapping from the standalone BF16 shard.

    The official PyTorch ``MultiheadAttention`` stores Q/K/V in fused
    ``in_proj`` tensors. Lara's MLX duration-head target keeps them separate,
    so the source tensors are split on axis zero. All remaining parameter names
    retain their suffix after the official ``duration_head.`` prefix is removed.
    """

    source = inspect_safetensors(shard_path)
    source.require_dtype(BF16_DTYPE)
    rules: list[dict[str, object]] = []
    unexpected_sources: list[str] = []
    for descriptor in source.tensors:
        source_key = descriptor.name
        if not source_key.startswith(DURATION_HEAD_SOURCE_PREFIX):
            unexpected_sources.append(source_key)
            continue
        target_suffix = source_key.removeprefix(DURATION_HEAD_SOURCE_PREFIX)
        if target_suffix in DURATION_HEAD_QKV_SOURCE_SUFFIXES:
            leaf = "weight" if target_suffix.endswith("weight") else "bias"
            target_prefix = target_suffix.removesuffix(f"in_proj_{leaf}")
            for part in ("q", "k", "v"):
                rules.append(
                    {
                        "source_file": shard_path.name,
                        "source_key": source_key,
                        "target_key": f"{target_prefix}to_{part}.{leaf}",
                        "transform": f"split_qkv_{part}",
                        "dtype": descriptor.dtype,
                        "shape": list(descriptor.shape),
                    }
                )
            continue
        rules.append(
            {
                "source_file": shard_path.name,
                "source_key": source_key,
                "target_key": target_suffix,
                "transform": IDENTITY_TRANSFORM,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
            }
        )
    if unexpected_sources or not rules:
        detail = unexpected_sources[0] if unexpected_sources else "duration_head"
        raise LaraError("LARA-MODEL-014", details={"source_key": detail})
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": "duration_head",
        "shards": [
            {
                "file": shard_path.name,
                "tensor_count": source.tensor_count,
                "tensor_bytes": source.total_tensor_bytes,
            }
        ],
        "rules": rules,
    }


def write_duration_head_mapping(output: Path, shard_path: Path) -> None:
    """Atomically write the checked standalone DurationHead mapping manifest."""

    mapping = build_duration_head_mapping(shard_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def validate_duration_head_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Prove an official manifest has exact DurationHead keys and layouts."""

    rules = validate_mapping(mapping, expected_target_keys=duration_head_target_keys())
    target_shapes = duration_head_target_shapes()
    for rule in rules:
        target_key = rule["target_key"]
        source_key = rule["source_key"]
        source_shape = rule.get("shape")
        transform = rule["transform"]
        if not isinstance(target_key, str) or not isinstance(source_key, str) or not isinstance(source_shape, list):
            raise LaraError("LARA-MODEL-014", details={"source_key": "duration_head_mapping_rule"})
        expected_target_shape = target_shapes[target_key]
        expected_source_shape = expected_target_shape
        if transform in QKV_SPLIT_TRANSFORMS:
            expected_source_shape = (expected_target_shape[0] * QKV_PART_COUNT, *expected_target_shape[1:])
        if tuple(source_shape) != expected_source_shape:
            raise LaraError(
                "LARA-MODEL-025",
                details={
                    "source_key": source_key,
                    "expected_shape": expected_source_shape,
                    "actual_shape": tuple(source_shape),
                },
            )
    return rules


def validate_mapping(
    mapping: dict[str, object],
    *,
    expected_target_keys: Iterable[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Validate an explicitly reviewed source-to-MLX mapping before loading.

    Templates intentionally start with unassigned targets. This gate requires
    every target to be reviewed, prevents duplicate target assignment and, when
    model parameters are available, proves their key set is exact.
    """

    rules_value = mapping.get("rules")
    if mapping.get("schema_version") != MAPPING_SCHEMA_VERSION or not isinstance(rules_value, list):
        raise LaraError("LARA-MODEL-014", details={"source_key": "mapping_schema"})
    targets: set[str] = set()
    source_rules: dict[tuple[str, str], list[dict[str, object]]] = {}
    validated: list[dict[str, object]] = []
    for value in rules_value:
        if not isinstance(value, dict):
            raise LaraError("LARA-MODEL-014", details={"source_key": "mapping_rule"})
        rule = cast(dict[str, object], value)
        source_file = rule.get("source_file")
        source_key = rule.get("source_key")
        target_key = rule.get("target_key")
        transform = rule.get("transform")
        if (
            not isinstance(source_file, str)
            or not isinstance(source_key, str)
            or not isinstance(target_key, str)
            or not target_key
        ):
            detail = source_key if isinstance(source_key, str) else "mapping_rule"
            raise LaraError("LARA-MODEL-014", details={"source_key": detail})
        if transform not in SUPPORTED_TRANSFORMS:
            raise LaraError("LARA-MODEL-016", details={"transform": transform})
        dtype = rule.get("dtype")
        if dtype is not None and dtype not in MAPPABLE_SOURCE_DTYPES:
            raise LaraError("LARA-MODEL-004", details={"name": source_key, "dtype": dtype})
        gate_source_key = rule.get("gate_source_key")
        if transform == FOLD_GATE_TRANSFORM:
            if not isinstance(gate_source_key, str) or not gate_source_key:
                raise LaraError("LARA-MODEL-014", details={"source_key": source_key})
        elif gate_source_key is not None:
            raise LaraError("LARA-MODEL-014", details={"source_key": source_key})
        if target_key in targets:
            raise LaraError("LARA-MODEL-015", details={"target_key": target_key})
        source = (source_file, source_key)
        targets.add(target_key)
        source_rules.setdefault(source, []).append(rule)
        validated.append(rule)
    for (source_file, source_key), rules_for_source in source_rules.items():
        if len(rules_for_source) <= 1:
            continue
        transforms = [rule["transform"] for rule in rules_for_source]
        if len(rules_for_source) != len(QKV_SPLIT_TRANSFORMS) or set(transforms) != QKV_SPLIT_TRANSFORMS:
            raise LaraError("LARA-MODEL-018", details={"source_file": source_file, "source_key": source_key})
    if expected_target_keys is not None:
        expected = set(expected_target_keys)
        missing = sorted(expected - targets)
        unexpected = sorted(targets - expected)
        if missing or unexpected:
            raise LaraError(
                "LARA-MODEL-017",
                details={"missing": ",".join(missing) or "none", "unexpected": ",".join(unexpected) or "none"},
            )
    return tuple(validated)


def load_and_validate_mapping(
    path: Path,
    *,
    expected_target_keys: Iterable[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Read a mapping file and apply the same explicit-review validation."""

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaraError("LARA-MODEL-014", details={"source_key": str(path)}) from exc
    if not isinstance(decoded, dict):
        raise LaraError("LARA-MODEL-014", details={"source_key": str(path)})
    return validate_mapping(decoded, expected_target_keys=expected_target_keys)
