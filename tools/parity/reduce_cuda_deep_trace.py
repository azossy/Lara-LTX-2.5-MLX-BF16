#!/usr/bin/env python3
"""Reduce a CUDA deep trace to the tensors required for MLX replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_SCHEMA_VERSION = 1
INPUT_SUFFIXES = (
    "x",
    "context",
    "timesteps",
    "embedded_timestep",
    "prompt_timestep",
    "cross_scale_shift_timestep",
    "cross_gate_timestep",
    "context_mask",
    "self_attention_mask",
    "positional_embeddings_cos",
    "positional_embeddings_sin",
    "cross_positional_embeddings_cos",
    "cross_positional_embeddings_sin",
)
MODALITIES = ("video", "audio")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _required(value: dict[str, Any], key: str, expected: type) -> Any:
    result = value.get(key)
    if not isinstance(result, expected) or (expected is int and isinstance(result, bool)):
        raise ValueError(f"[LARA-RUNTIME-008] Invalid deep-trace configuration: {key}")
    return result


def _source_pass_index(pass_config: dict[str, Any], suffix: str) -> int:
    default = _required(pass_config, "input_source_pass_index", int)
    overrides = pass_config.get("input_field_sources", {})
    if not isinstance(overrides, dict):
        raise ValueError("[LARA-RUNTIME-008] Invalid deep-trace input source map.")
    source = overrides.get(suffix, default)
    if not isinstance(source, int) or isinstance(source, bool) or source < 0:
        raise ValueError(f"[LARA-RUNTIME-008] Invalid source pass for {suffix}.")
    return source


def selected_names(archive_names: list[str], config: dict[str, Any]) -> set[str]:
    """Return the minimal self-contained set required by all deep parity tools."""
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("[LARA-RUNTIME-008] Unsupported deep-trace configuration schema.")
    stage = _required(config, "stage", str)
    selected_blocks = _required(config, "selected_block_indices", list)
    passes = _required(config, "passes", list)
    first_call = _required(config, "first_call", dict)
    if not selected_blocks or not passes:
        raise ValueError("[LARA-RUNTIME-008] Empty deep-trace replay layout.")

    available = set(archive_names)
    selected = {
        _required(first_call, "sigma_key", str),
        _required(first_call, "video_input_key", str),
        _required(first_call, "audio_input_key", str),
    }
    denoiser_prefix = f"{stage}_denoiser_call_00_"
    selected.update(name for name in available if name.startswith(denoiser_prefix))

    for pass_config in passes:
        if not isinstance(pass_config, dict):
            raise ValueError("[LARA-RUNTIME-008] Invalid deep-trace pass entry.")
        pass_index = _required(pass_config, "index", int)
        for block_index in selected_blocks:
            if not isinstance(block_index, int) or isinstance(block_index, bool) or block_index < 0:
                raise ValueError("[LARA-RUNTIME-008] Invalid deep-trace block index.")
            for modality in MODALITIES:
                selected.add(f"{stage}_deep_block_{block_index:02d}_pass_{pass_index:02d}_{modality}_output")
        for suffix in INPUT_SUFFIXES:
            source_index = _source_pass_index(pass_config, suffix)
            for modality in MODALITIES:
                name = f"{stage}_deep_block_00_pass_{source_index:02d}_{modality}_input_{suffix}"
                if name in available:
                    selected.add(name)

    missing = selected.difference(available)
    if missing:
        raise ValueError(f"[LARA-RUNTIME-008] Missing deep replay tensor: {sorted(missing)[0]}")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("[LARA-RUNTIME-008] Deep-trace configuration must be an object.")
    with np.load(arguments.input) as archive:
        names = selected_names(list(archive.files), config)
        arrays = {name: archive[name] for name in sorted(names)}

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    with temporary_output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_output.replace(arguments.output)

    report = {
        "schema_version": 1,
        "component": "cuda_deep_trace_reducer",
        "source": {
            "file": arguments.input.name,
            "sha256": _sha256(arguments.input),
        },
        "output": {
            "file": arguments.output.name,
            "sha256": _sha256(arguments.output),
            "size_bytes": arguments.output.stat().st_size,
            "tensor_count": len(arrays),
        },
        "config": {
            "file": arguments.config.name,
            "sha256": _sha256(arguments.config),
        },
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_report.replace(arguments.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
