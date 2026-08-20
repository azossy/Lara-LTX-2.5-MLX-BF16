#!/usr/bin/env python3
"""Compare exact CUDA inputs at the first block's attention and FF modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors
from lara_ltx.transformer import CheckpointTransformerBlocks

CONFIG_SCHEMA_VERSION = 1
BLOCK_INDEX = 0
MODALITY_MODULES = {
    "video": ("attn1", "attn2", "ff", "prompt_scale_shift_table"),
    "audio": ("audio_attn1", "audio_attn2", "audio_ff", "audio_prompt_scale_shift_table"),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--cuda-boundary", required=True, type=Path)
    parser.add_argument("--cuda-internal", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"unreadable_json:{path}"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_json_object:{path}"})
    return value


def _required(value: dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    result = value.get(key)
    if not isinstance(result, expected) or (isinstance(result, bool) and expected in (int, float)):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_internal_config:{key}"})
    return result


def _source_pass_index(pass_config: dict[str, Any], suffix: str) -> int:
    default = int(_required(pass_config, "input_source_pass_index", int))
    overrides = pass_config.get("input_field_sources", {})
    if not isinstance(overrides, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_internal_input_sources"})
    return int(overrides.get(suffix, default))


def _boundary_prefix(stage: str, pass_index: int, modality: str) -> str:
    return f"{stage}_deep_block_{BLOCK_INDEX:02d}_pass_{pass_index:02d}_{modality}_input"


def _internal_prefix(stage: str, pass_index: int) -> str:
    return f"{stage}_deep_block_{BLOCK_INDEX:02d}_pass_{pass_index:02d}_internal"


def _prompt_context(
    block: object,
    reference: dict[str, np.ndarray],
    *,
    stage: str,
    pass_config: dict[str, Any],
    modality: str,
    table_name: str,
) -> mx.array:
    context_index = _source_pass_index(pass_config, "context")
    timestep_index = _source_pass_index(pass_config, "prompt_timestep")
    context = mx.array(
        reference[f"{_boundary_prefix(stage, context_index, modality)}_context"],
        dtype=mx.bfloat16,
    )
    timestep = mx.array(
        reference[f"{_boundary_prefix(stage, timestep_index, modality)}_prompt_timestep"],
        dtype=mx.bfloat16,
    )
    table = getattr(block, table_name).astype(context.dtype)
    modulation = table[None, None, :, :] + timestep.reshape(
        timestep.shape[0],
        timestep.shape[1],
        table.shape[0],
        table.shape[1],
    )
    shift = modulation[:, :, 0, :]
    scale = modulation[:, :, 1, :]
    return context * (1 + scale) + shift


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_internal_config_schema"})
    stage = str(_required(config, "stage", str))
    passes = _required(config, "passes", list)
    lora_strength = float(_required(config, "lora_strength", (int, float)))
    with np.load(arguments.cuda_boundary) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(arguments.cuda_internal) as archive:
        reference.update({name: archive[name] for name in archive.files})

    provider = CheckpointTransformerBlocks(
        checkpoint=arguments.transformer_checkpoint,
        block_count=1,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=lora_strength,
    )
    block_index, block = next(iter(provider))
    if block_index != BLOCK_INDEX:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unexpected_internal_block_index"})

    comparisons: list[dict[str, Any]] = []
    for pass_config in passes:
        if not isinstance(pass_config, dict):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_internal_pass"})
        pass_index = int(_required(pass_config, "index", int))
        pass_name = str(_required(pass_config, "name", str))
        internal_prefix = _internal_prefix(stage, pass_index)
        for modality, module_names in MODALITY_MODULES.items():
            attn1_name, attn2_name, ff_name, table_name = module_names
            positional_index = _source_pass_index(pass_config, "positional_embeddings_cos")
            positional_prefix = _boundary_prefix(stage, positional_index, modality)
            positional = tuple(
                mx.array(reference[f"{positional_prefix}_positional_embeddings_{component}"], dtype=mx.bfloat16)
                for component in ("cos", "sin")
            )
            module_cases = (
                (
                    attn1_name,
                    getattr(block, attn1_name),
                    {"pe": positional},
                ),
                (
                    attn2_name,
                    getattr(block, attn2_name),
                    {
                        "context": _prompt_context(
                            block,
                            reference,
                            stage=stage,
                            pass_config=pass_config,
                            modality=modality,
                            table_name=table_name,
                        )
                    },
                ),
                (ff_name, getattr(block, ff_name), {}),
            )
            for cuda_name, module, keyword_arguments in module_cases:
                input_key = f"{internal_prefix}_{cuda_name}_input"
                output_key = f"{internal_prefix}_{cuda_name}_output"
                candidate = module(mx.array(reference[input_key], dtype=mx.bfloat16), **keyword_arguments)
                mx.eval(candidate)
                metrics = compare_tensors(
                    output_key,
                    reference[output_key].astype(np.float32),
                    np.asarray(candidate.astype(mx.float32)),
                )
                comparisons.append(
                    {
                        "pass_index": pass_index,
                        "pass_name": pass_name,
                        "modality": modality,
                        "module": cuda_name,
                        "metrics": metrics.to_dict(),
                    }
                )

    acceptance = _required(config, "acceptance", dict)
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    for comparison in comparisons:
        metrics = comparison["metrics"]
        comparison["passed"] = (
            metrics["nan_count"] == 0
            and metrics["inf_count"] == 0
            and metrics["normalized_rmse"] <= maximum_nrmse
            and metrics["cosine_similarity"] >= minimum_cosine
        )
    report = {
        "schema_version": 1,
        "component": "block0_exact_input_internal_modules",
        "lora_strength": lora_strength,
        "acceptance": acceptance,
        "comparisons": comparisons,
        "passed": all(value["passed"] for value in comparisons),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
