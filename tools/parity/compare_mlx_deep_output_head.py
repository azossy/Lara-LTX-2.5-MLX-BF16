#!/usr/bin/env python3
"""Compare LoRA-fused MLX output heads at exact CUDA block-47 boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors
from lara_ltx.transformer import load_production_transformer_output

CONFIG_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--output-mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
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
    if not isinstance(result, expected):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_deep_output_config:{key}"})
    return result


def _source_pass_index(pass_config: dict[str, Any], field: str) -> int:
    default = int(_required(pass_config, "input_source_pass_index", int))
    overrides = pass_config.get("input_field_sources", {})
    if not isinstance(overrides, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_output_field_sources"})
    return int(overrides.get(field, default))


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_deep_output_config_schema"})
    mapping = _read_json(arguments.output_mapping)
    passes = _required(config, "passes", list)
    selected_blocks = _required(config, "selected_block_indices", list)
    if not selected_blocks:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "missing_deep_output_block"})
    final_block = int(selected_blocks[-1])
    stage = str(_required(config, "stage", str))
    lora_strength = float(_required(config, "lora_strength", (int, float)))
    first_call = _required(config, "first_call", dict)
    acceptance = _required(config, "acceptance", dict)
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    with np.load(arguments.cuda_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    sigma = float(np.asarray(reference[str(_required(first_call, "sigma_key", str))]))
    if not np.isfinite(sigma) or sigma <= 0:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_output_sigma"})

    output_heads, fused_pair_count = load_production_transformer_output(
        checkpoint=arguments.transformer_checkpoint,
        mapping=mapping,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=lora_strength,
    )
    comparisons: list[dict[str, Any]] = []
    for pass_config in passes:
        if not isinstance(pass_config, dict):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_output_pass"})
        pass_index = int(_required(pass_config, "index", int))
        pass_name = str(_required(pass_config, "name", str))
        cuda_component = str(_required(pass_config, "cuda_component", str))
        for modality in ("video", "audio"):
            prefix = f"{stage}_deep_block_{final_block:02d}_pass_{pass_index:02d}_{modality}"
            hidden_key = f"{prefix}_output"
            timestep_source = _source_pass_index(pass_config, "embedded_timestep")
            timestep_key = f"{stage}_deep_block_00_pass_{timestep_source:02d}_{modality}_input_embedded_timestep"
            input_key = str(_required(first_call, f"{modality}_input_key", str))
            expected_key = f"{stage}_denoiser_call_00_{modality}_{cuda_component}"
            head = output_heads.video if modality == "video" else output_heads.audio
            velocity = head(
                mx.array(reference[hidden_key], dtype=mx.bfloat16),
                mx.array(reference[timestep_key], dtype=mx.bfloat16),
            )
            denoised = (
                mx.array(reference[input_key], dtype=mx.bfloat16).astype(mx.float32)
                - velocity.astype(mx.float32) * sigma
            ).astype(mx.bfloat16)
            mx.eval(denoised)
            metrics = compare_tensors(
                expected_key,
                reference[expected_key].astype(np.float32),
                np.asarray(denoised.astype(mx.float32)),
            )
            passed = (
                metrics.nan_count == 0
                and metrics.inf_count == 0
                and metrics.normalized_rmse <= maximum_nrmse
                and metrics.cosine_similarity >= minimum_cosine
            )
            comparisons.append(
                {
                    "pass_index": pass_index,
                    "pass_name": pass_name,
                    "modality": modality,
                    "metrics": metrics.to_dict(),
                    "passed": passed,
                }
            )
    report = {
        "schema_version": 1,
        "component": "lora_fused_transformer_output_head",
        "final_block_index": final_block,
        "sigma": sigma,
        "lora_strength": lora_strength,
        "fused_lora_pair_count": fused_pair_count,
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
