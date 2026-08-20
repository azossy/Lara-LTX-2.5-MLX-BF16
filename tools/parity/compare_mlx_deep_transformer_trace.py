#!/usr/bin/env python3
"""Locate the first full-transformer divergence against a CUDA v4 trace."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors
from lara_ltx.sampling import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
    attach_block_perturbations,
)
from lara_ltx.transformer import CheckpointTransformerBlocks, TransformerStream

CONFIG_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 4
PERTURBATION_TYPES = {
    "skip_video_self_attention": PerturbationType.SKIP_VIDEO_SELF_ATTN,
    "skip_audio_self_attention": PerturbationType.SKIP_AUDIO_SELF_ATTN,
    "skip_a2v_cross_attention": PerturbationType.SKIP_A2V_CROSS_ATTN,
    "skip_v2a_cross_attention": PerturbationType.SKIP_V2A_CROSS_ATTN,
}
STREAM_FIELDS = (
    "x",
    "context",
    "timesteps",
    "prompt_timestep",
    "cross_scale_shift_timestep",
    "cross_gate_timestep",
    "context_mask",
    "self_attention_mask",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--cuda-report", required=True, type=Path)
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
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_deep_config:{key}"})
    return result


def _pass_prefix(stage: str, block_index: int, pass_index: int, modality: str) -> str:
    return f"{stage}_deep_block_{block_index:02d}_pass_{pass_index:02d}_{modality}"


def _source_pass_index(pass_config: dict[str, Any], suffix: str) -> int:
    default = _required(pass_config, "input_source_pass_index", int)
    overrides = pass_config.get("input_field_sources", {})
    if not isinstance(overrides, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_input_field_sources"})
    source = overrides.get(suffix, default)
    if not isinstance(source, int) or isinstance(source, bool) or source < 0:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_deep_input_source:{suffix}"})
    return source


def _concat(
    reference: dict[str, np.ndarray],
    *,
    stage: str,
    block_index: int,
    passes: list[dict[str, Any]],
    modality: str,
    suffix: str,
) -> mx.array | None:
    keys = [
        f"{_pass_prefix(stage, block_index, _source_pass_index(value, suffix), modality)}_input_{suffix}"
        for value in passes
    ]
    present = [key in reference for key in keys]
    if not any(present):
        return None
    if not all(present):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"incomplete_deep_field:{suffix}"})
    return mx.array(np.concatenate([reference[key] for key in keys], axis=0), dtype=mx.bfloat16)


def _stream(
    reference: dict[str, np.ndarray],
    *,
    stage: str,
    block_index: int,
    passes: list[dict[str, Any]],
    modality: str,
) -> TransformerStream:
    values = {
        field: _concat(
            reference,
            stage=stage,
            block_index=block_index,
            passes=passes,
            modality=modality,
            suffix=field,
        )
        for field in STREAM_FIELDS
    }
    if values["x"] is None or values["context"] is None or values["timesteps"] is None:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "missing_required_deep_stream_field"})
    positional = tuple(
        _concat(
            reference,
            stage=stage,
            block_index=block_index,
            passes=passes,
            modality=modality,
            suffix=f"positional_embeddings_{component}",
        )
        for component in ("cos", "sin")
    )
    cross_positional = tuple(
        _concat(
            reference,
            stage=stage,
            block_index=block_index,
            passes=passes,
            modality=modality,
            suffix=f"cross_positional_embeddings_{component}",
        )
        for component in ("cos", "sin")
    )
    if any(value is None for value in positional):
        positional_embeddings = None
    else:
        positional_embeddings = positional  # type: ignore[assignment]
    if any(value is None for value in cross_positional):
        cross_positional_embeddings = None
    else:
        cross_positional_embeddings = cross_positional  # type: ignore[assignment]
    return TransformerStream(
        x=values["x"],
        context=values["context"],
        timesteps=values["timesteps"],
        prompt_timestep=values["prompt_timestep"],
        cross_scale_shift_timestep=values["cross_scale_shift_timestep"],
        cross_gate_timestep=values["cross_gate_timestep"],
        positional_embeddings=positional_embeddings,
        cross_positional_embeddings=cross_positional_embeddings,
        context_mask=values["context_mask"],
        self_attention_mask=values["self_attention_mask"],
    )


def _perturbations(pass_config: dict[str, Any]) -> PerturbationConfig:
    names = _required(pass_config, "perturbations", list)
    try:
        values = tuple(Perturbation(PERTURBATION_TYPES[str(name)], blocks=None) for name in names)
    except KeyError as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"unknown_perturbation:{error.args[0]}"}) from error
    return PerturbationConfig(values)


def _validate_configuration(
    config: dict[str, Any], cuda_report: dict[str, Any]
) -> tuple[str, int, list[int], list[dict[str, Any]]]:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_deep_config_schema"})
    if cuda_report.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_deep_trace_schema"})
    stage = _required(config, "stage", str)
    block_count = _required(config, "block_count", int)
    selected = [int(value) for value in _required(config, "selected_block_indices", list)]
    passes = _required(config, "passes", list)
    if (
        block_count <= 0
        or not selected
        or selected != sorted(set(selected))
        or selected[-1] >= block_count
        or not passes
        or cuda_report.get("deep_block_indices") != selected
    ):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_trace_layout"})
    indices = [_required(value, "index", int) for value in passes if isinstance(value, dict)]
    if len(indices) != len(passes) or indices != list(range(len(passes))):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_guidance_pass_order"})
    return stage, block_count, selected, passes


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    cuda_report = _read_json(arguments.cuda_report)
    stage, block_count, selected, passes = _validate_configuration(config, cuda_report)
    with np.load(arguments.cuda_reference) as archive:
        reference = {name: archive[name] for name in archive.files}

    first_block = selected[0]
    video = _stream(
        reference,
        stage=stage,
        block_index=first_block,
        passes=passes,
        modality="video",
    )
    audio = _stream(
        reference,
        stage=stage,
        block_index=first_block,
        passes=passes,
        modality="audio",
    )
    perturbations = BatchedPerturbationConfig(
        tuple(_perturbations(value) for value in passes),
        num_blocks=block_count,
        dtype=mx.bfloat16,
    )
    lora_strength = float(_required(config, "lora_strength", (int, float)))
    provider = CheckpointTransformerBlocks(
        checkpoint=arguments.transformer_checkpoint,
        block_count=block_count,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=lora_strength,
    )
    comparisons: list[dict[str, Any]] = []
    for block_index, block in provider:
        video, audio = attach_block_perturbations(video, audio, perturbations, block=block_index)
        assert video is not None and audio is not None
        video_output, audio_output = block(video, audio)
        assert video_output is not None and audio_output is not None
        mx.eval(video_output, audio_output)
        if block_index in selected:
            candidates = {
                "video": np.asarray(video_output.astype(mx.float32)),
                "audio": np.asarray(audio_output.astype(mx.float32)),
            }
            for pass_config in passes:
                pass_index = int(pass_config["index"])
                pass_name = str(pass_config["name"])
                for modality, candidate in candidates.items():
                    output_key = f"{_pass_prefix(stage, block_index, pass_index, modality)}_output"
                    if output_key not in reference:
                        raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_deep_output:{output_key}"})
                    metrics = compare_tensors(
                        output_key,
                        reference[output_key].astype(np.float32),
                        candidate[pass_index : pass_index + 1],
                    )
                    comparisons.append(
                        {
                            "block_index": block_index,
                            "pass_index": pass_index,
                            "pass_name": pass_name,
                            "modality": modality,
                            "metrics": metrics.to_dict(),
                        }
                    )
        video = replace(video, x=video_output)
        audio = replace(audio, x=audio_output)
        del block, video_output, audio_output

    acceptance = _required(config, "acceptance", dict)
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    for value in comparisons:
        metrics = value["metrics"]
        value["passed"] = (
            metrics["nan_count"] == 0
            and metrics["inf_count"] == 0
            and metrics["normalized_rmse"] <= maximum_nrmse
            and metrics["cosine_similarity"] >= minimum_cosine
        )
    first_failure = next((value for value in comparisons if not value["passed"]), None)
    report = {
        "schema_version": 1,
        "component": "full_transformer_deep_trace",
        "cuda_trace_schema_version": TRACE_SCHEMA_VERSION,
        "block_count": block_count,
        "selected_block_indices": selected,
        "guidance_passes": [{"index": value["index"], "name": value["name"]} for value in passes],
        "lora_strength": lora_strength,
        "acceptance": acceptance,
        "comparisons": comparisons,
        "first_failure": first_failure,
        "passed": first_failure is None,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
