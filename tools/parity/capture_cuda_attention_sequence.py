#!/usr/bin/env python3
"""Replay a compact CUDA block sequence and capture selected attention internals."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from capture_cuda_attention_subset import (
    ATTENTION_MODULES,
    CONFIG_SCHEMA_VERSION,
    DEEP_CONFIG_SCHEMA_VERSION,
    _fail,
    _load_block,
    _metrics,
    _perturbations,
    _read_json,
    _required,
    _sha256,
    _stream,
)
from cuda_attention_internals import (
    AttentionInternalsRecorder,
    TensorCapture,
    deduplicate_capture,
    write_artifact_shards,
)
from ltx_core.guidance.perturbations import PerturbationType
from ltx_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs

ARTIFACT_SCHEMA_VERSION = 6
MODALITIES = ("video", "audio")
STREAM_CAPTURE_FIELDS = (
    "x",
    "context",
    "timesteps",
    "embedded_timestep",
    "prompt_timestep",
    "cross_scale_shift_timestep",
    "cross_gate_timestep",
    "context_mask",
    "self_attention_mask",
    "self_attn_perturbation_mask",
    "cross_attn_perturbation_mask",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-subset", required=True, type=Path)
    parser.add_argument("--lora-subset", required=True, type=Path)
    parser.add_argument("--input-reference", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--deep-config", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _integer_list(value: dict[str, Any], key: str) -> list[int]:
    items = _required(value, key, list)
    if not items or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in items):
        raise ValueError(f"invalid_config:{key}")
    result = [int(item) for item in items]
    if result != sorted(set(result)):
        raise ValueError(f"invalid_config_order:{key}")
    return result


def _capture_stream(capture: TensorCapture, prefix: str, stream: TransformerArgs) -> None:
    for field in STREAM_CAPTURE_FIELDS:
        capture.add(f"{prefix}_{field}", getattr(stream, field, None))
    for field in ("positional_embeddings", "cross_positional_embeddings"):
        frequencies = getattr(stream, field, None)
        if frequencies is not None:
            capture.add(f"{prefix}_{field}_cos", frequencies[0])
            capture.add(f"{prefix}_{field}_sin", frequencies[1])


def _comparison_passed(metrics: dict[str, float | int], acceptance: dict[str, Any]) -> bool:
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    return bool(
        metrics["nan_count"] == 0
        and metrics["inf_count"] == 0
        and metrics["normalized_rmse"] <= maximum_nrmse
        and metrics["cosine_similarity"] >= minimum_cosine
    )


@torch.inference_mode()
def main() -> int:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    config = _read_json(arguments.config)
    deep_config = _read_json(arguments.deep_config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported_attention_sequence_config_schema")
    if config.get("cuda_artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported_attention_artifact_schema")
    if deep_config.get("schema_version") != DEEP_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported_deep_config_schema")

    stage = str(_required(config, "stage", str))
    if stage != _required(deep_config, "stage", str):
        raise ValueError("attention_and_deep_stage_mismatch")
    start_block = int(_required(config, "sequence_start_block_index", int))
    end_block = int(_required(config, "sequence_end_block_index", int))
    block_count = int(_required(deep_config, "block_count", int))
    capture_blocks = _integer_list(config, "capture_block_indices")
    reference_blocks = _integer_list(config, "reference_output_block_indices")
    if start_block != 0 or end_block < start_block or end_block >= block_count:
        raise ValueError("invalid_attention_sequence_range")
    if capture_blocks[-1] > end_block or reference_blocks[-1] > end_block:
        raise ValueError("attention_sequence_block_outside_range")

    pass_indices = _integer_list(config, "passes")
    deep_passes = _required(deep_config, "passes", list)
    if [value.get("index") for value in deep_passes if isinstance(value, dict)] != pass_indices:
        raise ValueError("attention_and_deep_pass_mismatch")
    module_names = tuple(_required(config, "modules", list))
    if not module_names or any(name not in ATTENTION_MODULES for name in module_names):
        raise ValueError("invalid_attention_modules")
    lora_strength = float(_required(config, "lora_strength", (int, float)))
    maximum_shard_bytes = int(_required(config, "maximum_artifact_shard_bytes", int))
    acceptance = _required(config, "acceptance", dict)

    with np.load(arguments.input_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    device = torch.device("cuda")
    states: list[tuple[TransformerArgs, TransformerArgs, Any, int]] = []
    for pass_config in deep_passes:
        if not isinstance(pass_config, dict):
            raise ValueError("invalid_pass_config")
        pass_index = int(_required(pass_config, "index", int))
        states.append(
            (
                _stream(
                    reference,
                    stage=stage,
                    block_index=start_block,
                    pass_config=pass_config,
                    modality="video",
                    device=device,
                ),
                _stream(
                    reference,
                    stage=stage,
                    block_index=start_block,
                    pass_config=pass_config,
                    modality="audio",
                    device=device,
                ),
                _perturbations(pass_config, device=device, block_count=block_count),
                pass_index,
            )
        )

    capture = TensorCapture()
    processor = BlockPerturbationsProcessor()
    comparisons: list[dict[str, Any]] = []
    for block_index in range(start_block, end_block + 1):
        block = _load_block(
            arguments.transformer_subset,
            arguments.lora_subset,
            block_index=block_index,
            lora_strength=lora_strength,
            device=device,
        )
        recorders: list[AttentionInternalsRecorder] = []
        if block_index in capture_blocks:
            recorders = [
                AttentionInternalsRecorder(capture, stage, block_index, name, getattr(block, name))
                for name in module_names
            ]
            for recorder in recorders:
                recorder.install()
        next_states: list[tuple[TransformerArgs, TransformerArgs, Any, int]] = []
        try:
            for video, audio, perturbations, pass_index in states:
                video = processor(
                    video,
                    perturbations,
                    block_index,
                    PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    PerturbationType.SKIP_A2V_CROSS_ATTN,
                )
                audio = processor(
                    audio,
                    perturbations,
                    block_index,
                    PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    PerturbationType.SKIP_V2A_CROSS_ATTN,
                )
                prefix = f"{stage}_deep_block_{block_index:02d}_pass_{pass_index:02d}"
                if block_index in capture_blocks:
                    _capture_stream(capture, f"{prefix}_video_input", video)
                    _capture_stream(capture, f"{prefix}_audio_input", audio)
                video_output, audio_output = block(video, audio)
                assert video_output is not None and audio_output is not None
                if block_index in capture_blocks:
                    capture.add(f"{prefix}_video_output", video_output.x)
                    capture.add(f"{prefix}_audio_output", audio_output.x)
                if block_index in reference_blocks:
                    for modality, output in (("video", video_output.x), ("audio", audio_output.x)):
                        key = f"{prefix}_{modality}_output"
                        expected = reference.get(key)
                        if expected is None:
                            raise ValueError(f"missing_reference_output:{key}")
                        metrics = _metrics(expected, output.detach().float().cpu().numpy())
                        comparisons.append(
                            {
                                "block_index": block_index,
                                "pass_index": pass_index,
                                "modality": modality,
                                "metrics": metrics,
                                "passed": _comparison_passed(metrics, acceptance),
                            }
                        )
                next_states.append(
                    (
                        replace(video, x=video_output.x),
                        replace(audio, x=audio_output.x),
                        perturbations,
                        pass_index,
                    )
                )
        finally:
            for recorder in recorders:
                recorder.remove()
        states = next_states
        del block
        torch.cuda.empty_cache()

    passed = all(value["passed"] for value in comparisons)
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    unique_arrays, aliases = deduplicate_capture(capture)
    artifact_shards = write_artifact_shards(
        arguments.artifact,
        unique_arrays,
        maximum_bytes=maximum_shard_bytes,
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "compact_checkpoint_attention_sequence_capture",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_block_range": [start_block, end_block],
        "capture_block_indices": capture_blocks,
        "lora_strength": lora_strength,
        "input_reference": {"file": arguments.input_reference.name, "sha256": _sha256(arguments.input_reference)},
        "transformer_subset": {
            "file": arguments.transformer_subset.name,
            "sha256": _sha256(arguments.transformer_subset),
        },
        "lora_subset": {"file": arguments.lora_subset.name, "sha256": _sha256(arguments.lora_subset)},
        "artifact": {
            "base_file": arguments.artifact.name,
            "shards": [
                {"file": shard.name, "size_bytes": shard.stat().st_size, "sha256": _sha256(shard)}
                for shard in artifact_shards
            ],
            "unique_tensor_count": len(unique_arrays),
            "alias_count": len(aliases),
        },
        "tensor_boundaries": capture.metadata,
        "tensor_aliases": aliases,
        "source_replay_comparisons": comparisons,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    print(
        json.dumps(
            {
                "ok": passed,
                "comparisons": len(comparisons),
                "tensor_count": len(capture.arrays),
                "unique_tensor_count": len(unique_arrays),
                "artifact_shards": len(artifact_shards),
            }
        )
    )
    return 0 if passed else 1


def _entrypoint() -> int:
    try:
        return main()
    except RuntimeError as error:
        if str(error) == "cuda_unavailable":
            return _fail("LARA-RUNTIME-002", "cuda_unavailable", "run_on_supported_cuda_server")
        return _fail("LARA-RUNTIME-008", type(error).__name__, "inspect_cuda_environment_and_retry")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _fail("LARA-RUNTIME-008", str(error), "verify_inputs_configuration_and_retry")


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
