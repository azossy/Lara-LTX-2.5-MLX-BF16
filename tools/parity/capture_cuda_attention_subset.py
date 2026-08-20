#!/usr/bin/env python3
"""Replay one block's exact inputs on a compact official CUDA checkpoint subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuda_attention_internals import (
    AttentionInternalsRecorder,
    TensorCapture,
    deduplicate_capture,
    write_artifact_shards,
)
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs
from safetensors import safe_open

ARTIFACT_SCHEMA_VERSION = 6
CONFIG_SCHEMA_VERSION = 1
DEEP_CONFIG_SCHEMA_VERSION = 1
CUDA_DEVICE = "cuda"
FILE_HASH_BUFFER_BYTES = 8 * 1024 * 1024
DEFAULT_AUDIO_FF_BIAS = True
BASE_BLOCK_PREFIX_TEMPLATE = "model.diffusion_model.transformer_blocks.{block_index}."
LORA_BLOCK_PREFIX_TEMPLATE = "diffusion_model.transformer_blocks.{block_index}."
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
    "embedded_timestep",
    "prompt_timestep",
    "cross_scale_shift_timestep",
    "cross_gate_timestep",
    "context_mask",
    "self_attention_mask",
)
ATTENTION_MODULES = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid_json_object:{path}")
    return value


def _required(value: dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    result = value.get(key)
    if not isinstance(result, expected) or (isinstance(result, bool) and expected in (int, float)):
        raise ValueError(f"invalid_config:{key}")
    return result


def _source_pass_index(pass_config: dict[str, Any], suffix: str) -> int:
    default = int(_required(pass_config, "input_source_pass_index", int))
    overrides = pass_config.get("input_field_sources", {})
    if not isinstance(overrides, dict):
        raise ValueError("invalid_input_field_sources")
    source = overrides.get(suffix, default)
    if not isinstance(source, int) or isinstance(source, bool) or source < 0:
        raise ValueError(f"invalid_input_source:{suffix}")
    return source


def _tensor(
    reference: dict[str, np.ndarray],
    *,
    stage: str,
    block_index: int,
    pass_config: dict[str, Any],
    modality: str,
    suffix: str,
    device: torch.device,
) -> torch.Tensor | None:
    source_pass = _source_pass_index(pass_config, suffix)
    key = f"{stage}_deep_block_{block_index:02d}_pass_{source_pass:02d}_{modality}_input_{suffix}"
    value = reference.get(key)
    if value is None:
        return None
    return torch.from_numpy(value).to(device=device, dtype=torch.bfloat16)


def _stream(
    reference: dict[str, np.ndarray],
    *,
    stage: str,
    block_index: int,
    pass_config: dict[str, Any],
    modality: str,
    device: torch.device,
) -> TransformerArgs:
    values = {
        field: _tensor(
            reference,
            stage=stage,
            block_index=block_index,
            pass_config=pass_config,
            modality=modality,
            suffix=field,
            device=device,
        )
        for field in STREAM_FIELDS
    }
    if any(values[field] is None for field in ("x", "context", "timesteps", "embedded_timestep")):
        raise ValueError(f"missing_required_stream:{modality}")

    positional = tuple(
        _tensor(
            reference,
            stage=stage,
            block_index=block_index,
            pass_config=pass_config,
            modality=modality,
            suffix=f"positional_embeddings_{component}",
            device=device,
        )
        for component in ("cos", "sin")
    )
    cross_positional = tuple(
        _tensor(
            reference,
            stage=stage,
            block_index=block_index,
            pass_config=pass_config,
            modality=modality,
            suffix=f"cross_positional_embeddings_{component}",
            device=device,
        )
        for component in ("cos", "sin")
    )
    if any(value is None for value in positional):
        raise ValueError(f"missing_positional_embeddings:{modality}")
    cross_frequencies = None if any(value is None for value in cross_positional) else cross_positional
    return TransformerArgs(
        x=values["x"],
        context=values["context"],
        context_mask=values["context_mask"],
        timesteps=values["timesteps"],
        embedded_timestep=values["embedded_timestep"],
        positional_embeddings=positional,
        cross_positional_embeddings=cross_frequencies,
        cross_scale_shift_timestep=values["cross_scale_shift_timestep"],
        cross_gate_timestep=values["cross_gate_timestep"],
        enabled=True,
        prompt_timestep=values["prompt_timestep"],
        self_attention_mask=values["self_attention_mask"],
    )


def _perturbations(pass_config: dict[str, Any], *, device: torch.device, block_count: int) -> BatchedPerturbationConfig:
    names = _required(pass_config, "perturbations", list)
    try:
        values = [Perturbation(PERTURBATION_TYPES[str(name)], blocks=None) for name in names]
    except KeyError as error:
        raise ValueError(f"unknown_perturbation:{error.args[0]}") from error
    return BatchedPerturbationConfig(
        [PerturbationConfig(values)],
        num_blocks=block_count,
        device=device,
        dtype=torch.bfloat16,
    )


def _block_from_metadata(metadata: dict[str, Any]) -> BasicAVTransformerBlock:
    config = _required(_required(metadata, "config", dict), "transformer", dict)
    video_heads = int(_required(config, "num_attention_heads", int))
    video_head_dimension = int(_required(config, "attention_head_dim", int))
    audio_heads = int(_required(config, "audio_num_attention_heads", int))
    audio_head_dimension = int(_required(config, "audio_attention_head_dim", int))
    apply_gated_attention = bool(_required(config, "apply_gated_attention", bool))
    cross_attention_adaln = bool(_required(config, "cross_attention_adaln", bool))
    video = TransformerConfig(
        dim=video_heads * video_head_dimension,
        heads=video_heads,
        d_head=video_head_dimension,
        context_dim=int(_required(config, "cross_attention_dim", int)),
        apply_gated_attention=apply_gated_attention,
        cross_attention_adaln=cross_attention_adaln,
        ff_bias=bool(_required(config, "ff_bias", bool)),
    )
    audio = TransformerConfig(
        dim=audio_heads * audio_head_dimension,
        heads=audio_heads,
        d_head=audio_head_dimension,
        context_dim=int(_required(config, "audio_cross_attention_dim", int)),
        apply_gated_attention=apply_gated_attention,
        cross_attention_adaln=cross_attention_adaln,
        ff_bias=bool(config.get("audio_ff_bias", DEFAULT_AUDIO_FF_BIAS)),
    )
    with torch.device("meta"):
        return BasicAVTransformerBlock(
            video=video,
            audio=audio,
            rope_type=LTXRopeType(_required(config, "rope_type", str)),
            norm_eps=float(_required(config, "norm_eps", (int, float))),
        )


def _state_dict(path: Path, *, prefix: str, device: torch.device, apply_comfy_map: bool) -> StateDict:
    values: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device=str(device)) as source:
        for source_name in source.keys():
            if not source_name.startswith(prefix):
                continue
            mapped = LTXV_MODEL_COMFY_RENAMING_MAP.apply_to_key(source_name) if apply_comfy_map else source_name
            if mapped is None:
                continue
            local_name = mapped.removeprefix(prefix.removeprefix("model.diffusion_model."))
            values[local_name] = source.get_tensor(source_name)
    if not values:
        raise ValueError(f"empty_checkpoint_subset:{path}")
    return StateDict(
        sd=values,
        device=device,
        size=sum(tensor.nbytes for tensor in values.values()),
        dtype={tensor.dtype for tensor in values.values()},
    )


def _load_block(
    transformer_subset: Path,
    lora_subset: Path,
    *,
    block_index: int,
    lora_strength: float,
    device: torch.device,
) -> BasicAVTransformerBlock:
    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(str(transformer_subset))
    block = _block_from_metadata(metadata)
    base_prefix = BASE_BLOCK_PREFIX_TEMPLATE.format(block_index=block_index)
    lora_prefix = LORA_BLOCK_PREFIX_TEMPLATE.format(block_index=block_index)
    base = _state_dict(transformer_subset, prefix=base_prefix, device=device, apply_comfy_map=True)
    lora = _state_dict(lora_subset, prefix=lora_prefix, device=device, apply_comfy_map=False)
    fused = apply_loras(
        base,
        [LoraStateDictWithStrength(lora, lora_strength)],
        preserve_input_device=False,
    )
    incompatible = block.load_state_dict(fused.sd, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("incompatible_block_subset")
    return block.to(device=device, dtype=torch.bfloat16).eval()


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    expected = reference.astype(np.float64, copy=False)
    actual = candidate.astype(np.float64, copy=False)
    difference = actual - expected
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(expected))))
    normalized_rmse = rmse / reference_rms if reference_rms else (0.0 if rmse == 0.0 else float("inf"))
    norm_product = float(np.linalg.norm(expected.ravel()) * np.linalg.norm(actual.ravel()))
    cosine = float(np.dot(expected.ravel(), actual.ravel()) / norm_product) if norm_product else 1.0
    return {
        "max_abs_error": float(np.max(np.abs(difference), initial=0.0)),
        "normalized_rmse": normalized_rmse,
        "cosine_similarity": cosine,
        "nan_count": int(np.isnan(actual).sum()),
        "inf_count": int(np.isinf(actual).sum()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_HASH_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(code: str, cause: str, action: str) -> int:
    print(json.dumps({"ok": False, "error_code": code, "cause": cause, "action": action}))
    return 2


@torch.inference_mode()
def main() -> int:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    config = _read_json(arguments.config)
    deep_config = _read_json(arguments.deep_config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported_attention_config_schema")
    if deep_config.get("schema_version") != DEEP_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported_deep_config_schema")
    stage = str(_required(config, "stage", str))
    block_index = int(_required(config, "block_index", int))
    lora_strength = float(_required(config, "lora_strength", (int, float)))
    block_count = int(_required(deep_config, "block_count", int))
    passes = _required(deep_config, "passes", list)
    expected_passes = _required(config, "passes", list)
    if [value.get("index") for value in passes if isinstance(value, dict)] != expected_passes:
        raise ValueError("attention_and_deep_pass_mismatch")
    module_names = tuple(_required(config, "modules", list))
    maximum_artifact_shard_bytes = int(_required(config, "maximum_artifact_shard_bytes", int))
    if not module_names or any(name not in ATTENTION_MODULES for name in module_names):
        raise ValueError("invalid_attention_modules")

    with np.load(arguments.input_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    device = torch.device(CUDA_DEVICE)
    block = _load_block(
        arguments.transformer_subset,
        arguments.lora_subset,
        block_index=block_index,
        lora_strength=lora_strength,
        device=device,
    )
    capture = TensorCapture()
    recorders = [
        AttentionInternalsRecorder(capture, stage, block_index, name, getattr(block, name)) for name in module_names
    ]
    for recorder in recorders:
        recorder.install()

    comparisons: list[dict[str, Any]] = []
    processor = BlockPerturbationsProcessor()
    try:
        for pass_config in passes:
            if not isinstance(pass_config, dict):
                raise ValueError("invalid_pass_config")
            pass_index = int(_required(pass_config, "index", int))
            perturbations = _perturbations(pass_config, device=device, block_count=block_count)
            video = _stream(
                reference,
                stage=stage,
                block_index=block_index,
                pass_config=pass_config,
                modality="video",
                device=device,
            )
            audio = _stream(
                reference,
                stage=stage,
                block_index=block_index,
                pass_config=pass_config,
                modality="audio",
                device=device,
            )
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
            video_output, audio_output = block(video, audio)
            assert video_output is not None and audio_output is not None
            for modality, output in (("video", video_output.x), ("audio", audio_output.x)):
                key = f"{stage}_deep_block_{block_index:02d}_pass_{pass_index:02d}_{modality}_output"
                expected = reference.get(key)
                if expected is None:
                    raise ValueError(f"missing_reference_output:{key}")
                comparisons.append(
                    {
                        "pass_index": pass_index,
                        "modality": modality,
                        "metrics": _metrics(expected, output.detach().float().cpu().numpy()),
                    }
                )
    finally:
        for recorder in recorders:
            recorder.remove()

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
    passed = all(comparison["passed"] for comparison in comparisons)
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    unique_arrays, aliases = deduplicate_capture(capture)
    artifact_shards = write_artifact_shards(
        arguments.artifact,
        unique_arrays,
        maximum_bytes=maximum_artifact_shard_bytes,
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "compact_checkpoint_exact_input_attention_capture",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "block_index": block_index,
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
    temporary_report = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_report.replace(arguments.report)
    print(
        json.dumps(
            {
                "ok": passed,
                "tensor_count": len(capture.arrays),
                "unique_tensor_count": len(unique_arrays),
                "artifact_shards": len(artifact_shards),
                "comparisons": len(comparisons),
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
