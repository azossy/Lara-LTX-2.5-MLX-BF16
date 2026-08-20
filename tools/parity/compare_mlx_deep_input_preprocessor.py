#!/usr/bin/env python3
"""Compare production MLX transformer input preparation with CUDA v4 fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors
from lara_ltx.sampling import (
    AudioLatentLayout,
    VideoLatentLayout,
    create_audio_state,
    create_video_state,
    unpatchify_audio,
    unpatchify_video,
)
from lara_ltx.transformer import TransformerModalityInput, load_production_transformer_input

CONFIG_SCHEMA_VERSION = 1
FIELD_SOURCES = {
    "x": "stream.x",
    "timesteps": "stream.timesteps",
    "embedded_timestep": "embedded_timestep",
    "prompt_timestep": "stream.prompt_timestep",
    "cross_scale_shift_timestep": "stream.cross_scale_shift_timestep",
    "cross_gate_timestep": "stream.cross_gate_timestep",
}
KEYFRAME_VARIANTS = ("production", "no_keyframes", "zero_keyframes")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--input-mapping", required=True, type=Path)
    parser.add_argument("--sampling-reference", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--sampling-config", required=True, type=Path)
    parser.add_argument("--deep-config", required=True, type=Path)
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
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_deep_input_config:{key}"})
    return result


def _repeat(value: mx.array, count: int) -> mx.array:
    return mx.concatenate(tuple(value for _ in range(count)), axis=0)


def _video_layout(shape_source: np.ndarray, frame_rate: float) -> VideoLatentLayout:
    batch, channels, frames, height, width = shape_source.shape
    return VideoLatentLayout(
        batch=int(batch),
        channels=int(channels),
        frames=int(frames),
        height=int(height),
        width=int(width),
        frame_rate=frame_rate,
    )


def _audio_layout(shape_source: np.ndarray) -> AudioLatentLayout:
    batch, channels, frames, mel_bins = shape_source.shape
    return AudioLatentLayout(
        batch=int(batch),
        channels=int(channels),
        frames=int(frames),
        mel_bins=int(mel_bins),
    )


def _candidate_field(prepared: Any, source: str) -> mx.array | None:
    value = prepared
    for component in source.split("."):
        value = getattr(value, component)
    return value


def _keyframes(value: mx.array | None, variant: str, count: int) -> mx.array | None:
    if variant == "no_keyframes" or value is None:
        return None
    repeated = _repeat(value, count)
    return mx.zeros_like(repeated) if variant == "zero_keyframes" else repeated


def _source_pass_index(pass_config: dict[str, Any], field: str) -> int:
    default = int(_required(pass_config, "input_source_pass_index", int))
    overrides = pass_config.get("input_field_sources", {})
    if not isinstance(overrides, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_input_field_sources"})
    return int(overrides.get(field, default))


def main() -> int:
    arguments = parse_arguments()
    sampling = _read_json(arguments.sampling_config)
    deep = _read_json(arguments.deep_config)
    if sampling.get("schema_version") != CONFIG_SCHEMA_VERSION or deep.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_deep_input_config_schema"})
    mapping = _read_json(arguments.input_mapping)
    keys = _required(sampling, "reference_keys", dict)
    passes = _required(deep, "passes", list)
    pass_count = len(passes)
    if pass_count <= 0:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "missing_deep_input_passes"})
    with np.load(arguments.sampling_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    with np.load(arguments.cuda_reference) as archive:
        for name in archive.files:
            if name in reference and not np.array_equal(reference[name], archive[name]):
                raise LaraError("LARA-RUNTIME-008", details={"reason": f"deep_reference_mismatch:{name}"})
            reference[name] = archive[name]
    frame_rate = float(_required(sampling, "frame_rate", (int, float)))
    video_layout = _video_layout(reference[str(keys["stage_one_video_layout"])], frame_rate)
    audio_layout = _audio_layout(reference[str(keys["stage_one_audio_layout"])])
    video_tokens = mx.array(reference[str(keys["stage_one_video_output"])], dtype=mx.bfloat16)
    audio_tokens = mx.array(reference[str(keys["stage_one_audio_output"])], dtype=mx.bfloat16)
    video_bundle = create_video_state(video_layout, initial_latent=unpatchify_video(video_tokens, video_layout))
    audio_bundle = create_audio_state(audio_layout, initial_latent=unpatchify_audio(audio_tokens, audio_layout))
    context_keys = {}
    for modality in ("video", "audio"):
        selected = []
        for pass_config in passes:
            if not isinstance(pass_config, dict):
                raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_input_pass"})
            component = str(_required(pass_config, "cuda_component", str))
            polarity = "negative" if component == "uncond" else "positive"
            selected.append(mx.array(reference[str(keys[f"{modality}_{polarity}"])], dtype=mx.bfloat16))
        context_keys[modality] = mx.concatenate(tuple(selected), axis=0)
    sigma_key = str(_required(_required(deep, "first_call", dict), "sigma_key", str))
    sigma = float(np.asarray(reference[sigma_key]))
    processor, fused_pair_count = load_production_transformer_input(
        checkpoint=arguments.transformer_checkpoint,
        mapping=mapping,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=float(_required(deep, "lora_strength", (int, float))),
    )
    acceptance = _required(deep, "acceptance", dict)
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    variants: list[dict[str, Any]] = []
    for variant in KEYFRAME_VARIANTS:
        modality_inputs = {}
        for modality, bundle in (("video", video_bundle), ("audio", audio_bundle)):
            state = bundle.state
            latent = _repeat(state.latent, pass_count)
            mask = _repeat(state.denoise_mask, pass_count)
            modality_inputs[modality] = TransformerModalityInput(
                latent=latent,
                sigma=mx.full((latent.shape[0],), sigma, dtype=latent.dtype),
                timesteps=mask[..., 0].astype(latent.dtype) * sigma,
                positions=_repeat(bundle.positions, pass_count),
                context=context_keys[modality],
                attention_mask=None,
                keyframes_mask=_keyframes(bundle.keyframes_mask, variant, pass_count),
            )
        prepared_video, prepared_audio = processor.prepare(modality_inputs["video"], modality_inputs["audio"])
        assert prepared_video is not None and prepared_audio is not None
        comparisons: list[dict[str, Any]] = []
        for pass_config in passes:
            if not isinstance(pass_config, dict):
                raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_deep_input_pass"})
            pass_index = int(_required(pass_config, "index", int))
            pass_name = str(_required(pass_config, "name", str))
            for modality, prepared in (("video", prepared_video), ("audio", prepared_audio)):
                for field, source in FIELD_SOURCES.items():
                    candidate = _candidate_field(prepared, source)
                    source_index = _source_pass_index(pass_config, field)
                    expected_prefix = f"{deep['stage']}_deep_block_00_pass_{source_index:02d}_{modality}_input"
                    expected_key = f"{expected_prefix}_{field}"
                    if candidate is None or expected_key not in reference:
                        continue
                    metrics = compare_tensors(
                        expected_key,
                        reference[expected_key].astype(np.float32),
                        np.asarray(candidate[pass_index : pass_index + 1].astype(mx.float32)),
                    )
                    comparisons.append(
                        {
                            "pass_index": pass_index,
                            "pass_name": pass_name,
                            "modality": modality,
                            "field": field,
                            "metrics": metrics.to_dict(),
                            "passed": (
                                metrics.nan_count == 0
                                and metrics.inf_count == 0
                                and metrics.normalized_rmse <= maximum_nrmse
                                and metrics.cosine_similarity >= minimum_cosine
                            ),
                        }
                    )
                for positional_name in ("positional_embeddings", "cross_positional_embeddings"):
                    positional = getattr(prepared.stream, positional_name)
                    if positional is None:
                        continue
                    for component, candidate in zip(("cos", "sin"), positional, strict=True):
                        field = f"{positional_name}_{component}"
                        source_index = _source_pass_index(pass_config, field)
                        expected_prefix = f"{deep['stage']}_deep_block_00_pass_{source_index:02d}_{modality}_input"
                        expected_key = f"{expected_prefix}_{field}"
                        metrics = compare_tensors(
                            expected_key,
                            reference[expected_key].astype(np.float32),
                            np.asarray(candidate[pass_index : pass_index + 1].astype(mx.float32)),
                        )
                        comparisons.append(
                            {
                                "pass_index": pass_index,
                                "pass_name": pass_name,
                                "modality": modality,
                                "field": f"{positional_name}_{component}",
                                "metrics": metrics.to_dict(),
                                "passed": (
                                    metrics.nan_count == 0
                                    and metrics.inf_count == 0
                                    and metrics.normalized_rmse <= maximum_nrmse
                                    and metrics.cosine_similarity >= minimum_cosine
                                ),
                            }
                        )
        variants.append(
            {
                "name": variant,
                "comparisons": comparisons,
                "passed": all(value["passed"] for value in comparisons),
            }
        )
    report = {
        "schema_version": 1,
        "component": "transformer_input_preprocessor_deep_trace",
        "sigma": sigma,
        "lora_strength": float(deep["lora_strength"]),
        "fused_lora_pair_count": fused_pair_count,
        "acceptance": acceptance,
        "variants": variants,
        "passing_variants": [value["name"] for value in variants if value["passed"]],
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passing_variants"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
