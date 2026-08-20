#!/usr/bin/env python3
"""Strict-load and execute the official MLX AV transformer input component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.models import (
    iter_component_weight_batches,
    transformer_input_target_shapes,
    validate_transformer_input_architecture,
    validate_transformer_input_mapping,
)
from lara_ltx.transformer import (
    AVTransformerInputPreprocessor,
    TransformerInputConfig,
    TransformerModalityInput,
)

ARTIFACT_SCHEMA_VERSION = 1
INPUT_CHANNELS = 128
VIDEO_HIDDEN_DIMENSION = 4_096
AUDIO_HIDDEN_DIMENSION = 2_048
ATTENTION_HEAD_COUNT = 32
VIDEO_POSITION_MAXIMUMS = (20, 2_048, 2_048)
AUDIO_POSITION_MAXIMUMS = (20,)
CROSS_ATTENTION_DIMENSION = AUDIO_HIDDEN_DIMENSION
ADALN_COEFFICIENT = 9
PROMPT_ADALN_COEFFICIENT = 2
SMOKE_BATCH_SIZE = 1
SMOKE_TOKEN_COUNT = 2
SMOKE_CONTEXT_TOKEN_COUNT = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _config(
    hidden_dimension: int,
    max_positions: tuple[int, ...],
    *,
    keyframes: bool,
    timestep_scale_multiplier: float,
) -> TransformerInputConfig:
    return TransformerInputConfig(
        input_channels=INPUT_CHANNELS,
        hidden_dimension=hidden_dimension,
        adaln_coefficient=ADALN_COEFFICIENT,
        prompt_adaln_coefficient=PROMPT_ADALN_COEFFICIENT,
        attention_heads=ATTENTION_HEAD_COUNT,
        max_positions=max_positions,
        timestep_scale_multiplier=timestep_scale_multiplier,
        use_keyframes_absolute_embedding=keyframes,
    )


def _positions(axis_count: int) -> mx.array:
    bounds = np.zeros(
        (SMOKE_BATCH_SIZE, axis_count, SMOKE_TOKEN_COUNT, 2),
        dtype=np.float32,
    )
    bounds[..., 1] = 1.0
    bounds[:, 0, 1, :] += 1.0
    return mx.array(bounds)


def _modality(hidden_dimension: int, axis_count: int, sigma: float) -> TransformerModalityInput:
    latent = mx.linspace(-0.5, 0.5, SMOKE_TOKEN_COUNT * INPUT_CHANNELS).reshape(
        SMOKE_BATCH_SIZE,
        SMOKE_TOKEN_COUNT,
        INPUT_CHANNELS,
    )
    return TransformerModalityInput(
        latent=latent.astype(mx.bfloat16),
        sigma=mx.array([sigma], dtype=mx.bfloat16),
        timesteps=mx.array([[0.125, 0.75]], dtype=mx.bfloat16),
        positions=_positions(axis_count),
        context=mx.ones(
            (SMOKE_BATCH_SIZE, SMOKE_CONTEXT_TOKEN_COUNT, hidden_dimension),
            dtype=mx.bfloat16,
        ),
        context_mask=mx.ones((SMOKE_BATCH_SIZE, SMOKE_CONTEXT_TOKEN_COUNT), dtype=mx.int32),
        attention_mask=mx.ones(
            (SMOKE_BATCH_SIZE, SMOKE_TOKEN_COUNT, SMOKE_TOKEN_COUNT),
            dtype=mx.float32,
        ),
        keyframes_mask=(mx.array([[[1], [0]]], dtype=mx.int32) if axis_count == len(VIDEO_POSITION_MAXIMUMS) else None),
    )


def _required_outputs(video: object, audio: object) -> tuple[mx.array, ...]:
    streams = (video.stream, audio.stream)
    values: list[mx.array] = [video.embedded_timestep, audio.embedded_timestep]
    for stream in streams:
        values.extend((stream.x, stream.timesteps, stream.context))
        for optional in (
            stream.prompt_timestep,
            stream.cross_scale_shift_timestep,
            stream.cross_gate_timestep,
            stream.context_mask,
            stream.self_attention_mask,
        ):
            if optional is None:
                raise LaraError("LARA-TENSOR-021", details={"reason": "missing_prepared_transformer_input"})
            values.append(optional)
        for positional in (stream.positional_embeddings, stream.cross_positional_embeddings):
            if positional is None:
                raise LaraError("LARA-TENSOR-021", details={"reason": "missing_positional_embeddings"})
            values.extend(positional)
    return tuple(values)


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    rules = validate_transformer_input_mapping(mapping)
    architecture = validate_transformer_input_architecture(mapping)
    processor = AVTransformerInputPreprocessor(
        video=_config(
            VIDEO_HIDDEN_DIMENSION,
            VIDEO_POSITION_MAXIMUMS,
            keyframes=True,
            timestep_scale_multiplier=architecture.timestep_scale_multiplier,
        ),
        audio=_config(
            AUDIO_HIDDEN_DIMENSION,
            AUDIO_POSITION_MAXIMUMS,
            keyframes=False,
            timestep_scale_multiplier=architecture.timestep_scale_multiplier,
        ),
        cross_attention_dimension=CROSS_ATTENTION_DIMENSION,
        av_cross_timestep_scale_multiplier=architecture.av_cross_timestep_scale_multiplier,
    )

    active_memory_before_loading = int(mx.get_active_memory())
    mx.reset_peak_memory()
    for batch in iter_component_weight_batches(
        (arguments.transformer_checkpoint,),
        rules,
        expected_target_shapes=transformer_input_target_shapes(),
    ):
        processor.load_weights(batch)
        mx.eval(*[value for _, value in batch])
    weight_load_memory = {
        "active_before_bytes": active_memory_before_loading,
        "active_after_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }

    video, audio = processor.prepare(
        _modality(VIDEO_HIDDEN_DIMENSION, len(VIDEO_POSITION_MAXIMUMS), 0.25),
        _modality(AUDIO_HIDDEN_DIMENSION, len(AUDIO_POSITION_MAXIMUMS), 0.75),
    )
    if video is None or audio is None:
        raise LaraError("LARA-TENSOR-021", details={"reason": "missing_modalities"})
    outputs = _required_outputs(video, audio)
    finite = [mx.all(mx.isfinite(value)) for value in outputs]
    mx.eval(*outputs, *finite)
    output_shapes = {
        "video_hidden": list(video.stream.x.shape),
        "video_timestep": list(video.stream.timesteps.shape),
        "video_prompt_timestep": list(video.stream.prompt_timestep.shape),
        "video_cross_scale_shift": list(video.stream.cross_scale_shift_timestep.shape),
        "video_cross_gate": list(video.stream.cross_gate_timestep.shape),
        "audio_hidden": list(audio.stream.x.shape),
        "audio_timestep": list(audio.stream.timesteps.shape),
        "audio_prompt_timestep": list(audio.stream.prompt_timestep.shape),
        "audio_cross_scale_shift": list(audio.stream.cross_scale_shift_timestep.shape),
        "audio_cross_gate": list(audio.stream.cross_gate_timestep.shape),
    }
    passed = all(bool(np.asarray(value)) for value in finite)
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "component": "transformer_input",
        "checkpoint": arguments.transformer_checkpoint.name,
        "mapping": arguments.mapping.name,
        "architecture": {
            "timestep_scale_multiplier": architecture.timestep_scale_multiplier,
            "av_ca_timestep_scale_multiplier": architecture.av_cross_timestep_scale_multiplier,
        },
        "mapped_tensor_count": len(rules),
        "output_count": len(outputs),
        "output_shapes": output_shapes,
        "weight_load_memory": weight_load_memory,
        "requires_finite_output": True,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
