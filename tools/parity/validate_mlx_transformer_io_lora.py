#!/usr/bin/env python3
"""Validate stage-local LoRA fusion for transformer input and output modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.models import build_lora_pairs
from lara_ltx.transformer import (
    TransformerModalityInput,
    load_production_transformer_input,
    load_production_transformer_output,
)

INPUT_CHANNELS = 128
VIDEO_DIMENSION = 4_096
AUDIO_DIMENSION = 2_048
VIDEO_POSITION_AXES = 3
AUDIO_POSITION_AXES = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--input-mapping", required=True, type=Path)
    parser.add_argument("--output-mapping", required=True, type=Path)
    parser.add_argument("--lora-strength", required=True, type=float)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _modality(hidden_dimension: int, axes: int) -> TransformerModalityInput:
    positions = np.zeros((1, axes, 2, 2), dtype=np.float32)
    positions[..., 1] = 1.0
    return TransformerModalityInput(
        latent=mx.zeros((1, 2, INPUT_CHANNELS), dtype=mx.bfloat16),
        sigma=mx.array([0.5], dtype=mx.bfloat16),
        timesteps=mx.array([[0.25, 0.75]], dtype=mx.bfloat16),
        positions=mx.array(positions),
        context=mx.ones((1, 3, hidden_dimension), dtype=mx.bfloat16),
        context_mask=mx.ones((1, 3), dtype=mx.int32),
        attention_mask=mx.ones((1, 2, 2), dtype=mx.float32),
        keyframes_mask=mx.zeros((1, 2, 1), dtype=mx.int32) if axes == VIDEO_POSITION_AXES else None,
    )


def main() -> int:
    arguments = parse_arguments()
    input_mapping = json.loads(arguments.input_mapping.read_text(encoding="utf-8"))
    output_mapping = json.loads(arguments.output_mapping.read_text(encoding="utf-8"))
    pairs = build_lora_pairs(arguments.lora_checkpoint, arguments.transformer_checkpoint)
    mx.reset_peak_memory()
    processor, input_pairs = load_production_transformer_input(
        checkpoint=arguments.transformer_checkpoint,
        mapping=input_mapping,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=arguments.lora_strength,
        lora_pairs=pairs,
    )
    video, audio = processor.prepare(
        _modality(VIDEO_DIMENSION, VIDEO_POSITION_AXES),
        _modality(AUDIO_DIMENSION, AUDIO_POSITION_AXES),
    )
    assert video is not None and audio is not None
    heads, output_pairs = load_production_transformer_output(
        checkpoint=arguments.transformer_checkpoint,
        mapping=output_mapping,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=arguments.lora_strength,
        lora_pairs=pairs,
    )
    video_output, audio_output = heads(
        video.stream.x,
        video.embedded_timestep,
        audio.stream.x,
        audio.embedded_timestep,
    )
    finite = (mx.all(mx.isfinite(video_output)), mx.all(mx.isfinite(audio_output)))
    mx.eval(video_output, audio_output, *finite)
    report = {
        "schema_version": 1,
        "component": "transformer_io_lora",
        "lora_strength": arguments.lora_strength,
        "fused_input_pairs": input_pairs,
        "fused_output_pairs": output_pairs,
        "output_shapes": {
            "video": list(video_output.shape),
            "audio": list(audio_output.shape),
        },
        "peak_bytes": int(mx.get_peak_memory()),
        "passed": all(bool(np.asarray(value)) for value in finite) and input_pairs + output_pairs == 28,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
