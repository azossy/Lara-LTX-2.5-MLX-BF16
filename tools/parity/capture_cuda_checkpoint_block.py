#!/usr/bin/env python3
"""Capture a real BF16 CUDA input/output boundary for one 22B AV transformer block."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from lara_ltx.errors import LaraError
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_pipelines.utils.blocks import DiffusionStage

ARTIFACT_SCHEMA_VERSION = 1
CUDA_DEVICE = "cuda"
VIDEO_DIMENSION = 4096
AUDIO_DIMENSION = 2048
ADALN_PARAMETER_COUNT = 9
PROMPT_ADALN_PARAMETER_COUNT = 2
AV_CROSS_SCALE_PARAMETER_COUNT = 4
AV_CROSS_GATE_PARAMETER_COUNT = 1
SINGLE_BATCH = 1
SINGLE_TOKEN = 1
DEFAULT_BLOCK_INDEX = 0
DEFAULT_SEED = 25_081_900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--block-index", type=int, default=DEFAULT_BLOCK_INDEX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _random(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=CUDA_DEVICE, dtype=torch.bfloat16)


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _args(
    *,
    x: torch.Tensor,
    context: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_timestep: torch.Tensor,
    cross_scale_shift_timestep: torch.Tensor,
    cross_gate_timestep: torch.Tensor,
) -> TransformerArgs:
    return TransformerArgs(
        x=x,
        context=context,
        context_mask=None,
        timesteps=timesteps,
        embedded_timestep=torch.zeros((SINGLE_BATCH, SINGLE_TOKEN), device=CUDA_DEVICE, dtype=torch.bfloat16),
        positional_embeddings=None,
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=cross_scale_shift_timestep,
        cross_gate_timestep=cross_gate_timestep,
        cross_attn_perturbation_mask=torch.ones(
            (SINGLE_BATCH, SINGLE_TOKEN, SINGLE_TOKEN), device=CUDA_DEVICE, dtype=torch.bfloat16
        ),
        enabled=True,
        prompt_timestep=prompt_timestep,
    )


@torch.inference_mode()
def main() -> int:
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise LaraError("LARA-RUNTIME-002")
    generator = torch.Generator(device=CUDA_DEVICE).manual_seed(arguments.seed)
    stage = DiffusionStage.from_checkpoint(
        str(arguments.transformer_checkpoint),
        torch.bfloat16,
        torch.device(CUDA_DEVICE),
    )
    transformer = stage._build_transformer()
    try:
        blocks = transformer.velocity_model.transformer_blocks
        if arguments.block_index < 0 or arguments.block_index >= len(blocks):
            raise LaraError("LARA-MODEL-028", details={"key": f"block_index={arguments.block_index}"})
        block = blocks[arguments.block_index]
        video_input = _random((SINGLE_BATCH, SINGLE_TOKEN, VIDEO_DIMENSION), generator)
        audio_input = _random((SINGLE_BATCH, SINGLE_TOKEN, AUDIO_DIMENSION), generator)
        video_context = _random((SINGLE_BATCH, SINGLE_TOKEN, VIDEO_DIMENSION), generator)
        audio_context = _random((SINGLE_BATCH, SINGLE_TOKEN, AUDIO_DIMENSION), generator)
        video = _args(
            x=video_input,
            context=video_context,
            timesteps=_random((SINGLE_BATCH, SINGLE_TOKEN, ADALN_PARAMETER_COUNT * VIDEO_DIMENSION), generator),
            prompt_timestep=_random(
                (SINGLE_BATCH, SINGLE_TOKEN, PROMPT_ADALN_PARAMETER_COUNT * VIDEO_DIMENSION), generator
            ),
            cross_scale_shift_timestep=_random(
                (SINGLE_BATCH, SINGLE_TOKEN, AV_CROSS_SCALE_PARAMETER_COUNT * VIDEO_DIMENSION), generator
            ),
            cross_gate_timestep=_random(
                (SINGLE_BATCH, SINGLE_TOKEN, AV_CROSS_GATE_PARAMETER_COUNT * VIDEO_DIMENSION), generator
            ),
        )
        audio = _args(
            x=audio_input,
            context=audio_context,
            timesteps=_random((SINGLE_BATCH, SINGLE_TOKEN, ADALN_PARAMETER_COUNT * AUDIO_DIMENSION), generator),
            prompt_timestep=_random(
                (SINGLE_BATCH, SINGLE_TOKEN, PROMPT_ADALN_PARAMETER_COUNT * AUDIO_DIMENSION), generator
            ),
            cross_scale_shift_timestep=_random(
                (SINGLE_BATCH, SINGLE_TOKEN, AV_CROSS_SCALE_PARAMETER_COUNT * AUDIO_DIMENSION), generator
            ),
            cross_gate_timestep=_random(
                (SINGLE_BATCH, SINGLE_TOKEN, AV_CROSS_GATE_PARAMETER_COUNT * AUDIO_DIMENSION), generator
            ),
        )
        video_output, audio_output = block(video, audio)
        assert video_output is not None and audio_output is not None
        arrays = {
            "video_input": _as_numpy(video_input),
            "audio_input": _as_numpy(audio_input),
            "video_context": _as_numpy(video_context),
            "audio_context": _as_numpy(audio_context),
            "video_timesteps": _as_numpy(video.timesteps),
            "audio_timesteps": _as_numpy(audio.timesteps),
            "video_prompt_timestep": _as_numpy(video.prompt_timestep),
            "audio_prompt_timestep": _as_numpy(audio.prompt_timestep),
            "video_cross_scale_shift_timestep": _as_numpy(video.cross_scale_shift_timestep),
            "audio_cross_scale_shift_timestep": _as_numpy(audio.cross_scale_shift_timestep),
            "video_cross_gate_timestep": _as_numpy(video.cross_gate_timestep),
            "audio_cross_gate_timestep": _as_numpy(audio.cross_gate_timestep),
            "video_output": _as_numpy(video_output.x),
            "audio_output": _as_numpy(audio_output.x),
        }
        arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.artifact, **arrays)
        report = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": arguments.transformer_checkpoint.name,
            "block_index": arguments.block_index,
            "seed": arguments.seed,
            "dtype": "BF16",
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha256(array)}
                for name, array in arrays.items()
            },
        }
        temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(arguments.report)
    finally:
        del transformer
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
