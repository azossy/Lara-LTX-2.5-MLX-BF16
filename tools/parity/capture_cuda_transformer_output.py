#!/usr/bin/env python3
"""Capture checkpoint-backed CUDA BF16 video/audio transformer output heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from lara_ltx.errors import LaraError
from ltx_pipelines.utils.blocks import DiffusionStage

ARTIFACT_SCHEMA_VERSION = 1
CUDA_DEVICE = "cuda"
VIDEO_HIDDEN_DIMENSION = 4096
AUDIO_HIDDEN_DIMENSION = 2048
BATCH_SIZE = 1
TOKEN_COUNT = 2
DEFAULT_SEED = 25_081_900


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _random(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=CUDA_DEVICE, dtype=torch.bfloat16)


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.inference_mode()
def main() -> int:
    arguments = parse_arguments()
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
        model = transformer.velocity_model
        video_hidden = _random((BATCH_SIZE, TOKEN_COUNT, VIDEO_HIDDEN_DIMENSION), generator)
        video_timestep = _random(video_hidden.shape, generator)
        audio_hidden = _random((BATCH_SIZE, TOKEN_COUNT, AUDIO_HIDDEN_DIMENSION), generator)
        audio_timestep = _random(audio_hidden.shape, generator)
        video_output = model._process_output(
            model.scale_shift_table,
            model.norm_out,
            model.proj_out,
            video_hidden,
            video_timestep,
        )
        audio_output = model._process_output(
            model.audio_scale_shift_table,
            model.audio_norm_out,
            model.audio_proj_out,
            audio_hidden,
            audio_timestep,
        )
        arrays = {
            "video_hidden": _as_numpy(video_hidden),
            "video_timestep": _as_numpy(video_timestep),
            "video_output": _as_numpy(video_output),
            "audio_hidden": _as_numpy(audio_hidden),
            "audio_timestep": _as_numpy(audio_timestep),
            "audio_output": _as_numpy(audio_output),
        }
        arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.artifact, **arrays)
        report = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": arguments.transformer_checkpoint.name,
            "seed": arguments.seed,
            "dtype": "BF16",
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha256(array)}
                for name, array in arrays.items()
            },
        }
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(arguments.report)
    finally:
        del transformer
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
