#!/usr/bin/env python3
"""Capture CUDA BF16 Gemma V2 LTX feature-projection boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from lara_ltx.errors import LaraError
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2, norm_and_concat_per_token_rms
from safetensors import safe_open

ARTIFACT_SCHEMA_VERSION = 1
CUDA_DEVICE = "cuda"
BF16_DTYPE = torch.bfloat16
HIDDEN_DIMENSION = 3840
HIDDEN_STATE_COUNT = 49
FLATTENED_DIMENSION = HIDDEN_DIMENSION * HIDDEN_STATE_COUNT
VIDEO_DIMENSION = 4096
AUDIO_DIMENSION = 2048
DEFAULT_SEED = 25_081_900
SEQUENCE_LENGTH = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_linear(
    checkpoint: Path,
    *,
    prefix: str,
    output_dimension: int,
    device: torch.device,
) -> torch.nn.Linear:
    linear = torch.nn.Linear(
        FLATTENED_DIMENSION,
        output_dimension,
        bias=True,
        device="meta",
        dtype=BF16_DTYPE,
    )
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        weight = source.get_tensor(f"{prefix}.weight").to(device=device, dtype=BF16_DTYPE)
        bias = source.get_tensor(f"{prefix}.bias").to(device=device, dtype=BF16_DTYPE)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    linear.bias = torch.nn.Parameter(bias, requires_grad=False)
    return linear


@torch.inference_mode()
def main() -> int:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise LaraError("LARA-RUNTIME-002")
    device = torch.device(CUDA_DEVICE)
    generator = torch.Generator(device=CUDA_DEVICE).manual_seed(arguments.seed)
    hidden_states = torch.randn(
        (1, SEQUENCE_LENGTH, HIDDEN_DIMENSION, HIDDEN_STATE_COUNT),
        generator=generator,
        device=device,
        dtype=BF16_DTYPE,
    )
    attention_mask = torch.tensor([[True, False, True]], device=device)
    video_linear = _load_linear(
        arguments.checkpoint,
        prefix="text_embedding_projection.video_aggregate_embed",
        output_dimension=VIDEO_DIMENSION,
        device=device,
    )
    audio_linear = _load_linear(
        arguments.checkpoint,
        prefix="text_embedding_projection.audio_aggregate_embed",
        output_dimension=AUDIO_DIMENSION,
        device=device,
    )
    extractor = FeatureExtractorV2(
        video_aggregate_embed=video_linear,
        embedding_dim=HIDDEN_DIMENSION,
        audio_aggregate_embed=audio_linear,
    ).eval()
    try:
        flattened = norm_and_concat_per_token_rms(hidden_states, attention_mask)
        video_features, audio_features = extractor(hidden_states, attention_mask)
        if audio_features is None:
            raise LaraError("LARA-MODEL-029", details={"key": "audio_aggregate_embed"})
        arrays = {
            "hidden_states": _as_numpy(hidden_states),
            "attention_mask": attention_mask.cpu().numpy(),
            "rms_flattened": _as_numpy(flattened),
            "video_features": _as_numpy(video_features),
            "audio_features": _as_numpy(audio_features),
        }
        arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.artifact, **arrays)
        report = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": arguments.checkpoint.name,
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
        del extractor, video_linear, audio_linear
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
