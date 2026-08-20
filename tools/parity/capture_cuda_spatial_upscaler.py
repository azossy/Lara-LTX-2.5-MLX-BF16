#!/usr/bin/env python3
"""Capture official CUDA x2 latent spatial-upscaler boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from lara_ltx.errors import LaraError
from ltx_pipelines.utils.blocks import VideoUpsampler

ARTIFACT_SCHEMA_VERSION = 1
CUDA_DEVICE = "cuda"
LATENT_DTYPE = torch.bfloat16
LATENT_CHANNELS = 128
DEFAULT_LATENT_KEY = "upsampler_input_video_latent"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-vae-checkpoint", required=True, type=Path)
    parser.add_argument("--upscaler-checkpoint", required=True, type=Path)
    parser.add_argument("--latent-artifact", required=True, type=Path)
    parser.add_argument("--latent-key", default=DEFAULT_LATENT_KEY)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_latent(artifact_path: Path, key: str) -> np.ndarray:
    try:
        with np.load(artifact_path) as artifact:
            if key not in artifact.files:
                raise LaraError("LARA-PARITY-002", details={"key": key})
            latent = artifact[key]
    except OSError as exc:
        raise LaraError("LARA-PARITY-002", details={"key": str(artifact_path)}) from exc
    if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
        raise LaraError("LARA-PARITY-002", details={"key": f"{key}:{latent.shape}"})
    return latent


@torch.inference_mode()
def main() -> int:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise LaraError("LARA-RUNTIME-002")
    device = torch.device(CUDA_DEVICE)
    latent = torch.from_numpy(_load_latent(arguments.latent_artifact, arguments.latent_key)).to(
        device=device,
        dtype=LATENT_DTYPE,
    )
    owner = VideoUpsampler(
        str(arguments.video_vae_checkpoint),
        str(arguments.upscaler_checkpoint),
        LATENT_DTYPE,
        device,
    )
    encoder = owner._encoder_builder.build(device=device, dtype=LATENT_DTYPE).eval()
    upsampler = owner._upsampler_builder.build(device=device, dtype=LATENT_DTYPE).eval()
    try:
        unnormalized = encoder.per_channel_statistics.un_normalize(latent)
        upscaled_unnormalized = upsampler(unnormalized)
        upscaled_normalized = encoder.per_channel_statistics.normalize(upscaled_unnormalized)
        arrays = {
            "normalized_input_latent": _as_numpy(latent),
            "unnormalized_input_latent": _as_numpy(unnormalized),
            "upscaled_unnormalized_latent": _as_numpy(upscaled_unnormalized),
            "upscaled_normalized_latent": _as_numpy(upscaled_normalized),
        }
        arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.artifact, **arrays)
        report = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "video_vae_checkpoint": arguments.video_vae_checkpoint.name,
            "upscaler_checkpoint": arguments.upscaler_checkpoint.name,
            "latent_artifact": arguments.latent_artifact.name,
            "latent_key": arguments.latent_key,
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
        del encoder, upsampler, owner
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
