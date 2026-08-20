#!/usr/bin/env python3
"""Capture official CUDA Audio VAE/vocoder boundaries from a fixed latent."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lara_ltx.errors import LaraError
from ltx_pipelines.utils.blocks import AudioDecoder

ARTIFACT_SCHEMA_VERSION = 1
CUDA_DEVICE = "cuda"
LATENT_DTYPE = torch.bfloat16
DEFAULT_LATENT_KEY = "audio_decoder_input_latent"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Official Audio VAE/vocoder BF16 checkpoint")
    parser.add_argument(
        "--latent-artifact",
        required=True,
        type=Path,
        help="NPZ artifact containing the fixed audio latent",
    )
    parser.add_argument("--latent-key", default=DEFAULT_LATENT_KEY, help="Audio latent array key in --latent-artifact")
    parser.add_argument("--artifact", required=True, type=Path, help="Output CUDA boundary NPZ")
    parser.add_argument("--report", required=True, type=Path, help="Output CUDA boundary JSON report")
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
    if latent.ndim != 4 or latent.shape[1] != 8:
        raise LaraError("LARA-PARITY-002", details={"key": f"{key}:{latent.shape}"})
    return latent


@torch.inference_mode()
def main() -> int:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise LaraError("LARA-RUNTIME-002")
    latent_array = _load_latent(arguments.latent_artifact, arguments.latent_key)
    device = torch.device(CUDA_DEVICE)
    latent = torch.from_numpy(latent_array).to(device=device, dtype=LATENT_DTYPE)
    owner = AudioDecoder(str(arguments.checkpoint), LATENT_DTYPE, device)
    decoder = owner._decoder_builder.build(device=device, dtype=LATENT_DTYPE).eval()
    vocoder = owner._vocoder_builder.build(device=device, dtype=LATENT_DTYPE).eval()
    captured: dict[str, np.ndarray] = {"audio_vae_input_latent": _as_numpy(latent)}

    def primary_hook(_: torch.nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["primary_vocoder_input"] = _as_numpy(inputs[0])
        captured["primary_vocoder_waveform"] = _as_numpy(output)

    def mel_hook(_: torch.nn.Module, inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> None:
        captured["bwe_mel_input_waveform"] = _as_numpy(inputs[0])
        captured["bwe_mel_spectrogram"] = _as_numpy(output[0])

    def bwe_hook(_: torch.nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["bwe_generator_input"] = _as_numpy(inputs[0])
        captured["bwe_residual_waveform"] = _as_numpy(output)

    hooks = [
        vocoder.vocoder.register_forward_hook(primary_hook),
        vocoder.mel_stft.register_forward_hook(mel_hook),
        vocoder.bwe_generator.register_forward_hook(bwe_hook),
    ]
    try:
        decoded_spectrogram = decoder(latent)
        captured["audio_vae_decoded_spectrogram"] = _as_numpy(decoded_spectrogram)
        final_waveform = vocoder(decoded_spectrogram)
        captured["vocoder_bwe_waveform"] = _as_numpy(final_waveform)
        arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.artifact, **captured)
        report = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": arguments.checkpoint.name,
            "latent_artifact": arguments.latent_artifact.name,
            "latent_key": arguments.latent_key,
            "decoder_dtype": "BF16",
            "vocoder_execution": "upstream_fp32_accumulation",
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha256(array)}
                for name, array in captured.items()
            },
        }
        temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(arguments.report)
    finally:
        for hook in hooks:
            hook.remove()
        del decoder, vocoder, owner
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
