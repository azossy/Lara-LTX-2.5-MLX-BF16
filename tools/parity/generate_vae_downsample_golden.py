#!/usr/bin/env python3
"""Generate deterministic CUDA BF16 golden vectors for VAE space-to-depth downsampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.sampling import SpaceToDepthDownsample

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_903
DOWNSAMPLE_BATCH_SIZE = 1
DOWNSAMPLE_INPUT_CHANNELS = 4
DOWNSAMPLE_OUTPUT_CHANNELS = 8
DOWNSAMPLE_INPUT_TIME = 3
DOWNSAMPLE_INPUT_HEIGHT = 4
DOWNSAMPLE_INPUT_WIDTH = 6
DOWNSAMPLE_STRIDE = (2, 2, 2)
DOWNSAMPLE_PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "vae_space_to_depth_weight__"
OUTPUT_STEM = "vae_space_to_depth_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    block = SpaceToDepthDownsample(
        dims=3,
        in_channels=DOWNSAMPLE_INPUT_CHANNELS,
        out_channels=DOWNSAMPLE_OUTPUT_CHANNELS,
        stride=DOWNSAMPLE_STRIDE,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * DOWNSAMPLE_PARAMETER_STANDARD_DEVIATION
            )
    input_tensor = torch.randn(
        (
            DOWNSAMPLE_BATCH_SIZE,
            DOWNSAMPLE_INPUT_CHANNELS,
            DOWNSAMPLE_INPUT_TIME,
            DOWNSAMPLE_INPUT_HEIGHT,
            DOWNSAMPLE_INPUT_WIDTH,
        ),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    output = block(input_tensor, causal=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        input=_as_numpy(input_tensor),
        output=_as_numpy(output),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "causal": True,
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "input_shape": list(input_tensor.shape),
        "output_channels": DOWNSAMPLE_OUTPUT_CHANNELS,
        "seed": seed,
        "stride": list(DOWNSAMPLE_STRIDE),
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    (output_directory / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    generate(arguments.output_directory, arguments.device, arguments.seed)


if __name__ == "__main__":
    main()
