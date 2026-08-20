#!/usr/bin/env python3
"""Generate deterministic CUDA BF16 golden vectors for VAE depth-to-space upsampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.sampling import DepthToSpaceUpsample

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_902
UPSAMPLE_BATCH_SIZE = 1
UPSAMPLE_INPUT_CHANNELS = 8
UPSAMPLE_INPUT_TIME = 2
UPSAMPLE_INPUT_HEIGHT = 2
UPSAMPLE_INPUT_WIDTH = 3
UPSAMPLE_STRIDE = (2, 2, 2)
UPSAMPLE_OUT_CHANNELS_REDUCTION_FACTOR = 1
UPSAMPLE_RESIDUAL = True
UPSAMPLE_PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "vae_depth_to_space_weight__"
OUTPUT_STEM = "vae_depth_to_space_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    block = DepthToSpaceUpsample(
        dims=3,
        in_channels=UPSAMPLE_INPUT_CHANNELS,
        stride=UPSAMPLE_STRIDE,
        residual=UPSAMPLE_RESIDUAL,
        out_channels_reduction_factor=UPSAMPLE_OUT_CHANNELS_REDUCTION_FACTOR,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * UPSAMPLE_PARAMETER_STANDARD_DEVIATION
            )
    input_tensor = torch.randn(
        (
            UPSAMPLE_BATCH_SIZE,
            UPSAMPLE_INPUT_CHANNELS,
            UPSAMPLE_INPUT_TIME,
            UPSAMPLE_INPUT_HEIGHT,
            UPSAMPLE_INPUT_WIDTH,
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
        "out_channels_reduction_factor": UPSAMPLE_OUT_CHANNELS_REDUCTION_FACTOR,
        "residual": UPSAMPLE_RESIDUAL,
        "seed": seed,
        "stride": list(UPSAMPLE_STRIDE),
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
