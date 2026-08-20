#!/usr/bin/env python3
"""Generate CUDA BF16 golden vectors for a timestep-conditioned VAE mid-block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.enums import NormLayerType
from ltx_core.model.video_vae.resnet import UNetMidBlock3D

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_908
CHANNELS = 4
NUM_LAYERS = 2
INPUT_SHAPE = (1, CHANNELS, 2, 3, 3)
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "vae_mid_block_weight__"
OUTPUT_STEM = "vae_mid_block_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    block = UNetMidBlock3D(
        dims=3,
        in_channels=CHANNELS,
        num_layers=NUM_LAYERS,
        resnet_groups=1,
        norm_layer=NormLayerType.PIXEL_NORM,
        timestep_conditioning=True,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )
    input_tensor = torch.randn(INPUT_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([0.05], device=device, dtype=torch.bfloat16)
    output = block(input_tensor, causal=True, timestep=timestep)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        input=_as_numpy(input_tensor),
        timestep=_as_numpy(timestep),
        output=_as_numpy(output),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "num_layers": NUM_LAYERS,
        "seed": seed,
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
