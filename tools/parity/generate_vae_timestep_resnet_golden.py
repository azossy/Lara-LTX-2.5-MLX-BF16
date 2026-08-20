#!/usr/bin/env python3
"""Generate CUDA BF16 golden vectors for a timestep-conditioned VAE ResNet block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.enums import NormLayerType
from ltx_core.model.video_vae.resnet import ResnetBlock3D

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_906
CHANNELS = 4
INPUT_SHAPE = (1, CHANNELS, 2, 3, 4)
TIMESTEP_SHAPE = (1, CHANNELS * 4, 1, 1, 1)
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "vae_timestep_resnet_weight__"
OUTPUT_STEM = "vae_timestep_resnet_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    block = ResnetBlock3D(
        dims=3,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        groups=1,
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
    timestep = torch.randn(TIMESTEP_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
    output = block(input_tensor, causal=True, timestep=timestep)
    normalized1 = block.norm1(input_tensor)
    ada_values = block.scale_shift_table[None, ..., None, None, None] + timestep.reshape(1, 4, CHANNELS, 1, 1, 1)
    shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)
    modulated1 = normalized1 * (1 + scale1) + shift1
    convolved1 = block.conv1(block.non_linearity(modulated1), causal=True)
    normalized2 = block.norm2(convolved1)
    modulated2 = normalized2 * (1 + scale2) + shift2
    convolved2 = block.conv2(block.non_linearity(modulated2), causal=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        input=_as_numpy(input_tensor),
        timestep=_as_numpy(timestep),
        output=_as_numpy(output),
        normalized1=_as_numpy(normalized1),
        modulated1=_as_numpy(modulated1),
        convolved1=_as_numpy(convolved1),
        normalized2=_as_numpy(normalized2),
        modulated2=_as_numpy(modulated2),
        convolved2=_as_numpy(convolved2),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "input_shape": list(input_tensor.shape),
        "seed": seed,
        "timestep_shape": list(timestep.shape),
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
