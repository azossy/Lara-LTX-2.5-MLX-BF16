#!/usr/bin/env python3
"""Generate deterministic CUDA BF16 golden vectors for a Diffusion VAE ResNet block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.enums import NormLayerType
from ltx_core.model.video_vae.resnet import ResnetBlock3D

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_901
BLOCK_BATCH_SIZE = 1
BLOCK_INPUT_CHANNELS = 3
BLOCK_OUTPUT_CHANNELS = 4
BLOCK_TIME = 3
BLOCK_HEIGHT = 4
BLOCK_WIDTH = 5
BLOCK_GROUPS = 1
BLOCK_PARAMETER_STANDARD_DEVIATION = 0.02
BLOCK_WEIGHT_PREFIX = "vae_resnet_weight__"
NORM_LAYER_BY_NAME = {
    "group_norm": NormLayerType.GROUP_NORM,
    "pixel_norm": NormLayerType.PIXEL_NORM,
}
OUTPUT_STEM_BY_NORM_LAYER = {
    "group_norm": "vae_resnet_group_bf16_cuda",
    "pixel_norm": "vae_resnet_bf16_cuda",
}


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int, norm_layer_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    norm_layer = NORM_LAYER_BY_NAME[norm_layer_name]
    output_stem = OUTPUT_STEM_BY_NORM_LAYER[norm_layer_name]
    block = ResnetBlock3D(
        dims=3,
        in_channels=BLOCK_INPUT_CHANNELS,
        out_channels=BLOCK_OUTPUT_CHANNELS,
        groups=BLOCK_GROUPS,
        norm_layer=norm_layer,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * BLOCK_PARAMETER_STANDARD_DEVIATION
            )
    input_tensor = torch.randn(
        (BLOCK_BATCH_SIZE, BLOCK_INPUT_CHANNELS, BLOCK_TIME, BLOCK_HEIGHT, BLOCK_WIDTH),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    output = block(input_tensor, causal=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{output_stem}.npz",
        input=_as_numpy(input_tensor),
        output=_as_numpy(output),
        **{f"{BLOCK_WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()},
    )
    metadata = {
        "artifact": f"{output_stem}.npz",
        "causal": True,
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "groups": BLOCK_GROUPS,
        "input_shape": list(input_tensor.shape),
        "norm_layer": norm_layer.value,
        "output_channels": BLOCK_OUTPUT_CHANNELS,
        "seed": seed,
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    (output_directory / "vae_resnet_bf16_cuda.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--norm-layer", choices=tuple(NORM_LAYER_BY_NAME), default="pixel_norm")
    arguments = parser.parse_args()
    generate(arguments.output_directory, arguments.device, arguments.seed, arguments.norm_layer)


if __name__ == "__main__":
    main()
