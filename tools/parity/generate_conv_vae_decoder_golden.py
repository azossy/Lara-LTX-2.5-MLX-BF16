#!/usr/bin/env python3
"""Generate a deterministic CUDA BF16 golden vector for the Conv VAE decoder subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder
from ltx_core.model.video_vae.enums import NormLayerType

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_904
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
BASE_CHANNELS = 4
PATCH_SIZE = 2
INPUT_SHAPE = (1, INPUT_CHANNELS, 2, 2, 2)
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "conv_vae_decoder_weight__"
OUTPUT_STEM = "conv_vae_decoder_bf16_cuda"
DECODER_BLOCKS = [
    ("res_x_y", {"multiplier": 2}),
    ("compress_all", {"multiplier": 2}),
]


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    decoder = ConvVideoDecoder(
        convolution_dimensions=3,
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        patch_size=PATCH_SIZE,
        decoder_blocks=DECODER_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=True,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )
    input_tensor = torch.randn(INPUT_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
    output = decoder(input_tensor)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        input=_as_numpy(input_tensor),
        output=_as_numpy(output),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in decoder.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "causal": True,
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
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
