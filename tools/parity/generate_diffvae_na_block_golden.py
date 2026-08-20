#!/usr/bin/env python3
"""Generate a CUDA BF16 golden vector for a production deterministic DiffVAE NABlock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.transformer.blocks import NABlock
from ltx_core.model.video_vae.transformer.fallback_na import EagerSdpaAttention

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_910
DIM = 16
HEAD_DIM = 16
KERNEL_SIZE = (3, 3, 3)
INPUT_SHAPE = (1, 3, 4, 5, DIM)
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "diffvae_na_block_weight__"
OUTPUT_STEM = "diffvae_na_block_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    block = NABlock(dim=DIM, kernel_size=KERNEL_SIZE, head_dim=HEAD_DIM).to(device=device, dtype=torch.bfloat16)
    block.attn.attention_function = EagerSdpaAttention()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )
    input_tensor = torch.randn(INPUT_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
    output = block(input_tensor)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        input=_as_numpy(input_tensor),
        output=_as_numpy(output),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "input_shape": list(input_tensor.shape),
        "kernel_size": list(KERNEL_SIZE),
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
