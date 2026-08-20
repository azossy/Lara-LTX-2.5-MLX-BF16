#!/usr/bin/env python3
"""Generate CUDA BF16 golden vectors for a stage-5 CombinedDiffusionNABlock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.transformer import CombinedDiffusionNABlock, EagerSdpaAttention

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_912
DIM = 16
CONTEXT_CHANNELS = 8
HEAD_DIM = 16
KERNEL_SIZE = (3, 3, 3)
INPUT_SHAPE = (1, 3, 3, 4, CONTEXT_CHANNELS + DIM)
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "diffvae_combined_block_weight__"
OUTPUT_STEM = "diffvae_combined_block_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    block = CombinedDiffusionNABlock(
        dim=DIM,
        kernel_size=KERNEL_SIZE,
        context_channels=CONTEXT_CHANNELS,
        head_dim=HEAD_DIM,
    ).to(device=device, dtype=torch.bfloat16)
    block.attn.attention_function = EagerSdpaAttention()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )
    context_and_x = torch.randn(INPUT_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
    modulation = tuple(
        torch.randn((1, 1, 1, 1, DIM), generator=generator, device=device, dtype=torch.bfloat16) for _ in range(7)
    )
    output = block(context_and_x, modulation)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        context_and_x=_as_numpy(context_and_x),
        modulation=_as_numpy(torch.stack(modulation, dim=1)),
        output=_as_numpy(output),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "device": str(device),
        "dtype": str(torch.bfloat16),
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
