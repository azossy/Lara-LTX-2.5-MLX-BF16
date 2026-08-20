#!/usr/bin/env python3
"""Generate CUDA BF16 golden vectors for one production DiffVAE deterministic stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.transformer.blocks import NABlock
from ltx_core.model.video_vae.transformer.fallback_na import EagerSdpaAttention
from ltx_core.model.video_vae.transformer.layers import LinearPixelShuffleUpsample

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_911
CHANNELS = 16
DEPTH = 2
HEAD_DIM = 16
KERNEL_SIZE = (3, 3, 3)
UPSAMPLE_STRIDE = (1, 2, 2)
UPSAMPLE_REDUCTION = 2
INPUT_SHAPE = (1, 3, 3, 4, CHANNELS)
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "diffvae_det_stage_weight__"
OUTPUT_STEM = "diffvae_det_stage_bf16_cuda"


class DeterministicStage(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [NABlock(dim=CHANNELS, kernel_size=KERNEL_SIZE, head_dim=HEAD_DIM) for _ in range(DEPTH)]
        )
        for block in self.blocks:
            block.attn.attention_function = EagerSdpaAttention()
        self.upsample = LinearPixelShuffleUpsample(CHANNELS, UPSAMPLE_STRIDE, UPSAMPLE_REDUCTION)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.upsample(x, drop_leading_frame=True)


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    stage = DeterministicStage().to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in stage.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )
    input_tensor = torch.randn(INPUT_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
    output = stage(input_tensor)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        input=_as_numpy(input_tensor),
        output=_as_numpy(output),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in stage.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "depth": DEPTH,
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
