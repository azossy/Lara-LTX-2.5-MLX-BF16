#!/usr/bin/env python3
"""Generate CUDA BF16 golden vectors for shared production Diffusion VAE layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.transformer.layers import AdaLNZero, LinearPixelShuffleUpsample
from ltx_core.model.video_vae.transformer.swiglu import SwiGLU

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_909
DIM = 8
HIDDEN_DIM = 32
TIMESTEP_EMBEDDING_DIM = 16
UPSAMPLE_STRIDE = (2, 2, 2)
UPSAMPLE_REDUCTION = 2
PARAMETER_STANDARD_DEVIATION = 0.02
OUTPUT_STEM = "diffvae_layers_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _randomize(module: torch.nn.Module, generator: torch.Generator, device: torch.device) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    upsample = LinearPixelShuffleUpsample(DIM, UPSAMPLE_STRIDE, UPSAMPLE_REDUCTION).to(
        device=device, dtype=torch.bfloat16
    )
    adaln = AdaLNZero(DIM, TIMESTEP_EMBEDDING_DIM).to(device=device, dtype=torch.bfloat16)
    swiglu = SwiGLU(DIM, HIDDEN_DIM).to(device=device, dtype=torch.bfloat16)
    for module in (upsample, adaln, swiglu):
        _randomize(module, generator, device)

    upsample_input = torch.randn((1, 2, 3, 4, DIM), generator=generator, device=device, dtype=torch.bfloat16)
    timestep_input = torch.randn((2, TIMESTEP_EMBEDDING_DIM), generator=generator, device=device, dtype=torch.bfloat16)
    swiglu_input = torch.randn((2, 5, DIM), generator=generator, device=device, dtype=torch.bfloat16)
    upsample_output = upsample(upsample_input, drop_leading_frame=True)
    adaln_output = torch.stack(adaln(timestep_input), dim=1)
    swiglu_output = swiglu(swiglu_input)

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        upsample_input=_as_numpy(upsample_input),
        upsample_output=_as_numpy(upsample_output),
        timestep_input=_as_numpy(timestep_input),
        adaln_output=_as_numpy(adaln_output),
        swiglu_input=_as_numpy(swiglu_input),
        swiglu_output=_as_numpy(swiglu_output),
        **{f"upsample_weight__{name}": _as_numpy(value) for name, value in upsample.state_dict().items()},
        **{f"adaln_weight__{name}": _as_numpy(value) for name, value in adaln.state_dict().items()},
        **{f"swiglu_weight__{name}": _as_numpy(value) for name, value in swiglu.state_dict().items()},
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
