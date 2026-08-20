#!/usr/bin/env python3
"""Generate deterministic CUDA BF16 golden vectors for the PixArt timestep embedder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_907
EMBEDDING_DIM = 16
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "timestep_embedding_weight__"
OUTPUT_STEM = "timestep_embedding_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
        embedding_dim=EMBEDDING_DIM,
        size_emb_dim=0,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in embedder.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )
    timestep = torch.tensor([0.05, 1.0], device=device, dtype=torch.bfloat16)
    output = embedder(timestep=timestep, hidden_dtype=torch.bfloat16)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        timestep=_as_numpy(timestep),
        output=_as_numpy(output),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in embedder.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "embedding_dim": EMBEDDING_DIM,
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
