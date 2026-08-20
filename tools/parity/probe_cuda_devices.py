#!/usr/bin/env python3
"""Prove deterministic BF16 compute on every configured CUDA reference GPU."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

DEFAULT_SEED = 25_081_900
ATTENTION_BATCH = 1
ATTENTION_HEADS = 4
ATTENTION_TOKENS = 128
ATTENTION_HEAD_DIM = 64


def _probe(device_index: int, seed: int) -> dict[str, object]:
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"[LARA-RUNTIME-002] CUDA device {device_index} is unavailable; "
            f"detected {torch.cuda.device_count()} device(s). Start the configured GPU instance and retry."
        )

    device = torch.device(f"cuda:{device_index}")
    generator = torch.Generator(device=device).manual_seed(seed + device_index)
    query = torch.randn(
        (ATTENTION_BATCH, ATTENTION_HEADS, ATTENTION_TOKENS, ATTENTION_HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(query.shape, generator=generator, device=device, dtype=torch.bfloat16)
    value = torch.randn(query.shape, generator=generator, device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(device)
    output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    checksum = output.float().sum()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(
            f"[LARA-RUNTIME-002] CUDA device {device_index} produced non-finite BF16 SDPA output. "
            "Verify the pinned driver, CUDA and PyTorch environment."
        )

    properties = torch.cuda.get_device_properties(device)
    return {
        "index": device_index,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "output_checksum_float32": float(checksum.cpu()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")

    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "dtype": str(torch.bfloat16),
        "seed": arguments.seed,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": [_probe(device, arguments.seed) for device in arguments.devices],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
