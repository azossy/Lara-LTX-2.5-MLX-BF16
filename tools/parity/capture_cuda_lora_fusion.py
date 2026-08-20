#!/usr/bin/env python3
"""Capture an official CUDA BF16 distilled-LoRA fusion reference tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from lara_ltx.errors import LaraError
from lara_ltx.models.lora import DistilledLoraStrengths, build_lora_pairs
from safetensors import safe_open

ARTIFACT_SCHEMA_VERSION = 1
CUDA_DEVICE = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _to_bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy()


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _fuse(base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, strength: float) -> torch.Tensor:
    """Match the upstream BF16 rule: ``(B * strength) @ A + base``."""

    delta = torch.matmul(b * strength, a).to(dtype=torch.bfloat16)
    return delta.add_(base).to(dtype=torch.bfloat16)


@torch.inference_mode()
def main() -> int:
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise LaraError("LARA-RUNTIME-002")
    selected = min(
        build_lora_pairs(arguments.lora_checkpoint, arguments.transformer_checkpoint),
        key=lambda pair: (pair.base_shape[0] * pair.base_shape[1], pair.target_key),
    )
    with safe_open(arguments.transformer_checkpoint, framework="pt", device=CUDA_DEVICE) as transformer_handle:
        base = transformer_handle.get_tensor(selected.target_key)
    with safe_open(arguments.lora_checkpoint, framework="pt", device=CUDA_DEVICE) as lora_handle:
        a = lora_handle.get_tensor(selected.a_key)
        b = lora_handle.get_tensor(selected.b_key)
    strengths = DistilledLoraStrengths()
    fused_stage_1 = _fuse(base, a, b, strengths.stage_1)
    fused_stage_2 = _fuse(base, a, b, strengths.stage_2)
    arrays = {
        "base_bf16_bits": _to_bits(base),
        "a_bf16_bits": _to_bits(a),
        "b_bf16_bits": _to_bits(b),
        "fused_stage_1_bf16_bits": _to_bits(fused_stage_1),
        "fused_stage_2_bf16_bits": _to_bits(fused_stage_2),
    }
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.artifact, **arrays)
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "algorithm": "(B * strength) @ A + base, BF16 accumulation/output",
        "pair": {
            "target_key": selected.target_key,
            "a_key": selected.a_key,
            "b_key": selected.b_key,
            "rank": selected.rank,
            "base_shape": list(selected.base_shape),
        },
        "stage_strengths": {"stage_1": strengths.stage_1, "stage_2": strengths.stage_2},
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha256(array)}
            for name, array in arrays.items()
        },
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
