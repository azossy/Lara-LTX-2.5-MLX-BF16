"""Header-level contracts for stage-local distilled-LoRA application."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors

LORA_MANIFEST_SCHEMA_VERSION = 1
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
BASE_TRANSFORMER_SOURCE_PREFIX = "model."
BASE_TRANSFORMER_ALLOWED_DTYPES = frozenset({BF16_DTYPE, "F32"})
DEFAULT_DISTILLED_LORA_STRENGTH_STAGE_1 = 0.25
DEFAULT_DISTILLED_LORA_STRENGTH_STAGE_2 = 0.5


@dataclass(frozen=True)
class LoraPair:
    """One rank-decomposed update and the exact base weight it modifies."""

    target_key: str
    a_key: str
    b_key: str
    rank: int
    base_shape: tuple[int, int]


@dataclass(frozen=True)
class DistilledLoraStrengths:
    """The two official HQ-stage strengths, independently configurable by callers."""

    stage_1: float = DEFAULT_DISTILLED_LORA_STRENGTH_STAGE_1
    stage_2: float = DEFAULT_DISTILLED_LORA_STRENGTH_STAGE_2

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in (self.stage_1, self.stage_2)):
            raise LaraError("LARA-MODEL-027", details={"value": f"{self.stage_1},{self.stage_2}"})


def build_lora_pairs(lora_shard_path: Path, transformer_shard_path: Path) -> tuple[LoraPair, ...]:
    """Validate every LoRA A/B pair against the pinned base transformer header.

    This is intentionally header-only: validation runs before materializing the
    22B base model and prevents a malformed adapter from entering either stage.
    """

    lora_shard = inspect_safetensors(lora_shard_path)
    transformer_shard = inspect_safetensors(transformer_shard_path)
    lora_shard.require_dtype(BF16_DTYPE)
    unsupported_base = next(
        (
            descriptor
            for descriptor in transformer_shard.tensors
            if descriptor.dtype not in BASE_TRANSFORMER_ALLOWED_DTYPES
        ),
        None,
    )
    if unsupported_base is not None:
        raise LaraError("LARA-MODEL-004", details={"name": unsupported_base.name, "dtype": unsupported_base.dtype})
    descriptors = {descriptor.name: descriptor for descriptor in lora_shard.tensors}
    base_descriptors = {descriptor.name: descriptor for descriptor in transformer_shard.tensors}
    pairs: list[LoraPair] = []

    for a_key in sorted(name for name in descriptors if name.endswith(LORA_A_SUFFIX)):
        b_key = f"{a_key[: -len(LORA_A_SUFFIX)]}{LORA_B_SUFFIX}"
        a_descriptor = descriptors[a_key]
        b_descriptor = descriptors.get(b_key)
        target_key = _target_key(a_key)
        base_descriptor = base_descriptors.get(target_key)
        if b_descriptor is None or base_descriptor is None or base_descriptor.dtype != BF16_DTYPE:
            raise LaraError("LARA-MODEL-026", details={"key": a_key})
        if len(a_descriptor.shape) != 2 or len(b_descriptor.shape) != 2:
            raise LaraError("LARA-MODEL-026", details={"key": a_key})
        rank, input_dim = a_descriptor.shape
        output_dim, paired_rank = b_descriptor.shape
        if rank != paired_rank or base_descriptor.shape != (output_dim, input_dim):
            raise LaraError("LARA-MODEL-026", details={"key": a_key})
        pairs.append(
            LoraPair(
                target_key=target_key,
                a_key=a_key,
                b_key=b_key,
                rank=rank,
                base_shape=(output_dim, input_dim),
            )
        )

    b_without_a = [name for name in descriptors if name.endswith(LORA_B_SUFFIX) and _a_key(name) not in descriptors]
    if b_without_a or not pairs:
        raise LaraError("LARA-MODEL-026", details={"key": (b_without_a or [lora_shard.path.name])[0]})
    return tuple(pairs)


def build_lora_manifest(lora_shard_path: Path, transformer_shard_path: Path) -> dict[str, object]:
    """Create the review artifact for each official distilled-LoRA pair."""

    pairs = build_lora_pairs(lora_shard_path, transformer_shard_path)
    return {
        "schema_version": LORA_MANIFEST_SCHEMA_VERSION,
        "component": "distilled_lora",
        "lora_file": lora_shard_path.name,
        "base_transformer_file": transformer_shard_path.name,
        "pairs": [asdict(pair) for pair in pairs],
    }


def write_lora_manifest(output: Path, lora_shard_path: Path, transformer_shard_path: Path) -> None:
    """Atomically write a validated LoRA-to-base-weight manifest."""

    manifest = build_lora_manifest(lora_shard_path, transformer_shard_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def _target_key(a_key: str) -> str:
    return f"{BASE_TRANSFORMER_SOURCE_PREFIX}{a_key[: -len(LORA_A_SUFFIX)]}.weight"


def _a_key(b_key: str) -> str:
    return f"{b_key[: -len(LORA_B_SUFFIX)]}{LORA_A_SUFFIX}"
