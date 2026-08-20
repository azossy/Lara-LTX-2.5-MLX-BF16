#!/usr/bin/env python3
"""Extract the minimum replay tensors from an expanded CUDA sampling trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SDE_PREFIXES = ("stage_1_sde_", "stage_2_sde_")
REFERENCE_KEYS_FIELD = "reference_keys"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--denoiser-call-limit", type=int, default=0)
    parser.add_argument("--only-denoiser", action="store_true")
    return parser.parse_args()


def selected_names(
    archive_names: list[str],
    config: dict[str, object],
    *,
    denoiser_call_limit: int = 0,
    only_denoiser: bool = False,
) -> set[str]:
    references = config.get(REFERENCE_KEYS_FIELD)
    if not isinstance(references, dict) or not all(isinstance(value, str) for value in references.values()):
        raise ValueError("[LARA-RUNTIME-008] Invalid sampling reference-key configuration.")
    selected = set() if only_denoiser else set(references.values())
    if not only_denoiser:
        selected.update(name for name in archive_names if name.startswith(SDE_PREFIXES))
    if denoiser_call_limit < 0:
        raise ValueError("[LARA-RUNTIME-008] Denoiser call limit must be non-negative.")
    if only_denoiser and denoiser_call_limit == 0:
        raise ValueError("[LARA-RUNTIME-008] Denoiser-only extraction requires a positive call limit.")
    for stage in ("stage_1", "stage_2"):
        for call_index in range(denoiser_call_limit):
            prefix = f"{stage}_denoiser_call_{call_index:02d}_"
            selected.update(name for name in archive_names if name.startswith(prefix))
    missing = selected.difference(archive_names)
    if missing:
        raise ValueError(f"[LARA-RUNTIME-008] Missing replay tensor: {sorted(missing)[0]}")
    return selected


def main() -> int:
    arguments = parse_arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    with np.load(arguments.input) as archive:
        names = selected_names(
            list(archive.files),
            config,
            denoiser_call_limit=arguments.denoiser_call_limit,
            only_denoiser=arguments.only_denoiser,
        )
        arrays = {name: archive[name] for name in sorted(names)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
