#!/usr/bin/env python3
"""Create an explicit, unassigned MLX mapping template from a safetensors header."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lara_ltx.errors import LaraError
from lara_ltx.models import inspect_safetensors

TEMPLATE_VERSION = 2
UNASSIGNED_TRANSFORM = "unassigned"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    checkpoint = inspect_safetensors(arguments.checkpoint)
    template = {
        "source_file": arguments.checkpoint.name,
        "mapping_version": TEMPLATE_VERSION,
        "tensor_count": checkpoint.tensor_count,
        "weights": [
            {
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "source": tensor.name,
                "target": None,
                "transform": UNASSIGNED_TRANSFORM,
            }
            for tensor in sorted(checkpoint.tensors, key=lambda item: item.name)
        ],
    }
    try:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(arguments.output)}) from exc


if __name__ == "__main__":
    main()
