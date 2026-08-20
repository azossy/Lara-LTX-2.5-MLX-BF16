#!/usr/bin/env python3
"""Print a validated safetensors inventory without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lara_ltx.models import inspect_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--require-dtype")
    arguments = parser.parse_args()
    checkpoint = inspect_safetensors(arguments.checkpoint)
    if arguments.require_dtype:
        checkpoint.require_dtype(arguments.require_dtype)
    report = {
        "file_size": checkpoint.file_size,
        "header_size": checkpoint.header_size,
        "metadata": checkpoint.metadata,
        "path": str(checkpoint.path),
        "tensor_bytes": checkpoint.total_tensor_bytes,
        "tensor_count": checkpoint.tensor_count,
        "tensors": [
            {"dtype": tensor.dtype, "name": tensor.name, "shape": tensor.shape} for tensor in checkpoint.tensors
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
