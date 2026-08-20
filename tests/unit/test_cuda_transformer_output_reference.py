from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def test_cuda_transformer_output_reference_is_complete_and_hash_verified() -> None:
    root = Path("golden")
    report = json.loads((root / "cuda_transformer_output_reference.json").read_text(encoding="utf-8"))
    arrays = np.load(root / "cuda_transformer_output_reference.npz")

    assert set(arrays.files) == set(report["arrays"])
    assert report["dtype"] == "BF16"
    for name, metadata in report["arrays"].items():
        value = arrays[name]
        assert list(value.shape) == metadata["shape"]
        assert hashlib.sha256(value.tobytes()).hexdigest() == metadata["sha256"]
