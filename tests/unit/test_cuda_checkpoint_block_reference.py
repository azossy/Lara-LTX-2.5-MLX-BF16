from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from lara_ltx.models import validate_transformer_block_mapping


def test_checkpoint_backed_cuda_block_reference_and_mapping_are_verified() -> None:
    root = Path("golden")
    report = json.loads((root / "cuda_checkpoint_block_0.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "manifests" / "transformer_block_0_mapping.json").read_text(encoding="utf-8"))
    arrays = np.load(root / "cuda_checkpoint_block_0.npz")

    assert report["block_index"] == 0
    assert len(validate_transformer_block_mapping(mapping)) == 84
    assert set(arrays.files) == set(report["arrays"])
    for name, metadata in report["arrays"].items():
        value = arrays[name]
        assert list(value.shape) == metadata["shape"]
        assert hashlib.sha256(value.tobytes()).hexdigest() == metadata["sha256"]
