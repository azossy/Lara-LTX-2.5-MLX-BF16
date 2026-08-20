from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_FUSION_PEAK_BYTES = 1_050_624


def test_cuda_lora_fusion_reference_is_complete_and_hash_verified() -> None:
    root = Path("golden")
    report = json.loads((root / "cuda_lora_fusion_reference.json").read_text(encoding="utf-8"))
    arrays = np.load(root / "cuda_lora_fusion_reference.npz")

    assert report["pair"]["base_shape"] == [32, 2048]
    assert report["stage_strengths"] == {"stage_1": 0.25, "stage_2": 0.5}
    for name, metadata in report["arrays"].items():
        value = arrays[name]
        assert list(value.shape) == metadata["shape"]
        assert str(value.dtype) == metadata["dtype"]
        assert hashlib.sha256(value.tobytes()).hexdigest() == metadata["sha256"]


def test_target_mac_lora_fusion_report_passes_both_stage_gates() -> None:
    report = json.loads(Path("golden/mlx_lora_fusion_report.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert set(report["stages"]) == {"stage_1", "stage_2"}
    assert max(stage["max_abs_error"] for stage in report["stages"].values()) <= report["max_absolute_tolerance"]
    assert report["fusion_memory"]["peak_bytes"] == EXPECTED_FUSION_PEAK_BYTES
