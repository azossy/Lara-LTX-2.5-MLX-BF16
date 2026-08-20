from __future__ import annotations

import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.parity import ParityTolerance, find_first_divergence


def test_first_divergence_stops_at_earliest_failed_boundary() -> None:
    reference = {
        "input": np.ones((2, 2), dtype=np.float32),
        "block_0": np.ones((2, 2), dtype=np.float32),
        "block_1": np.ones((2, 2), dtype=np.float32),
    }
    candidate = {
        "input": reference["input"].copy(),
        "block_0": reference["block_0"] + np.float32(0.5),
        "block_1": reference["block_1"] + np.float32(1.0),
    }

    result = find_first_divergence(
        reference,
        candidate,
        ordered_keys=("input", "block_0", "block_1"),
        tolerance=ParityTolerance(max_normalized_rmse=0.02, min_cosine_similarity=0.9999),
    )

    assert result.first_divergent_key == "block_0"
    assert [boundary.key for boundary in result.boundaries] == ["input", "block_0"]
    assert not result.passed


def test_first_divergence_accepts_exact_ordered_boundaries() -> None:
    values = {"a": np.arange(4, dtype=np.float32), "b": np.ones((2,), dtype=np.float32)}

    result = find_first_divergence(
        values,
        values,
        ordered_keys=("a", "b"),
        tolerance=ParityTolerance(max_normalized_rmse=0.0, min_cosine_similarity=1.0),
    )

    assert result.passed
    assert result.first_divergent_key is None


def test_first_divergence_rejects_missing_boundary() -> None:
    with pytest.raises(LaraError, match="LARA-PARITY-004"):
        find_first_divergence(
            {"a": np.ones((1,))},
            {},
            ordered_keys=("a",),
            tolerance=ParityTolerance(max_normalized_rmse=0.0, min_cosine_similarity=1.0),
        )
