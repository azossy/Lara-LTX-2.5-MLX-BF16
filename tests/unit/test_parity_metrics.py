import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors


def test_identical_tensors_have_perfect_metrics() -> None:
    reference = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)

    metrics = compare_tensors("block.0.output", reference, reference.copy())

    assert metrics.max_abs_error == 0.0
    assert metrics.rmse == 0.0
    assert metrics.normalized_rmse == 0.0
    assert metrics.cosine_similarity == pytest.approx(1.0)
    assert metrics.reference_checksum == metrics.candidate_checksum


def test_shape_mismatch_has_stable_error_code() -> None:
    with pytest.raises(LaraError) as raised:
        compare_tensors("bad-layout", np.zeros((2, 2)), np.zeros((4,)))

    assert raised.value.code == "LARA-PARITY-001"


def test_normalized_rmse_tracks_reference_scale() -> None:
    reference = np.array([10.0, -10.0], dtype=np.float32)
    candidate = np.array([10.1, -9.9], dtype=np.float32)

    metrics = compare_tensors("scaled", reference, candidate)

    assert metrics.reference_rms == pytest.approx(10.0)
    assert metrics.normalized_rmse == pytest.approx(0.01, abs=1e-6)
