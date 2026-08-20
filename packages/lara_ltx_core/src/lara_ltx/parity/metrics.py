"""Backend-neutral tensor comparison metrics."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lara_ltx.errors import LaraError

STATISTICS_DTYPE = np.float64
RELATIVE_ERROR_FLOOR = np.finfo(np.float32).tiny


@dataclass(frozen=True)
class TensorMetrics:
    name: str
    shape: tuple[int, ...]
    reference_dtype: str
    candidate_dtype: str
    reference_checksum: str
    candidate_checksum: str
    finite_count: int
    nan_count: int
    inf_count: int
    max_abs_error: float
    mean_abs_error: float
    max_relative_error: float
    rmse: float
    reference_rms: float
    normalized_rmse: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _checksum(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def compare_tensors(name: str, reference: np.ndarray, candidate: np.ndarray) -> TensorMetrics:
    """Compare equal-layout tensors without accepting a tolerance implicitly."""

    if reference.shape != candidate.shape:
        raise LaraError(
            "LARA-PARITY-001",
            details={"reference_shape": reference.shape, "candidate_shape": candidate.shape},
        )

    reference_stats = reference.astype(STATISTICS_DTYPE, copy=False)
    candidate_stats = candidate.astype(STATISTICS_DTYPE, copy=False)
    difference = candidate_stats - reference_stats
    absolute_error = np.abs(difference)
    relative_error = absolute_error / np.maximum(np.abs(reference_stats), RELATIVE_ERROR_FLOOR)
    candidate_finite = np.isfinite(candidate_stats)
    reference_norm = float(np.linalg.norm(reference_stats.ravel()))
    candidate_norm = float(np.linalg.norm(candidate_stats.ravel()))
    norm_product = reference_norm * candidate_norm
    cosine = float(np.dot(reference_stats.ravel(), candidate_stats.ravel()) / norm_product) if norm_product else 1.0
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(reference_stats))))
    normalized_rmse = rmse / reference_rms if reference_rms else (0.0 if rmse == 0.0 else float("inf"))

    return TensorMetrics(
        name=name,
        shape=reference.shape,
        reference_dtype=str(reference.dtype),
        candidate_dtype=str(candidate.dtype),
        reference_checksum=_checksum(reference),
        candidate_checksum=_checksum(candidate),
        finite_count=int(candidate_finite.sum()),
        nan_count=int(np.isnan(candidate_stats).sum()),
        inf_count=int(np.isinf(candidate_stats).sum()),
        max_abs_error=float(absolute_error.max(initial=0.0)),
        mean_abs_error=float(absolute_error.mean()),
        max_relative_error=float(relative_error.max(initial=0.0)),
        rmse=rmse,
        reference_rms=reference_rms,
        normalized_rmse=normalized_rmse,
        cosine_similarity=cosine,
    )
