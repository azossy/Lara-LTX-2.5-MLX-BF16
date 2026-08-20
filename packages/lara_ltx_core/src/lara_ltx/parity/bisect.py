"""Deterministic first-divergence localization for ordered backend boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lara_ltx.errors import LaraError

from .metrics import TensorMetrics, compare_tensors


@dataclass(frozen=True)
class ParityTolerance:
    max_normalized_rmse: float
    min_cosine_similarity: float
    max_absolute_error: float | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_normalized_rmse)
            or self.max_normalized_rmse < 0
            or not math.isfinite(self.min_cosine_similarity)
            or not -1.0 <= self.min_cosine_similarity <= 1.0
            or (
                self.max_absolute_error is not None
                and (not math.isfinite(self.max_absolute_error) or self.max_absolute_error < 0)
            )
        ):
            raise LaraError("LARA-PARITY-004", details={"reason": "invalid_tolerance"})

    def accepts(self, metrics: TensorMetrics) -> bool:
        if metrics.reference_checksum == metrics.candidate_checksum:
            return True
        absolute_pass = self.max_absolute_error is None or metrics.max_abs_error <= self.max_absolute_error
        return (
            metrics.nan_count == 0
            and metrics.inf_count == 0
            and metrics.normalized_rmse <= self.max_normalized_rmse
            and metrics.cosine_similarity >= self.min_cosine_similarity
            and absolute_pass
        )


@dataclass(frozen=True)
class BoundaryParity:
    key: str
    passed: bool
    tolerance: ParityTolerance
    metrics: TensorMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "passed": self.passed,
            "tolerance": asdict(self.tolerance),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class DivergenceReport:
    boundaries: tuple[BoundaryParity, ...]
    first_divergent_key: str | None

    @property
    def passed(self) -> bool:
        return self.first_divergent_key is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "first_divergent_key": self.first_divergent_key,
            "passed": self.passed,
        }


def find_first_divergence(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    ordered_keys: Sequence[str],
    tolerance: ParityTolerance,
    overrides: Mapping[str, ParityTolerance] | None = None,
) -> DivergenceReport:
    """Compare ordered checkpoints and stop after the first failed boundary."""

    if not ordered_keys or len(set(ordered_keys)) != len(ordered_keys):
        raise LaraError("LARA-PARITY-004", details={"reason": "invalid_ordered_keys"})
    results: list[BoundaryParity] = []
    for key in ordered_keys:
        if key not in reference or key not in candidate:
            raise LaraError("LARA-PARITY-004", details={"reason": f"missing_{key}"})
        selected = (overrides or {}).get(key, tolerance)
        metrics = compare_tensors(key, reference[key], candidate[key])
        passed = selected.accepts(metrics)
        results.append(BoundaryParity(key=key, passed=passed, tolerance=selected, metrics=metrics))
        if not passed:
            return DivergenceReport(tuple(results), first_divergent_key=key)
    return DivergenceReport(tuple(results), first_divergent_key=None)
