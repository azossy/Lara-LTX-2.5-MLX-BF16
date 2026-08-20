"""Token-count shifted LTX-2 sigma scheduler without PyTorch."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from lara_ltx.errors import LaraError

BASE_SHIFT_ANCHOR = 1_024
MAX_SHIFT_ANCHOR = 4_096
DEFAULT_MAX_SHIFT = 2.05
DEFAULT_BASE_SHIFT = 0.95
DEFAULT_TERMINAL_SIGMA = 0.1
DEFAULT_TOKEN_COUNT = MAX_SHIFT_ANCHOR


class LTX2Scheduler:
    """Generate the official shifted, optionally terminal-stretched schedule."""

    def execute(
        self,
        *,
        steps: int,
        token_count: int = DEFAULT_TOKEN_COUNT,
        max_shift: float = DEFAULT_MAX_SHIFT,
        base_shift: float = DEFAULT_BASE_SHIFT,
        stretch: bool = True,
        terminal: float = DEFAULT_TERMINAL_SIGMA,
    ) -> mx.array:
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps <= 0
            or (stretch and steps < 2)
            or not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
            or not all(math.isfinite(value) for value in (max_shift, base_shift, terminal))
            or (stretch and not 0.0 < terminal < 1.0)
        ):
            raise LaraError("LARA-SAMPLING-001", details={"reason": "invalid_scheduler_configuration"})
        sigmas = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)
        slope = np.float32((max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR))
        intercept = np.float32(base_shift - slope * BASE_SHIFT_ANCHOR)
        sigma_shift = np.float32(token_count) * slope + intercept
        shift_exponential = np.float32(math.exp(float(sigma_shift)))
        nonzero = sigmas != 0
        shifted = np.zeros_like(sigmas)
        shifted[nonzero] = shift_exponential / (
            shift_exponential + (np.float32(1.0) / sigmas[nonzero] - np.float32(1.0))
        )
        if stretch:
            active = shifted[nonzero]
            one_minus = np.float32(1.0) - active
            scale = one_minus[-1] / np.float32(1.0 - terminal)
            shifted[nonzero] = np.float32(1.0) - one_minus / scale
        return mx.array(shifted, dtype=mx.float32)
