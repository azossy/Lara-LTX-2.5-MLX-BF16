from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "parity" / "replay_cuda_sampler_boundaries.py"


def _tool():
    spec = importlib.util.spec_from_file_location("replay_cuda_sampler_boundaries", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expected_denoiser_calls_include_terminal_prediction() -> None:
    tool = _tool()

    assert tool._expected_denoiser_calls(mx.array([1.0, 0.5, 0.0], dtype=mx.float32)) == 5
    assert tool._expected_denoiser_calls(mx.array([1.0, 0.5, 0.25], dtype=mx.float32)) == 4


def test_schedule_uses_versioned_indices() -> None:
    tool = _tool()
    reference = {"sigmas": np.array([1.0, 0.75, 0.5, 0.0], dtype=np.float32)}
    config = {
        "reference_keys": {"stage_one_sigmas": "sigmas"},
        "stage_one_schedule_indices": [0, 2, 3],
    }

    schedule = tool._schedule(reference, config, "stage_one")

    np.testing.assert_array_equal(np.asarray(schedule), [1.0, 0.5, 0.0])


def test_captured_denoiser_rejects_sigma_mismatch() -> None:
    tool = _tool()
    reference = {"stage_1_denoiser_call_00_sigma": np.array(1.0, dtype=np.float32)}
    denoiser = tool.CapturedDenoiser(reference, "stage_1", expected_calls=1)

    with pytest.raises(LaraError, match="sampler_replay_sigma_mismatch"):
        denoiser(None, None, 0.5)


def test_tensor_reports_missing_reference_with_lara_error() -> None:
    tool = _tool()

    with pytest.raises(LaraError, match="missing_sampler_replay_tensor:missing"):
        tool._tensor({}, "missing")


def test_versioned_cuda_output_injected_sampler_report_passes() -> None:
    report = json.loads((PROJECT_ROOT / "golden" / "mlx_sampler_only_replay_report.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["first_failure"] is None
    assert [stage["denoiser_call_count"] for stage in report["stages"]] == [31, 7]
