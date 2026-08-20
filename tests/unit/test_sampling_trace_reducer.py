from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "parity" / "reduce_cuda_sampling_trace.py"


def _tool():
    spec = importlib.util.spec_from_file_location("reduce_cuda_sampling_trace", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reducer_keeps_configured_boundaries_and_all_sde_noise() -> None:
    selected = _tool().selected_names(
        ["initial", "stage_1_sde_step_00_video", "stage_2_sde_substep_00_audio", "diagnostic"],
        {"reference_keys": {"initial": "initial"}},
    )

    assert selected == {"initial", "stage_1_sde_step_00_video", "stage_2_sde_substep_00_audio"}


def test_reducer_rejects_missing_configured_boundary() -> None:
    with pytest.raises(ValueError, match="Missing replay tensor"):
        _tool().selected_names(["present"], {"reference_keys": {"required": "missing"}})


def test_reducer_can_include_bounded_denoiser_calls() -> None:
    selected = _tool().selected_names(
        [
            "initial",
            "stage_1_denoiser_call_00_video_input",
            "stage_1_denoiser_call_01_video_input",
            "stage_2_denoiser_call_00_audio_output",
        ],
        {"reference_keys": {"initial": "initial"}},
        denoiser_call_limit=1,
    )

    assert "stage_1_denoiser_call_00_video_input" in selected
    assert "stage_2_denoiser_call_00_audio_output" in selected
    assert "stage_1_denoiser_call_01_video_input" not in selected


def test_reducer_can_emit_denoiser_only_transport_delta() -> None:
    selected = _tool().selected_names(
        ["initial", "stage_1_sde_step_00_video", "stage_1_denoiser_call_00_video_output"],
        {"reference_keys": {"initial": "initial"}},
        denoiser_call_limit=1,
        only_denoiser=True,
    )

    assert selected == {"stage_1_denoiser_call_00_video_output"}
