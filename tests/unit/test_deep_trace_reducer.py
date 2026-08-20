from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "parity" / "reduce_cuda_deep_trace.py"


def _tool():
    spec = importlib.util.spec_from_file_location("reduce_cuda_deep_trace", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "stage_1",
        "selected_block_indices": [0, 47],
        "passes": [
            {"index": 0, "input_source_pass_index": 0},
            {
                "index": 1,
                "input_source_pass_index": 0,
                "input_field_sources": {"context": 1},
            },
        ],
        "first_call": {
            "sigma_key": "stage_1_denoiser_call_00_sigma",
            "video_input_key": "stage_1_denoiser_call_00_video_input",
            "audio_input_key": "stage_1_denoiser_call_00_audio_input",
        },
    }


def _required_names() -> list[str]:
    names = [
        "stage_1_denoiser_call_00_sigma",
        "stage_1_denoiser_call_00_video_input",
        "stage_1_denoiser_call_00_audio_input",
        "stage_1_denoiser_call_00_video_cond",
    ]
    for pass_index in (0, 1):
        for block_index in (0, 47):
            for modality in ("video", "audio"):
                names.append(f"stage_1_deep_block_{block_index:02d}_pass_{pass_index:02d}_{modality}_output")
    for source_index in (0, 1):
        for modality in ("video", "audio"):
            names.append(f"stage_1_deep_block_00_pass_{source_index:02d}_{modality}_input_context")
    return names


def test_reducer_uses_field_level_source_aliases() -> None:
    selected = _tool().selected_names(_required_names(), _config())

    assert "stage_1_deep_block_00_pass_01_video_input_context" in selected
    assert "stage_1_deep_block_00_pass_00_video_input_context" in selected
    assert "stage_1_denoiser_call_00_video_cond" in selected


def test_reducer_rejects_missing_block_output() -> None:
    names = _required_names()
    names.remove("stage_1_deep_block_47_pass_01_audio_output")

    with pytest.raises(ValueError, match="Missing deep replay tensor"):
        _tool().selected_names(names, _config())
