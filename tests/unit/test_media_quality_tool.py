from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "parity" / "compare_media_quality.py"


def _tool():
    spec = importlib.util.spec_from_file_location("compare_media_quality", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report() -> dict[str, object]:
    return {
        "video": {
            "frame_metrics": {"cosine_similarity": 0.9},
            "temporal_delta_metrics": {"cosine_similarity": 0.8},
        },
        "audio": {
            "metrics": {"cosine_similarity": 0.7},
            "alignment": {"lag_ms": 10.0},
        },
    }


def test_media_comparison_without_thresholds_is_diagnostic_only() -> None:
    acceptance, passed = _tool()._threshold_result(_report(), None)

    assert acceptance is None
    assert passed is None


def test_media_comparison_uses_versioned_threshold_file(tmp_path: Path) -> None:
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "minimum_frame_cosine": 0.85,
                "minimum_temporal_cosine": 0.75,
                "minimum_audio_aligned_cosine": 0.65,
                "maximum_audio_lag_ms": 20.0,
            }
        ),
        encoding="utf-8",
    )

    acceptance, passed = _tool()._threshold_result(_report(), thresholds)

    assert passed is True
    assert acceptance is not None
    assert all(acceptance["checks"].values())
