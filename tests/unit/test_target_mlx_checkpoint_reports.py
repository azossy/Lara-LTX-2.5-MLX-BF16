import json
from pathlib import Path

import pytest

REPORTS = (
    "mlx_checkpoint_block_0_report.json",
    "mlx_gemma_feature_report.json",
    "mlx_transformer_output_report.json",
)


@pytest.mark.parametrize("report_name", REPORTS)
def test_target_mlx_checkpoint_report_passes_frozen_criteria(report_name: str) -> None:
    report = json.loads((Path("golden") / report_name).read_text(encoding="utf-8"))
    criteria = report["acceptance_criteria"]
    streams = report.get("streams") or {name: report[name] for name in ("video", "audio")}

    assert report["passed"] is True
    assert report["weight_load_memory"]["peak_bytes"] > 0
    for metrics in streams.values():
        assert metrics["nan_count"] == 0
        assert metrics["inf_count"] == 0
        assert metrics["normalized_rmse"] <= criteria["max_normalized_rmse"]
        assert metrics["cosine_similarity"] >= criteria["min_cosine_similarity"]
