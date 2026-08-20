import json
from pathlib import Path


def test_diffusion_decoder_load_report_passes() -> None:
    report = json.loads(Path("golden/mlx_diffusion_decoder_load_report.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["mapped_target_count"] == 407
    assert report["loaded_parameter_count"] == 407
    assert report["dtype_counts"] == {"mlx.core.bfloat16": 407}
    assert report["weight_load_memory"]["peak_bytes"] > 0
