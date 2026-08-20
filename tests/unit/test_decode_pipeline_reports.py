import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "report_name",
    [
        "mlx_diffusion_decoder_smoke_report.json",
        "mlx_media_mux_smoke_report.json",
        "mlx_decode_pipeline_smoke_report.json",
    ],
)
def test_decode_pipeline_report_passes(report_name: str) -> None:
    report = json.loads((Path("golden") / report_name).read_text(encoding="utf-8"))

    assert report["passed"] is True


def test_checkpoint_decode_pipeline_preserves_media_contract() -> None:
    report = json.loads(Path("golden/mlx_decode_pipeline_smoke_report.json").read_text(encoding="utf-8"))
    streams = {stream["codec_type"]: stream for stream in report["ffprobe"]["streams"]}

    assert report["outputs"]["video_shape"] == [1, 3, 17, 320, 512]
    assert report["outputs"]["audio_shape"] == [1, 2, 33120]
    assert streams["video"]["nb_frames"] == "17"
    assert streams["video"]["avg_frame_rate"] == "24/1"
    assert streams["audio"]["sample_rate"] == "48000"
    assert streams["audio"]["channels"] == 2
