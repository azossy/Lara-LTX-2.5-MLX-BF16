from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.quality.run_videoscore2 import VideoScoreError, load_manifest, parse_hard_scores


def _write_manifest(path: Path, video: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "paired-demo",
                "cases": [{"id": "cuda", "video_path": str(video), "prompt": "A dancer."}],
            }
        ),
        encoding="utf-8",
    )


def test_load_manifest_accepts_existing_video(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, video)

    manifest = load_manifest(manifest_path)

    assert manifest.name == "paired-demo"
    assert manifest.cases[0].case_id == "cuda"


def test_load_manifest_rejects_missing_video(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, tmp_path / "missing.mp4")

    with pytest.raises(VideoScoreError, match="missing_video"):
        load_manifest(manifest_path)


def test_parse_hard_scores_reads_official_format() -> None:
    output = (
        "visual quality: 4;\n"
        "text-to-video alignment: 5,\n"
        "physical/common-sense consistency: 3"
    )

    assert parse_hard_scores(output) == (4, 5, 3)


def test_parse_hard_scores_reads_numbered_descriptive_format_after_think_block() -> None:
    output = (
        "<think>Visual Quality Assessment: a score of 1 was considered.</think>\n"
        "(1) visual quality \N{EN DASH} clarity, smoothness, artifacts: 4\n"
        "(2) text-to-video alignment \N{EN DASH} fidelity to the prompt: 5\n"
        "(3) physical/common-sense consistency \N{EN DASH} naturalness and physics plausibility: 4\n"
    )

    assert parse_hard_scores(output) == (4, 5, 4)


def test_parse_hard_scores_rejects_missing_or_out_of_range_values() -> None:
    assert parse_hard_scores("no scores") == (None, None, None)
    assert parse_hard_scores(
        "visual quality: 7; text-to-video alignment: 5, physical/common-sense consistency: 3"
    ) == (None, None, None)
