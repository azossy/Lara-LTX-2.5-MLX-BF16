from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.quality.corpus import QualityCorpusError, load_corpus, validation_report
from tools.quality.run_mlx_corpus import _generation_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "golden" / "manifests" / "quality_corpus.json"


def test_release_quality_corpus_has_required_diversity() -> None:
    report = validation_report(load_corpus(CORPUS_PATH))

    assert report["passed"] is True
    assert report["scope"] == "structure_only_not_quality_acceptance"
    assert report["case_count"] == 4
    assert report["unique_resolutions"] == 4


def test_quality_corpus_rejects_duplicate_cases(tmp_path: Path) -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    payload["cases"].append(payload["cases"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QualityCorpusError, match="duplicate_case_id"):
        load_corpus(path)


def test_quality_corpus_rejects_unsupported_temporal_grid(tmp_path: Path) -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["num_frames"] = 18
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QualityCorpusError, match="invalid_case_values"):
        load_corpus(path)


def test_generation_report_preserves_partial_progress() -> None:
    corpus = load_corpus(CORPUS_PATH)
    report = _generation_report(
        corpus=corpus,
        model="local-model",
        results=[{"case_id": corpus.cases[0].case_id}],
        requested_case_count=2,
    )

    assert report["status"] == "partial"
    assert report["passed"] is False
    assert report["completed_case_count"] == 1
    assert report["total_case_count"] == 4
