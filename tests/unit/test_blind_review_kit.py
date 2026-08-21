from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.quality.corpus import QualityCorpusError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "quality" / "build_blind_review_kit.py"
CORPUS_PATH = PROJECT_ROOT / "golden" / "manifests" / "quality_corpus.json"
CONFIG_PATH = PROJECT_ROOT / "golden" / "manifests" / "blind_review_config.json"


def _tool():
    spec = importlib.util.spec_from_file_location("build_blind_review_kit", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _media_directories(root: Path) -> tuple[Path, Path]:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    cuda_directory = root / "reference"
    mlx_directory = root / "candidate"
    cuda_directory.mkdir()
    mlx_directory.mkdir()
    for index, case in enumerate(corpus["cases"]):
        case_id = case["id"]
        (cuda_directory / f"{case_id}.mp4").write_bytes(f"reference-{index}".encode())
        (mlx_directory / f"{case_id}.mp4").write_bytes(f"candidate-{index}".encode())
    return cuda_directory, mlx_directory


def test_blind_review_manifest_hides_backend_and_assignment_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LARA_BLIND_REVIEW_SEED", "25082100")
    cuda_directory, mlx_directory = _media_directories(tmp_path)
    tool = _tool()
    manifests = []
    keys = []
    for run_index in range(2):
        output = tmp_path / f"kit_{run_index}"
        key_path = tmp_path / f"key_{run_index}.json"
        manifests.append(
            tool.build_kit(
                config_path=CONFIG_PATH,
                corpus_path=CORPUS_PATH,
                cuda_directory=cuda_directory,
                mlx_directory=mlx_directory,
                output_directory=output,
                key_report=key_path,
            )
        )
        submission = json.loads((output / "review_submission_template.json").read_text(encoding="utf-8"))
        assert "cuda" not in json.dumps(submission).lower()
        assert "mlx" not in json.dumps(submission).lower()
        assert len(submission["reviews"]) == 4
        keys.append(json.loads(key_path.read_text(encoding="utf-8")))

    assert manifests[0] == manifests[1]
    assert keys[0] == keys[1]
    assert "cuda" not in json.dumps(manifests[0]).lower()
    assert "mlx" not in json.dumps(manifests[0]).lower()
    assert all(set(item["candidates"]) == {"A", "B"} for item in manifests[0]["items"])
    assert all(
        {candidate["backend"] for candidate in item["candidates"].values()} == {"cuda", "mlx"}
        for item in keys[0]["items"]
    )


def test_blind_review_rejects_missing_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARA_BLIND_REVIEW_SEED", "25082100")
    cuda_directory, mlx_directory = _media_directories(tmp_path)
    next(cuda_directory.glob("*.mp4")).unlink()

    with pytest.raises(QualityCorpusError, match="missing_blind_review_media"):
        _tool().build_kit(
            config_path=CONFIG_PATH,
            corpus_path=CORPUS_PATH,
            cuda_directory=cuda_directory,
            mlx_directory=mlx_directory,
            output_directory=tmp_path / "kit",
            key_report=tmp_path / "key.json",
        )
