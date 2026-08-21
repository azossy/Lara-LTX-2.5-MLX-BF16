from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.quality.run_clap_alignment import (
    ClapAlignmentError,
    _normalized_diagonal,
    load_alignment_manifest,
)


def _manifest(tmp_path: Path, cases: list[dict[str, str]]) -> Path:
    path = tmp_path / "alignment.json"
    path.write_text(
        json.dumps({"schema_version": 1, "name": "fixture", "cases": cases}),
        encoding="utf-8",
    )
    return path


def test_load_alignment_manifest_validates_audio_and_hashes_manifest(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    manifest = load_alignment_manifest(
        _manifest(tmp_path, [{"id": "sample", "audio_path": str(audio), "prompt": "soft wind"}])
    )
    assert manifest.name == "fixture"
    assert manifest.cases[0].audio_path == audio
    assert len(manifest.sha256) == 64


def test_load_alignment_manifest_rejects_missing_audio(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path,
        [{"id": "missing", "audio_path": str(tmp_path / "missing.wav"), "prompt": "room tone"}],
    )
    with pytest.raises(ClapAlignmentError, match="missing_audio"):
        load_alignment_manifest(path)


def test_normalized_diagonal_returns_matching_pair_cosines() -> None:
    audio = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert _normalized_diagonal(audio, text) == pytest.approx([1.0, 2**-0.5])


def test_normalized_diagonal_rejects_shape_mismatch() -> None:
    with pytest.raises(ClapAlignmentError, match="invalid_embedding_shape"):
        _normalized_diagonal(np.ones((1, 2)), np.ones((2, 2)))
