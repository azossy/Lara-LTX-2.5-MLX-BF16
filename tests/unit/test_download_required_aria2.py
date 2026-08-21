from __future__ import annotations

import hashlib
from pathlib import Path

from tools.models.download_required_aria2 import (
    _entry_needs_download,
    _entry_sha256_matches,
    _remaining_download_bytes,
)


def test_entry_needs_download_rejects_matching_sparse_transfer_with_control_file(tmp_path: Path) -> None:
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"complete")
    Path(f"{destination}.aria2").write_bytes(b"incomplete ranges")

    assert _entry_needs_download({"path": destination.name, "size": destination.stat().st_size}, tmp_path)


def test_entry_needs_download_accepts_matching_size_without_control_file(tmp_path: Path) -> None:
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"complete")

    assert not _entry_needs_download(
        {"path": destination.name, "size": destination.stat().st_size},
        tmp_path,
    )


def test_entry_needs_download_rejects_missing_or_wrong_size(tmp_path: Path) -> None:
    entry = {"path": "model.safetensors", "size": 8}
    assert _entry_needs_download(entry, tmp_path)

    (tmp_path / "model.safetensors").write_bytes(b"short")
    assert _entry_needs_download(entry, tmp_path)


def test_entry_sha256_matches_only_the_pinned_digest(tmp_path: Path) -> None:
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"verified payload")
    expected = hashlib.sha256(destination.read_bytes()).hexdigest()

    assert _entry_sha256_matches({"sha256": expected}, destination, chunk_bytes=4)
    assert not _entry_sha256_matches({"sha256": "0" * 64}, destination, chunk_bytes=4)


def test_remaining_download_bytes_is_zero_for_complete_files(tmp_path: Path) -> None:
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"complete")

    assert _remaining_download_bytes(
        [{"path": destination.name, "size": destination.stat().st_size}],
        tmp_path,
    ) == 0


def test_remaining_download_bytes_counts_missing_files(tmp_path: Path) -> None:
    assert _remaining_download_bytes([{"path": "missing.safetensors", "size": 1024}], tmp_path) == 1024


def test_remaining_download_bytes_uses_allocated_ranges_for_aria2_resume(tmp_path: Path) -> None:
    destination = tmp_path / "sparse.safetensors"
    expected_size = 16 * 1024 * 1024
    with destination.open("wb") as handle:
        handle.seek(expected_size - 1)
        handle.write(b"\0")
    Path(f"{destination}.aria2").write_bytes(b"incomplete ranges")

    allocated_bytes = min(expected_size, destination.stat().st_blocks * 512)
    assert _remaining_download_bytes(
        [{"path": destination.name, "size": expected_size}],
        tmp_path,
    ) == expected_size - allocated_bytes
