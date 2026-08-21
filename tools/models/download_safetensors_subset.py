#!/usr/bin/env python3
"""Download selected safetensors keys as a compact, loadable checkpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import struct
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from huggingface_hub import get_token

AUTH_ERROR_CODE = "LARA-MODEL-001"
VERIFY_ERROR_CODE = "LARA-MODEL-002"
DEFAULT_CONNECTION_COUNT = 16
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30
DEFAULT_LOW_SPEED_SECONDS = 60
DEFAULT_LOW_SPEED_BYTES_PER_SECOND = 1024
DEFAULT_RANGE_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_RANGE_RETRIES = 8
COPY_BUFFER_BYTES = 8 * 1024 * 1024
HEADER_LENGTH_BYTES = 8
HEADER_ALIGNMENT_BYTES = 8
DOWNLOAD_LOCK_FILENAME = ".download.lock"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--key-prefix", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--header-cache", type=Path)
    parser.add_argument("--curl", default="curl")
    parser.add_argument("--connection-count", type=int, default=DEFAULT_CONNECTION_COUNT)
    return parser.parse_args()


def _fail(code: str, cause: str, action: str) -> int:
    print(json.dumps({"ok": False, "error_code": code, "cause": cause, "action": action}))
    return 2


def _download_url(endpoint: str, *, repository_id: str, revision: str, relative_path: str) -> str:
    return "/".join(
        (
            endpoint.rstrip("/"),
            quote(repository_id, safe="/"),
            "resolve",
            quote(revision, safe=""),
            quote(relative_path, safe="/"),
        )
    )


def _read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        encoded_length = handle.read(HEADER_LENGTH_BYTES)
        if len(encoded_length) != HEADER_LENGTH_BYTES:
            raise ValueError("truncated_header_length")
        header_length = struct.unpack("<Q", encoded_length)[0]
        encoded_header = handle.read(header_length)
    if len(encoded_header) != header_length:
        raise ValueError("truncated_header")
    header = json.loads(encoded_header)
    if not isinstance(header, dict):
        raise ValueError("invalid_header_object")
    return header_length, header


def _select_tensors(header: dict[str, Any], prefixes: tuple[str, ...]) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for name, descriptor in header.items():
        if name == "__metadata__" or not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if not isinstance(descriptor, dict):
            raise ValueError(f"invalid_tensor_descriptor:{name}")
        offsets = descriptor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(value, int) for value in offsets):
            raise ValueError(f"invalid_tensor_offsets:{name}")
        if offsets[0] < 0 or offsets[1] <= offsets[0]:
            raise ValueError(f"invalid_tensor_range:{name}")
        selected.append((name, descriptor))
    if not selected:
        raise ValueError("no_matching_tensors")
    return selected


def _compact_header(
    original: dict[str, Any], selected: list[tuple[str, dict[str, Any]]]
) -> tuple[bytes, list[tuple[str, int, int]]]:
    compact: dict[str, Any] = {}
    metadata = original.get("__metadata__")
    if isinstance(metadata, dict):
        compact["__metadata__"] = metadata
    payload_offset = 0
    ranges: list[tuple[str, int, int]] = []
    for name, descriptor in selected:
        source_start, source_end = descriptor["data_offsets"]
        size = source_end - source_start
        copied = dict(descriptor)
        copied["data_offsets"] = [payload_offset, payload_offset + size]
        compact[name] = copied
        ranges.append((name, source_start, source_end))
        payload_offset += size
    encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % HEADER_ALIGNMENT_BYTES
    return encoded + (b" " * padding), ranges


def _piece_download(
    *,
    executable: str,
    config: Path,
    url: str,
    piece: Path,
    absolute_start: int,
    absolute_end: int,
) -> None:
    expected_size = absolute_end - absolute_start + 1
    for attempt in range(DEFAULT_RANGE_RETRIES):
        downloaded = piece.stat().st_size if piece.is_file() else 0
        if downloaded == expected_size:
            return
        if downloaded > expected_size:
            raise ValueError(f"oversized_piece:{piece.name}")
        partial = piece.with_suffix(f"{piece.suffix}.partial")
        partial.unlink(missing_ok=True)
        command = (
            executable,
            "-K",
            str(config),
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            str(DEFAULT_CONNECT_TIMEOUT_SECONDS),
            "--speed-time",
            str(DEFAULT_LOW_SPEED_SECONDS),
            "--speed-limit",
            str(DEFAULT_LOW_SPEED_BYTES_PER_SECOND),
            "--range",
            f"{absolute_start + downloaded}-{absolute_end}",
            "--output",
            str(partial),
            url,
        )
        error: subprocess.CalledProcessError | None = None
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as caught:
            error = caught
        if partial.is_file() and partial.stat().st_size:
            remaining = expected_size - downloaded
            if partial.stat().st_size > remaining:
                raise ValueError(f"oversized_partial:{piece.name}")
            with piece.open("ab") as destination, partial.open("rb") as source:
                shutil.copyfileobj(source, destination, length=COPY_BUFFER_BYTES)
            partial.unlink()
        if piece.is_file() and piece.stat().st_size == expected_size:
            return
        if error is None:
            raise ValueError(f"short_piece:{piece.name}")
        if attempt == DEFAULT_RANGE_RETRIES - 1:
            raise error
    raise ValueError(f"unreachable_piece_retry:{piece.name}")


def _download_ranges(
    directory: Path,
    *,
    executable: str,
    config: Path,
    url: str,
    ranges: list[tuple[str, int, int]],
    source_payload_offset: int,
    connection_count: int,
) -> list[tuple[Path, int]]:
    pieces: list[tuple[Path, int]] = []
    downloads: list[dict[str, Any]] = []
    piece_index = 0
    for _name, start, end in ranges:
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + DEFAULT_RANGE_CHUNK_BYTES, end)
            piece = directory / f"range-{piece_index:05d}.bin"
            pieces.append((piece, chunk_end - chunk_start))
            downloads.append(
                {
                    "executable": executable,
                    "config": config,
                    "url": url,
                    "piece": piece,
                    "absolute_start": source_payload_offset + chunk_start,
                    "absolute_end": source_payload_offset + chunk_end - 1,
                }
            )
            piece_index += 1
            chunk_start = chunk_end
    with concurrent.futures.ThreadPoolExecutor(max_workers=connection_count) as executor:
        futures = [executor.submit(_piece_download, **download) for download in downloads]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return pieces


@contextmanager
def _exclusive_download(directory: Path):
    """Prevent two resumable writers from appending to the same range pieces."""

    lock_path = directory / DOWNLOAD_LOCK_FILENAME
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("concurrent_subset_download") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    arguments = parse_arguments()
    if arguments.connection_count <= 0:
        return _fail(VERIFY_ERROR_CODE, "invalid_connection_count", "provide_positive_connection_count")
    if not arguments.endpoint.startswith(("https://", "http://")):
        return _fail(VERIFY_ERROR_CODE, "invalid_endpoint", "provide_http_endpoint")
    if shutil.which(arguments.curl) is None:
        return _fail(AUTH_ERROR_CODE, "curl_unavailable", "install_curl_and_retry")
    token = get_token()
    if not token:
        return _fail(AUTH_ERROR_CODE, "missing_huggingface_token", "authenticate_and_retry")
    if arguments.header_cache is None or not arguments.header_cache.is_file():
        return _fail(VERIFY_ERROR_CODE, "missing_header_cache", "download_checkpoint_header_range_and_retry")

    try:
        original_header_length, original_header = _read_header(arguments.header_cache)
        selected = _select_tensors(original_header, tuple(arguments.key_prefix))
        compact_header, ranges = _compact_header(original_header, selected)
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
        return _fail(VERIFY_ERROR_CODE, str(error), "replace_header_cache_and_retry")

    url = _download_url(
        arguments.endpoint,
        repository_id=arguments.repository_id,
        revision=arguments.revision,
        relative_path=arguments.relative_path,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        parts_directory = arguments.output.parent / f"{arguments.output.name}.parts"
        parts_directory.mkdir(parents=True, exist_ok=True)
        try:
            with _exclusive_download(parts_directory):
                curl_config = parts_directory / "curl.conf"
                curl_config.write_text(f'header = "Authorization: Bearer {token}"\n', encoding="utf-8")
                os.chmod(curl_config, 0o600)
                pieces = _download_ranges(
                    parts_directory,
                    executable=arguments.curl,
                    config=curl_config,
                    url=url,
                    ranges=ranges,
                    source_payload_offset=HEADER_LENGTH_BYTES + original_header_length,
                    connection_count=arguments.connection_count,
                )
                temporary_output = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
                with temporary_output.open("wb") as destination:
                    destination.write(struct.pack("<Q", len(compact_header)))
                    destination.write(compact_header)
                    for piece, expected_size in pieces:
                        if piece.stat().st_size != expected_size:
                            raise ValueError(f"piece_size_mismatch:{piece.name}")
                        with piece.open("rb") as source:
                            shutil.copyfileobj(source, destination, length=COPY_BUFFER_BYTES)
                _read_header(temporary_output)
                temporary_output.replace(arguments.output)
        finally:
            if arguments.output.is_file():
                shutil.rmtree(parts_directory)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        cause = str(error) if isinstance(error, ValueError) else type(error).__name__
        return _fail(AUTH_ERROR_CODE, cause, "check_network_token_and_retry")

    payload_bytes = sum(end - start for _name, start, end in ranges)
    print(
        json.dumps(
            {
                "ok": True,
                "tensor_count": len(ranges),
                "payload_bytes": payload_bytes,
                "output_bytes": arguments.output.stat().st_size,
                "sha256": _sha256(arguments.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
