"""Resumable, hash-verifying download of the pinned BF16 pack through aria2."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - AutoDL utility compatibility on Python 3.10/3.11
    import tomli as tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote

from huggingface_hub import get_token

AUTH_ERROR_CODE = "LARA-MODEL-001"
VERIFY_ERROR_CODE = "LARA-MODEL-002"
DEFAULT_CONNECTION_COUNT = 16
DEFAULT_MIN_SPLIT_SIZE = "1M"
ARIA2_CONTROL_SUFFIX = ".aria2"


def _sha256(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(code: str, cause: str, action: str) -> int:
    print(json.dumps({"ok": False, "error_code": code, "cause": cause, "action": action}))
    return 2


def _download_url(endpoint: str, *, repository_id: str, revision: str, relative_path: str) -> str:
    normalized_endpoint = endpoint.rstrip("/")
    return "/".join(
        (
            normalized_endpoint,
            quote(repository_id, safe="/"),
            "resolve",
            quote(revision, safe=""),
            quote(relative_path, safe="/"),
        )
    )


def _entry_needs_download(entry: dict[str, Any], model_dir: Path) -> bool:
    destination = model_dir / str(entry["path"])
    control_file = Path(f"{destination}{ARIA2_CONTROL_SUFFIX}")
    return (
        not destination.is_file()
        or destination.stat().st_size != int(entry["size"])
        or control_file.exists()
    )


def _remaining_download_bytes(entries: list[dict[str, Any]], model_dir: Path) -> int:
    """Return additional storage needed for missing or resumable file ranges.

    aria2 range downloads can expose the final logical file size while the file
    is still sparse.  In that case ``st_blocks`` reflects the disk space that
    has actually been populated and avoids demanding capacity for the whole
    model pack again on every resume or verification-only run.
    """
    remaining = 0
    for entry in entries:
        expected_size = int(entry["size"])
        destination = model_dir / str(entry["path"])
        control_file = Path(f"{destination}{ARIA2_CONTROL_SUFFIX}")
        if not destination.is_file():
            remaining += expected_size
        elif not _entry_needs_download(entry, model_dir):
            continue
        elif control_file.exists():
            allocated_bytes = min(expected_size, destination.stat().st_blocks * 512)
            remaining += max(0, expected_size - allocated_bytes)
        else:
            # A wrong-sized file without aria2 resume metadata may need a full
            # replacement, so retain the conservative capacity requirement.
            remaining += expected_size
    return remaining


def _entry_sha256_matches(entry: dict[str, Any], destination: Path, chunk_bytes: int) -> bool:
    expected_sha256 = str(entry.get("sha256", "")).lower()
    if len(expected_sha256) != 64:
        return False
    return hmac.compare_digest(_sha256(destination, chunk_bytes), expected_sha256)


def _download_with_aria2(
    *,
    executable: str,
    endpoint: str,
    repository_id: str,
    revision: str,
    relative_path: str,
    model_dir: Path,
    connection_count: int,
) -> None:
    token = get_token()
    if not token:
        raise RuntimeError("missing_huggingface_token")
    destination = model_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lara-hf-aria2-", dir=model_dir) as temporary_directory:
        config_path = Path(temporary_directory) / "aria2.conf"
        config_path.write_text(f"header=Authorization: Bearer {token}\n", encoding="utf-8")
        os.chmod(config_path, 0o600)
        command = (
            executable,
            f"--conf-path={config_path}",
            "--continue=true",
            f"--max-connection-per-server={connection_count}",
            f"--split={connection_count}",
            f"--min-split-size={DEFAULT_MIN_SPLIT_SIZE}",
            "--file-allocation=none",
            "--console-log-level=warn",
            f"--dir={destination.parent}",
            f"--out={destination.name}",
            _download_url(
                endpoint,
                repository_id=repository_id,
                revision=revision,
                relative_path=relative_path,
            ),
        )
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--aria2", default="aria2c")
    parser.add_argument("--connection-count", type=int, default=DEFAULT_CONNECTION_COUNT)
    arguments = parser.parse_args()
    if arguments.connection_count <= 0:
        return _fail(VERIFY_ERROR_CODE, "invalid_connection_count", "provide_positive_connection_count")
    if not arguments.endpoint.startswith(("https://", "http://")):
        return _fail(VERIFY_ERROR_CODE, "invalid_endpoint", "provide_http_endpoint")
    if shutil.which(arguments.aria2) is None:
        return _fail(AUTH_ERROR_CODE, "aria2_unavailable", "install_aria2_and_retry")

    with arguments.config.open("rb") as handle:
        config = tomllib.load(handle)
    manifest: dict[str, Any] = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    model_config = config["upstream"]["model"]
    reserve = int(model_config["minimum_free_bytes_after_download"])
    arguments.model_dir.mkdir(parents=True, exist_ok=True)
    entries = list(manifest["model"]["files"])
    pending_entries = [entry for entry in entries if _entry_needs_download(entry, arguments.model_dir)]
    remaining_download_bytes = _remaining_download_bytes(entries, arguments.model_dir)
    available = shutil.disk_usage(arguments.model_dir).free
    required_available = remaining_download_bytes + reserve
    if available < required_available:
        return _fail(
            VERIFY_ERROR_CODE,
            f"insufficient_space:{available}",
            f"provide_at_least:{required_available}",
        )

    hash_chunk_bytes = int(config["runtime"]["hash_chunk_bytes"])
    repository_id = str(model_config["repository_id"])
    revision = str(model_config["revision"])
    worker_count = max(1, int(model_config.get("download_worker_count", 1)))
    try:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(pending_entries) or 1)) as executor:
            futures = {
                executor.submit(
                    _download_with_aria2,
                    executable=arguments.aria2,
                    endpoint=arguments.endpoint,
                    repository_id=repository_id,
                    revision=revision,
                    relative_path=str(entry["path"]),
                    model_dir=arguments.model_dir,
                    connection_count=arguments.connection_count,
                ): str(entry["path"])
                for entry in pending_entries
            }
            for future in as_completed(futures):
                future.result()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        return _fail(AUTH_ERROR_CODE, type(error).__name__, "check_network_token_and_retry")

    for entry in entries:
        relative_path = str(entry["path"])
        destination = arguments.model_dir / relative_path
        if not destination.is_file() or destination.stat().st_size != int(entry["size"]):
            return _fail(VERIFY_ERROR_CODE, f"size_mismatch:{relative_path}", "download_affected_file_again")
        if not _entry_sha256_matches(entry, destination, hash_chunk_bytes):
            return _fail(
                VERIFY_ERROR_CODE,
                f"sha256_mismatch:{relative_path}",
                "download_affected_file_again",
            )
        entry["verification"] = "size_and_sha256_verified"

    temporary = arguments.manifest.with_suffix(f"{arguments.manifest.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(arguments.manifest)
    print(json.dumps({"ok": True, "files": len(manifest["model"]["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
