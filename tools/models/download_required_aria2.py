"""Resumable, hash-verifying download of the pinned BF16 pack through aria2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote

from huggingface_hub import get_token

AUTH_ERROR_CODE = "LARA-MODEL-001"
VERIFY_ERROR_CODE = "LARA-MODEL-002"
DEFAULT_CONNECTION_COUNT = 16
DEFAULT_MIN_SPLIT_SIZE = "1M"


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
    required_bytes = int(manifest["model"]["required_bytes"])
    reserve = int(model_config["minimum_free_bytes_after_download"])
    arguments.model_dir.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(arguments.model_dir).free
    if available < required_bytes + reserve:
        return _fail(
            VERIFY_ERROR_CODE,
            f"insufficient_space:{available}",
            f"provide_at_least:{required_bytes + reserve}",
        )

    hash_chunk_bytes = int(config["runtime"]["hash_chunk_bytes"])
    repository_id = str(model_config["repository_id"])
    revision = str(model_config["revision"])
    for entry in manifest["model"]["files"]:
        relative_path = str(entry["path"])
        destination = arguments.model_dir / relative_path
        try:
            if not destination.is_file() or destination.stat().st_size != int(entry["size"]):
                _download_with_aria2(
                    executable=arguments.aria2,
                    endpoint=arguments.endpoint,
                    repository_id=repository_id,
                    revision=revision,
                    relative_path=relative_path,
                    model_dir=arguments.model_dir,
                    connection_count=arguments.connection_count,
                )
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            return _fail(AUTH_ERROR_CODE, type(error).__name__, "check_network_token_and_retry")
        if not destination.is_file() or destination.stat().st_size != int(entry["size"]):
            return _fail(VERIFY_ERROR_CODE, f"size_mismatch:{relative_path}", "download_affected_file_again")
        entry["sha256"] = _sha256(destination, hash_chunk_bytes)
        entry["verification"] = "size_and_sha256_verified"

    temporary = arguments.manifest.with_suffix(f"{arguments.manifest.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(arguments.manifest)
    print(json.dumps({"ok": True, "files": len(manifest["model"]["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
