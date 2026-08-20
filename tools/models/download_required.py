#!/usr/bin/env python3
"""Download only the pinned full-precision files required by Lara HQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

AUTH_ERROR_CODE = "LARA-MODEL-001"
VERIFY_ERROR_CODE = "LARA-MODEL-002"
TERMINAL_HTTP_STATUS_CODES = frozenset({401, 403, 404})


def _sha256(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(code: str, cause: str, action: str) -> int:
    print(json.dumps({"ok": False, "error_code": code, "cause": cause, "action": action}))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_config = config["upstream"]["model"]
    required_bytes = int(manifest["model"]["required_bytes"])
    reserve = int(model_config["minimum_free_bytes_after_download"])
    args.model_dir.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(args.model_dir).free
    if available < required_bytes + reserve:
        return _fail(
            VERIFY_ERROR_CODE,
            f"insufficient_space:{available}",
            f"provide_at_least:{required_bytes + reserve}",
        )

    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(config["runtime"]["network_timeout_seconds"])
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(config["runtime"]["network_timeout_seconds"])
    retry_count = int(config["runtime"]["network_retry_count"])
    retry_delay_seconds = int(config["runtime"]["network_retry_delay_seconds"])
    hash_chunk_bytes = int(config["runtime"]["hash_chunk_bytes"])
    repo_id = model_config["repository_id"]
    revision = model_config["revision"]

    for entry in manifest["model"]["files"]:
        downloaded: Path | None = None
        last_error: Exception | None = None
        for attempt in range(retry_count):
            try:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=entry["path"],
                        revision=revision,
                        local_dir=args.model_dir,
                    )
                )
                break
            except HfHubHTTPError as exc:
                last_error = exc
                if exc.response.status_code in TERMINAL_HTTP_STATUS_CODES:
                    break
            except OSError as exc:
                last_error = exc
            if attempt + 1 < retry_count:
                time.sleep(retry_delay_seconds)

        if downloaded is None:
            if isinstance(last_error, HfHubHTTPError):
                cause = f"http_status:{last_error.response.status_code}"
                return _fail(AUTH_ERROR_CODE, cause, "check_gated_repo_access")
            return _fail(AUTH_ERROR_CODE, type(last_error).__name__, "check_network_and_retry")

        actual_size = downloaded.stat().st_size
        if actual_size != entry["size"]:
            return _fail(VERIFY_ERROR_CODE, f"size_mismatch:{entry['path']}", "download_affected_file_again")
        entry["sha256"] = _sha256(downloaded, hash_chunk_bytes)
        entry["verification"] = "size_and_sha256_verified"

    temporary = args.manifest.with_suffix(f"{args.manifest.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    print(json.dumps({"ok": True, "files": len(manifest["model"]["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
