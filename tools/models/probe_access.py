#!/usr/bin/env python3
"""Probe gated checkpoint access without downloading tensor payloads."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
import tomllib
from pathlib import Path
from typing import Any

AUTH_ERROR_CODE = "LARA-MODEL-001"
NETWORK_ERROR_CODE = "LARA-MODEL-013"
TERMINAL_HTTP_STATUS_CODES = frozenset({401, 403, 404})
RESULT_POLL_TIMEOUT_SECONDS = 0.1


def _failure(code: str, cause: str, action: str) -> int:
    print(json.dumps({"ok": False, "error_code": code, "cause": cause, "action": action}))
    return 2


def _configured_probe(config_path: Path, manifest_path: Path) -> tuple[str, str, str, int, int, int]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["model"]["files"]
    if not files:
        raise ValueError("manifest_has_no_model_files")
    model_config = config["upstream"]["model"]
    runtime = config["runtime"]
    return (
        str(model_config["repository_id"]),
        str(model_config["revision"]),
        str(files[0]["path"]),
        int(runtime["network_timeout_seconds"]),
        int(runtime["network_retry_count"]),
        int(runtime["network_retry_delay_seconds"]),
    )


def _probe_once(
    result_sender: multiprocessing.connection.Connection,
    repo_id: str,
    revision: str,
    filename: str,
    timeout_seconds: int,
) -> None:
    """Perform one killable metadata request in an isolated child process."""

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError

        information = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            etag_timeout=timeout_seconds,
            dry_run=True,
        )
        result_sender.send(("ok", getattr(information, "file_size", None)))
    except HfHubHTTPError as exc:
        result_sender.send(("http", exc.response.status_code))
    except Exception as exc:  # Boundary to an external library/network process.
        result_sender.send(("error", type(exc).__name__))
    finally:
        result_sender.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        repo_id, revision, filename, timeout_seconds, retry_count, retry_delay_seconds = _configured_probe(
            arguments.config,
            arguments.manifest,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return _failure(NETWORK_ERROR_CODE, type(exc).__name__, "check_probe_config_and_manifest")

    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout_seconds)
    last_cause = "probe_did_not_return"
    for attempt in range(retry_count):
        receiver, sender = multiprocessing.Pipe(duplex=False)
        process = multiprocessing.Process(
            target=_probe_once,
            args=(sender, repo_id, revision, filename, timeout_seconds),
        )
        process.start()
        sender.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            receiver.close()
            last_cause = "timeout"
        elif receiver.poll(RESULT_POLL_TIMEOUT_SECONDS):
            status, value = receiver.recv()
            receiver.close()
            if status == "ok":
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "authorized": True,
                            "repository": repo_id,
                            "revision": revision,
                            "file": filename,
                            "size": value,
                        }
                    )
                )
                return 0
            if status == "http" and value in TERMINAL_HTTP_STATUS_CODES:
                return _failure(AUTH_ERROR_CODE, f"http_status:{value}", "check_gated_repo_access")
            last_cause = f"http_status:{value}" if status == "http" else str(value)
        else:
            receiver.close()
            last_cause = "probe_did_not_return"
        if attempt + 1 < retry_count:
            time.sleep(retry_delay_seconds)
    return _failure(NETWORK_ERROR_CODE, last_cause, "check_network_and_retry")


if __name__ == "__main__":
    raise SystemExit(main())
