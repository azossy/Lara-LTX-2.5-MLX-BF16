#!/usr/bin/env python3
"""Capture a reproducible, non-secret runtime environment manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - remote benchmark compatibility on Python 3.10/3.11
    import tomli as tomllib

COMMAND_TIMEOUT_SECONDS = 20
PACKAGE_NAMES = ("huggingface-hub", "mlx", "numpy", "safetensors", "torch")
NVIDIA_QUERY_ARGUMENTS = (
    "--query-gpu=index,name,memory.total,driver_version",
    "--format=csv,noheader",
)
SAFE_HARDWARE_LABELS = (
    "Model Name:",
    "Model Identifier:",
    "Chip:",
    "Total Number of Cores:",
    "Memory:",
)
UTC = timezone.utc


def _run(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {
        "available": result.returncode == 0,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _safe_hardware_output(command_result: dict[str, Any]) -> dict[str, Any]:
    if not command_result.get("available"):
        return command_result
    lines = command_result["stdout"].splitlines()
    command_result["stdout"] = "\n".join(
        line.strip() for line in lines if line.strip().startswith(SAFE_HARDWARE_LABELS)
    )
    return command_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    commands = config["commands"]

    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "packages": _versions(),
        "commands": {
            "git": _run([commands["git"], "--version"]),
            "nvidia_smi": _run([commands["nvidia_smi"], *NVIDIA_QUERY_ARGUMENTS]),
            "system_profiler": _safe_hardware_output(_run([commands["system_profiler"], "SPHardwareDataType"])),
            "sw_vers": _run([commands["sw_vers"]]),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
