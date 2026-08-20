#!/usr/bin/env python3
"""Fail fast when local source differs from the pinned canonical revision."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path

ERROR_CODE = "LARA-UPSTREAM-001"
GIT_TIMEOUT_SECONDS = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    expected = config["upstream"]["source"]["commit"]
    git_command = config["commands"]["git"]

    try:
        actual = subprocess.run(
            [git_command, "-C", str(args.checkout), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"ok": False, "error_code": ERROR_CODE, "cause": type(exc).__name__}))
        return 2

    result = {"ok": actual == expected, "expected": expected, "actual": actual}
    if not result["ok"]:
        result["error_code"] = ERROR_CODE
    print(json.dumps(result))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
