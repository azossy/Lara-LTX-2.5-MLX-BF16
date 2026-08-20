#!/usr/bin/env python3
"""Locate the first failing ordered tensor boundary between two NPZ archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import ParityTolerance, find_first_divergence

SUPPORTED_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--tolerances", required=True, type=Path)
    parser.add_argument("--component")
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-PARITY-004", details={"reason": f"unreadable_{path.name}"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-PARITY-004", details={"reason": f"invalid_{path.name}"})
    return value


def _tolerance(value: dict[str, Any]) -> ParityTolerance:
    try:
        maximum = value.get("max_absolute_error")
        return ParityTolerance(
            max_normalized_rmse=float(value["max_normalized_rmse"]),
            min_cosine_similarity=float(value["min_cosine_similarity"]),
            max_absolute_error=None if maximum is None else float(maximum),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LaraError("LARA-PARITY-004", details={"reason": "invalid_tolerance_profile"}) from error


def main() -> int:
    arguments = parse_arguments()
    key_config = _json(arguments.keys)
    tolerance_config = _json(arguments.tolerances)
    if key_config.get("schema_version") != SUPPORTED_SCHEMA_VERSION or tolerance_config.get(
        "schema_version"
    ) != SUPPORTED_SCHEMA_VERSION:
        raise LaraError("LARA-PARITY-004", details={"reason": "unsupported_schema"})
    ordered_keys = key_config.get("ordered_keys")
    if not isinstance(ordered_keys, list) or not all(isinstance(key, str) for key in ordered_keys):
        raise LaraError("LARA-PARITY-004", details={"reason": "invalid_ordered_keys"})
    selected = tolerance_config.get("default")
    if arguments.component is not None:
        selected = tolerance_config.get("component_overrides", {}).get(arguments.component, selected)
    if not isinstance(selected, dict):
        raise LaraError("LARA-PARITY-004", details={"reason": "missing_tolerance"})
    try:
        with np.load(arguments.reference) as archive:
            reference = {key: archive[key] for key in ordered_keys}
        with np.load(arguments.candidate) as archive:
            candidate = {key: archive[key] for key in ordered_keys}
    except (OSError, KeyError, ValueError) as error:
        raise LaraError("LARA-PARITY-004", details={"reason": "unreadable_tensor_archive"}) from error
    result = find_first_divergence(
        reference,
        candidate,
        ordered_keys=ordered_keys,
        tolerance=_tolerance(selected),
    )
    report = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "component": arguments.component,
        **result.to_dict(),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
