#!/usr/bin/env python3
"""Audit every versioned major-component parity report from one manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lara_ltx.errors import LaraError

SUPPORTED_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-PARITY-004", details={"reason": f"unreadable_{path}"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-PARITY-004", details={"reason": f"invalid_{path}"})
    return value


def main() -> int:
    arguments = parse_arguments()
    index = _json(arguments.index)
    components = index.get("components")
    if index.get("schema_version") != SUPPORTED_SCHEMA_VERSION or not isinstance(components, list):
        raise LaraError("LARA-PARITY-004", details={"reason": "invalid_evidence_index"})
    results: list[dict[str, object]] = []
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("name"), str) or not isinstance(
            component.get("report"), str
        ):
            raise LaraError("LARA-PARITY-004", details={"reason": "invalid_evidence_entry"})
        report_path = arguments.root / component["report"]
        report = _json(report_path)
        results.append(
            {
                "name": component["name"],
                "report": component["report"],
                "schema_version": report.get("schema_version"),
                "passed": report.get("passed") is True,
            }
        )
    passed = all(result["passed"] for result in results)
    output = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "component": "major_parity_evidence_audit",
        "component_count": len(results),
        "results": results,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
