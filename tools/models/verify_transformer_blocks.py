#!/usr/bin/env python3
"""Validate every production AV transformer-block mapping from one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lara_ltx.errors import LaraError
from lara_ltx.models.transformer_block import build_transformer_block_mapping

DEFAULT_TRANSFORMER_BLOCK_COUNT = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-count", type=int, default=DEFAULT_TRANSFORMER_BLOCK_COUNT)
    return parser.parse_args()


def _mapping_sha256(mapping: dict[str, object]) -> str:
    encoded = json.dumps(mapping, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    arguments = parse_args()
    if arguments.block_count <= 0:
        raise LaraError("LARA-MODEL-028", details={"key": f"block_count={arguments.block_count}"})
    blocks = []
    for block_index in range(arguments.block_count):
        mapping = build_transformer_block_mapping(arguments.checkpoint, block_index)
        rules = mapping["rules"]
        assert isinstance(rules, list)
        blocks.append(
            {
                "index": block_index,
                "rule_count": len(rules),
                "mapping_sha256": _mapping_sha256(mapping),
            }
        )
    report = {
        "schema_version": 1,
        "component": "transformer_blocks",
        "checkpoint": arguments.checkpoint.name,
        "block_count": arguments.block_count,
        "blocks": blocks,
    }
    temporary = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
