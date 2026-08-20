#!/usr/bin/env python3
"""Write an exact production AV transformer-block mapping manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.models.transformer_block import write_transformer_block_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--block-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    write_transformer_block_mapping(arguments.output, arguments.checkpoint, arguments.block_index)


if __name__ == "__main__":
    main()
