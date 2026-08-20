#!/usr/bin/env python3
"""Create the exact official Gemma 4 text-core mapping manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.models import write_gemma_text_mapping


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    write_gemma_text_mapping(arguments.output, arguments.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
