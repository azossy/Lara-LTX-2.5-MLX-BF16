#!/usr/bin/env python3
"""Write the reviewed Gemma V2 feature-extractor mapping manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.models.gemma_feature import write_gemma_feature_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    write_gemma_feature_mapping(arguments.output, arguments.checkpoint)


if __name__ == "__main__":
    main()
