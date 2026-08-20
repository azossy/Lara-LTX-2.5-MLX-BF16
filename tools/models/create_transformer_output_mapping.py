#!/usr/bin/env python3
"""Write the exact video/audio transformer output mapping manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.models.transformer_output import write_transformer_output_mapping


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    write_transformer_output_mapping(arguments.output, arguments.checkpoint)


if __name__ == "__main__":
    main()
