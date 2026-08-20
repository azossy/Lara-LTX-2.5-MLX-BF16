#!/usr/bin/env python3
"""Write the reviewed MLX mapping manifest for the official DurationHead shard."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.models import write_duration_head_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_duration_head_mapping(arguments.output, arguments.checkpoint)


if __name__ == "__main__":
    main()
