#!/usr/bin/env python3
"""Write the validated distilled-LoRA pair manifest for the pinned BF16 pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.models.lora import write_lora_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lora_checkpoint", type=Path)
    parser.add_argument("transformer_checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    write_lora_manifest(arguments.output, arguments.lora_checkpoint, arguments.transformer_checkpoint)


if __name__ == "__main__":
    main()
