"""Create the reviewed mapping for the LTX 2.5 Audio VAE core."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.models import write_audio_vae_mapping


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Verified official Audio VAE/vocoder safetensors shard")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON mapping path")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    write_audio_vae_mapping(arguments.output, arguments.checkpoint)


if __name__ == "__main__":
    main()
