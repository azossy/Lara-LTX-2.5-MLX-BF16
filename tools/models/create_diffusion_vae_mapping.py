"""Create the audited official Diffusion VAE decoder mapping manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from lara_ltx.errors import LaraError
from lara_ltx.models import write_diffusion_vae_decoder_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        write_diffusion_vae_decoder_mapping(arguments.output, arguments.checkpoint)
    except LaraError as error:
        print(str(error))
        return 2
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
