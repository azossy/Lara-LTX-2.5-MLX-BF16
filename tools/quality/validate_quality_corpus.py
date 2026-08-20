#!/usr/bin/env python3
"""Validate P5 corpus structure without making a quality-parity claim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.quality.corpus import QualityCorpusError, load_corpus, validation_report, write_json_atomic
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from corpus import QualityCorpusError, load_corpus, validation_report, write_json_atomic


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        write_json_atomic(arguments.report, validation_report(load_corpus(arguments.corpus)))
    except QualityCorpusError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
