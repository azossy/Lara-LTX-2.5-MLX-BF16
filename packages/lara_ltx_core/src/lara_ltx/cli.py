"""Thin localized command-line wrapper around :class:`LTXPipeline`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lara_ltx.errors import LaraError, localize

EXIT_SUCCESS = 0
EXIT_USER_ERROR = 2
EXIT_INTERRUPTED = 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=localize("cli.description"))
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help=localize("cli.generate.help"))
    generate.add_argument("--model", required=True, help=localize("cli.model.help"))
    generate.add_argument("--prompt", required=True, help=localize("cli.prompt.help"))
    generate.add_argument("--negative-prompt", help=localize("cli.negative_prompt.help"))
    generate.add_argument("--seed", type=int, help=localize("cli.seed.help"))
    generate.add_argument("--height", type=int, help=localize("cli.height.help"))
    generate.add_argument("--width", type=int, help=localize("cli.width.help"))
    generate.add_argument("--num-frames", type=int, help=localize("cli.num_frames.help"))
    generate.add_argument("--frame-rate", type=float, help=localize("cli.frame_rate.help"))
    generate.add_argument("--steps", type=int, help=localize("cli.steps.help"))
    generate.add_argument("--profile", type=Path, help=localize("cli.profile.help"))
    generate.add_argument("--cache-dir", type=Path, help=localize("cli.cache_dir.help"))
    generate.add_argument("--local-files-only", action="store_true", help=localize("cli.local_only.help"))
    generate.add_argument("--output", required=True, type=Path, help=localize("cli.output.help"))
    return parser


def _generate(arguments: argparse.Namespace) -> int:
    from lara_ltx.pipeline import LTXPipeline

    pipeline = LTXPipeline.from_pretrained(
        arguments.model,
        cache_dir=arguments.cache_dir,
        local_files_only=arguments.local_files_only,
        profile_path=arguments.profile,
    )
    video = pipeline(
        prompt=arguments.prompt,
        negative_prompt=arguments.negative_prompt,
        seed=arguments.seed,
        height=arguments.height,
        width=arguments.width,
        num_frames=arguments.num_frames,
        frame_rate=arguments.frame_rate,
        num_inference_steps=arguments.steps,
    )
    output = video.save(arguments.output)
    print(localize("cli.generate.saved", details={"path": output}))
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command == "generate":
            return _generate(arguments)
        raise LaraError("LARA-CLI-001", details={"reason": "unsupported_command"})
    except LaraError as error:
        print(str(error), file=sys.stderr)
        return EXIT_USER_ERROR
    except KeyboardInterrupt:
        print(localize("cli.interrupted"), file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
