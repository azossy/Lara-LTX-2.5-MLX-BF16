#!/usr/bin/env python3
"""Generate every versioned P5 quality case through one public MLX pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lara_ltx import LTXPipeline

try:
    from tools.quality.corpus import (
        ERROR_CODE,
        QualityCorpus,
        QualityCorpusError,
        file_sha256,
        load_corpus,
        message,
        write_json_atomic,
    )
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from corpus import (
        ERROR_CODE,
        QualityCorpus,
        QualityCorpusError,
        file_sha256,
        load_corpus,
        message,
        write_json_atomic,
    )

REPORT_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def _generation_report(
    *,
    corpus: QualityCorpus,
    model: str,
    results: list[dict[str, object]],
    requested_case_count: int,
) -> dict[str, object]:
    total_case_count = len(corpus.cases)
    completed_case_count = len(results)
    complete = completed_case_count == total_case_count
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": "mlx_quality_corpus_generation",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "corpus": {"name": corpus.name, "sha256": corpus.sha256},
        "model": model,
        "total_case_count": total_case_count,
        "requested_case_count": requested_case_count,
        "completed_case_count": completed_case_count,
        "results": results,
        "status": "complete" if complete else "partial",
        "passed": complete,
        "scope": "generation_only_not_quality_acceptance",
    }


def main() -> int:
    arguments = parse_arguments()
    try:
        corpus = load_corpus(arguments.corpus)
        if arguments.max_cases is not None and arguments.max_cases <= 0:
            raise QualityCorpusError("invalid_max_cases")
        selected_cases = corpus.cases[: arguments.max_cases]
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        pipeline: LTXPipeline | None = None
        results = []
        write_json_atomic(
            arguments.report,
            _generation_report(
                corpus=corpus,
                model=arguments.model,
                results=results,
                requested_case_count=len(selected_cases),
            ),
        )
        for case in selected_cases:
            requested_output = arguments.output_dir / f"{case.case_id}.mp4"
            started = time.perf_counter()
            reused = arguments.resume and requested_output.is_file() and requested_output.stat().st_size > 0
            if reused:
                output = requested_output
            else:
                try:
                    if pipeline is None:
                        pipeline = LTXPipeline.from_pretrained(
                            arguments.model,
                            profile_path=arguments.profile,
                            cache_dir=arguments.cache_dir,
                            local_files_only=arguments.local_files_only,
                        )
                    video = pipeline(
                        prompt=case.prompt,
                        seed=case.seed,
                        width=case.width,
                        height=case.height,
                        num_frames=case.num_frames,
                        frame_rate=case.frame_rate,
                        num_inference_steps=case.num_inference_steps,
                    )
                    output = video.save(requested_output)
                except Exception as error:
                    raise RuntimeError(
                        f"[{ERROR_CODE}] {message('generation_failed', case_id=case.case_id)}"
                    ) from error
            elapsed_seconds = time.perf_counter() - started
            results.append(
                {
                    "case_id": case.case_id,
                    "elapsed_seconds": elapsed_seconds,
                    "generated_frames_per_second": case.num_frames / elapsed_seconds,
                    "output": str(output),
                    "size_bytes": output.stat().st_size,
                    "sha256": file_sha256(output),
                    "reused": reused,
                    "parameters": {
                        "seed": case.seed,
                        "width": case.width,
                        "height": case.height,
                        "num_frames": case.num_frames,
                        "frame_rate": case.frame_rate,
                        "num_inference_steps": case.num_inference_steps,
                    },
                }
            )
            write_json_atomic(
                arguments.report,
                _generation_report(
                    corpus=corpus,
                    model=arguments.model,
                    results=results,
                    requested_case_count=len(selected_cases),
                ),
            )
    except (QualityCorpusError, RuntimeError, OSError) as error:
        print(error, file=sys.stderr)
        return 2
    print(message("completed", count=len(results), report=arguments.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
