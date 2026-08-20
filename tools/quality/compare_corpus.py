#!/usr/bin/env python3
"""Compare every matched CUDA/MLX audiovisual quality-corpus case."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from tools.quality.corpus import (
        ERROR_CODE,
        QualityCorpusError,
        load_corpus,
        message,
        write_json_atomic,
    )
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from corpus import ERROR_CODE, QualityCorpusError, load_corpus, message, write_json_atomic

REPORT_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--comparator", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--max-audio-lag-ms", required=True, type=float)
    parser.add_argument("--case-report-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        corpus = load_corpus(arguments.corpus)
        arguments.case_report_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for case in corpus.cases:
            reference = arguments.reference_dir / f"{case.case_id}.mp4"
            candidate = arguments.candidate_dir / f"{case.case_id}.mp4"
            case_report = arguments.case_report_dir / f"{case.case_id}.json"
            if not reference.is_file() or not candidate.is_file():
                raise QualityCorpusError(f"missing_media_pair:{case.case_id}")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(arguments.comparator),
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--ffmpeg",
                    arguments.ffmpeg,
                    "--ffprobe",
                    arguments.ffprobe,
                    "--max-audio-lag-ms",
                    str(arguments.max_audio_lag_ms),
                    "--report",
                    str(case_report),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise QualityCorpusError(f"comparison_failed:{case.case_id}")
            comparison = json.loads(case_report.read_text(encoding="utf-8"))
            results.append(
                {
                    "case_id": case.case_id,
                    "report": str(case_report),
                    "frame_cosine": comparison["video"]["frame_metrics"]["cosine_similarity"],
                    "frame_psnr_db": comparison["video"]["frame_metrics"]["psnr_db"],
                    "temporal_cosine": comparison["video"]["temporal_delta_metrics"]["cosine_similarity"],
                    "audio_cosine": comparison["audio"]["metrics"]["cosine_similarity"],
                    "aligned_audio_cosine": comparison["audio"]["alignment"]["aligned_cosine_similarity"],
                    "audio_lag_ms": comparison["audio"]["alignment"]["lag_ms"],
                }
            )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "component": "cuda_mlx_quality_corpus_diagnostics",
            "corpus": {"name": corpus.name, "sha256": corpus.sha256},
            "case_count": len(results),
            "results": results,
            "passed": len(results) == len(corpus.cases),
            "scope": "paired_diagnostics_not_perceptual_or_blind_acceptance",
        }
        write_json_atomic(arguments.report, report)
    except (QualityCorpusError, OSError, KeyError, TypeError, ValueError) as error:
        details = message("comparison_failed", case_id=getattr(error, "reason", "unknown"))
        print(f"[{ERROR_CODE}] {details}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
