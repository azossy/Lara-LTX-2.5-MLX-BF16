#!/usr/bin/env python3
"""Build a deterministic, backend-anonymized audiovisual review kit."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from tools.quality.corpus import QualityCorpusError, file_sha256, load_corpus, message, write_json_atomic
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from corpus import QualityCorpusError, file_sha256, load_corpus, message, write_json_atomic

CONFIG_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
BACKENDS = ("cuda", "mlx")
LABEL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]{0,15}$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--cuda-directory", required=True, type=Path)
    parser.add_argument("--mlx-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--key-report", required=True, type=Path)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        labels = tuple(str(item) for item in value["candidate_labels"])
        dimensions = tuple(str(item) for item in value["rating_dimensions"])
        choices = tuple(str(item) for item in value["allowed_choices"])
        seed_environment_variable = str(value["assignment_seed_environment_variable"])
        assignment_seed = int(os.environ[seed_environment_variable])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise QualityCorpusError("invalid_blind_review_config") from error
    if (
        value.get("schema_version") != CONFIG_SCHEMA_VERSION
        or len(labels) != len(BACKENDS)
        or len(set(labels)) != len(labels)
        or any(not LABEL_PATTERN.fullmatch(label) for label in labels)
        or not dimensions
        or not choices
        or not seed_environment_variable
        or assignment_seed < 0
    ):
        raise QualityCorpusError("invalid_blind_review_config")
    return {
        "labels": labels,
        "dimensions": dimensions,
        "choices": choices,
        "assignment_seed": assignment_seed,
        "seed_environment_variable": seed_environment_variable,
    }


def build_kit(
    *,
    config_path: Path,
    corpus_path: Path,
    cuda_directory: Path,
    mlx_directory: Path,
    output_directory: Path,
    key_report: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    corpus = load_corpus(corpus_path)
    labels = config["labels"]
    rng = random.Random(config["assignment_seed"])
    media_directory = output_directory / "media"
    media_directory.mkdir(parents=True, exist_ok=True)
    public_items: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    for index, case in enumerate(corpus.cases, start=1):
        item_id = f"item_{index:02d}"
        ordered_backends = list(BACKENDS)
        rng.shuffle(ordered_backends)
        public_candidates: dict[str, str] = {}
        private_candidates: dict[str, dict[str, str]] = {}
        for label, backend in zip(labels, ordered_backends, strict=True):
            source_directory = cuda_directory if backend == "cuda" else mlx_directory
            source = source_directory / f"{case.case_id}.mp4"
            if not source.is_file() or source.stat().st_size <= 0:
                raise QualityCorpusError(f"missing_blind_review_media:{case.case_id}:{backend}")
            destination = media_directory / f"{item_id}_{label}.mp4"
            shutil.copyfile(source, destination)
            source_hash = file_sha256(source)
            if file_sha256(destination) != source_hash:
                raise QualityCorpusError(f"blind_review_copy_mismatch:{case.case_id}:{backend}")
            public_candidates[label] = str(destination.relative_to(output_directory))
            private_candidates[label] = {"backend": backend, "source_sha256": source_hash}
        public_items.append(
            {
                "item_id": item_id,
                "prompt": case.prompt,
                "review_focus": list(case.review_focus),
                "candidates": public_candidates,
            }
        )
        private_items.append(
            {
                "item_id": item_id,
                "case_id": case.case_id,
                "candidates": private_candidates,
            }
        )
    public_manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": "blind_audiovisual_review_kit",
        "corpus": {"name": corpus.name, "sha256": corpus.sha256},
        "instructions": message("blind_review_instructions"),
        "items": public_items,
        "submission_schema": {
            "rating_dimensions": list(config["dimensions"]),
            "allowed_choices": list(config["choices"]),
            "required_fields": ["reviewer_id", "item_id", "ratings", "notes"],
        },
        "backend_identity_disclosed": False,
    }
    private_key = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": "blind_audiovisual_review_key",
        "corpus": {"name": corpus.name, "sha256": corpus.sha256},
        "assignment_seed": config["assignment_seed"],
        "items": private_items,
    }
    submission_template = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": "blind_audiovisual_review_submission",
        "reviewer_id": "",
        "reviews": [
            {
                "item_id": item["item_id"],
                "ratings": {dimension: "" for dimension in config["dimensions"]},
                "notes": "",
            }
            for item in public_items
        ],
    }
    write_json_atomic(output_directory / "review_manifest.json", public_manifest)
    write_json_atomic(output_directory / "review_submission_template.json", submission_template)
    write_json_atomic(key_report, private_key)
    return public_manifest


def main() -> int:
    arguments = parse_arguments()
    try:
        report = build_kit(
            config_path=arguments.config,
            corpus_path=arguments.corpus,
            cuda_directory=arguments.cuda_directory,
            mlx_directory=arguments.mlx_directory,
            output_directory=arguments.output_directory,
            key_report=arguments.key_report,
        )
    except (QualityCorpusError, OSError) as error:
        print(f"{error}\n{message('blind_review_failed')}", file=sys.stderr)
        return 2
    print(message("blind_review_completed", count=len(report["items"]), output=arguments.output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
