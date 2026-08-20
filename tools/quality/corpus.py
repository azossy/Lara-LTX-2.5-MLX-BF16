"""Strict schema and diversity validation for the P5 quality corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_SCHEMA_VERSION = 1
ERROR_CODE = "LARA-QUALITY-001"
DEFAULT_LOCALE = "en"
LOCALE_ENVIRONMENT_VARIABLE = "LARA_LOCALE"
MESSAGES_DIRECTORY = Path(__file__).with_name("messages")
CASE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
HASH_BLOCK_BYTES = 1024 * 1024


class QualityCorpusError(ValueError):
    """Stable developer-tool failure without importing the MLX runtime."""

    def __init__(self, reason: str) -> None:
        self.code = ERROR_CODE
        self.reason = reason
        super().__init__(f"[{ERROR_CODE}] {message('invalid_corpus', reason=reason)}")


def message(key: str, **details: object) -> str:
    requested = os.getenv(LOCALE_ENVIRONMENT_VARIABLE, DEFAULT_LOCALE).split("_", maxsplit=1)[0].lower()
    locale = requested if requested in {"en", "ko"} else DEFAULT_LOCALE
    try:
        catalog = json.loads((MESSAGES_DIRECTORY / f"{locale}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        catalog = json.loads((MESSAGES_DIRECTORY / f"{DEFAULT_LOCALE}.json").read_text(encoding="utf-8"))
    return str(catalog.get(key, key)).format(**details)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    prompt: str
    seed: int
    width: int
    height: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    review_focus: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QualityCase":
        try:
            result = cls(
                case_id=str(value["id"]),
                prompt=str(value["prompt"]),
                seed=int(value["seed"]),
                width=int(value["width"]),
                height=int(value["height"]),
                num_frames=int(value["num_frames"]),
                frame_rate=float(value["frame_rate"]),
                num_inference_steps=int(value["num_inference_steps"]),
                review_focus=tuple(str(item) for item in value["review_focus"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise QualityCorpusError("invalid_case_fields") from error
        result.validate()
        return result

    def validate(self) -> None:
        if not CASE_ID_PATTERN.fullmatch(self.case_id):
            raise QualityCorpusError(f"invalid_case_id:{self.case_id}")
        if (
            not self.prompt.strip()
            or self.seed < 0
            or self.width <= 0
            or self.height <= 0
            or self.width % 64
            or self.height % 64
            or self.num_frames <= 0
            or (self.num_frames - 1) % 8
            or self.frame_rate <= 0
            or self.num_inference_steps < 2
            or not self.review_focus
        ):
            raise QualityCorpusError(f"invalid_case_values:{self.case_id}")


@dataclass(frozen=True)
class QualityCorpus:
    name: str
    path: Path
    sha256: str
    cases: tuple[QualityCase, ...]
    requirements: dict[str, Any]


def load_corpus(path: Path) -> QualityCorpus:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if int(raw["schema_version"]) != CORPUS_SCHEMA_VERSION:
            raise QualityCorpusError("unsupported_schema")
        name = str(raw["name"]).strip()
        requirements = dict(raw["requirements"])
        cases = tuple(QualityCase.from_dict(item) for item in raw["cases"])
    except QualityCorpusError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise QualityCorpusError("unreadable_manifest") from error
    if not name or not cases:
        raise QualityCorpusError("empty_corpus")
    ids = [item.case_id for item in cases]
    if len(ids) != len(set(ids)):
        raise QualityCorpusError("duplicate_case_id")
    _validate_diversity(cases, requirements)
    return QualityCorpus(name=name, path=path, sha256=file_sha256(path), cases=cases, requirements=requirements)


def _validate_diversity(cases: tuple[QualityCase, ...], requirements: dict[str, Any]) -> None:
    try:
        minimum_cases = int(requirements["minimum_case_count"])
        minimum_seeds = int(requirements["minimum_unique_seeds"])
        minimum_resolutions = int(requirements["minimum_unique_resolutions"])
        minimum_frames = int(requirements["minimum_unique_frame_counts"])
        required_focus = {str(item) for item in requirements["required_review_focus"]}
    except (KeyError, TypeError, ValueError) as error:
        raise QualityCorpusError("invalid_requirements") from error
    observed_focus = {focus for case in cases for focus in case.review_focus}
    checks = (
        len(cases) >= minimum_cases,
        len({case.seed for case in cases}) >= minimum_seeds,
        len({(case.width, case.height) for case in cases}) >= minimum_resolutions,
        len({case.num_frames for case in cases}) >= minimum_frames,
        required_focus.issubset(observed_focus),
    )
    if not all(checks):
        raise QualityCorpusError("insufficient_diversity")


def validation_report(corpus: QualityCorpus) -> dict[str, Any]:
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "component": "quality_corpus_structure",
        "corpus": {"name": corpus.name, "path": str(corpus.path), "sha256": corpus.sha256},
        "case_count": len(corpus.cases),
        "unique_seeds": len({case.seed for case in corpus.cases}),
        "unique_resolutions": len({(case.width, case.height) for case in corpus.cases}),
        "unique_frame_counts": len({case.num_frames for case in corpus.cases}),
        "review_focus": sorted({focus for case in corpus.cases for focus in case.review_focus}),
        "passed": True,
        "scope": "structure_only_not_quality_acceptance",
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
