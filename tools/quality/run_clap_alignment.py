#!/usr/bin/env python3
"""Measure prompt/audio alignment with a pinned LAION CLAP checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.quality.corpus import ERROR_CODE, file_sha256, message, write_json_atomic
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from corpus import ERROR_CODE, file_sha256, message, write_json_atomic

MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
DEFAULT_DEVICE = "cuda"


class ClapAlignmentError(ValueError):
    """Stable quality-tool failure with a documented recovery action."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"[{ERROR_CODE}] {message('clap_failed', reason=reason)}")


@dataclass(frozen=True)
class AlignmentCase:
    case_id: str
    audio_path: Path
    prompt: str


@dataclass(frozen=True)
class AlignmentManifest:
    name: str
    path: Path
    sha256: str
    cases: tuple[AlignmentCase, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--enable-fusion", action="store_true")
    return parser.parse_args()


def load_alignment_manifest(path: Path) -> AlignmentManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if int(raw["schema_version"]) != MANIFEST_SCHEMA_VERSION:
            raise ClapAlignmentError("unsupported_manifest_schema")
        name = str(raw["name"]).strip()
        cases = tuple(
            AlignmentCase(
                case_id=str(item["id"]).strip(),
                audio_path=Path(str(item["audio_path"])),
                prompt=str(item["prompt"]).strip(),
            )
            for item in raw["cases"]
        )
    except ClapAlignmentError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ClapAlignmentError("unreadable_manifest") from error
    if not name or not cases:
        raise ClapAlignmentError("empty_manifest")
    identifiers = [item.case_id for item in cases]
    if any(not item.case_id or not item.prompt for item in cases):
        raise ClapAlignmentError("empty_case_field")
    if len(identifiers) != len(set(identifiers)):
        raise ClapAlignmentError("duplicate_case_id")
    missing = [str(item.audio_path) for item in cases if not item.audio_path.is_file()]
    if missing:
        raise ClapAlignmentError(f"missing_audio:{missing[0]}")
    return AlignmentManifest(name=name, path=path, sha256=file_sha256(path), cases=cases)


def _normalized_diagonal(audio_embeddings: Any, text_embeddings: Any) -> list[float]:
    import numpy as np

    audio = np.asarray(audio_embeddings, dtype=np.float64)
    text = np.asarray(text_embeddings, dtype=np.float64)
    if audio.ndim != 2 or text.ndim != 2 or audio.shape != text.shape or audio.shape[0] == 0:
        raise ClapAlignmentError("invalid_embedding_shape")
    audio_norm = np.linalg.norm(audio, axis=1, keepdims=True)
    text_norm = np.linalg.norm(text, axis=1, keepdims=True)
    if np.any(audio_norm == 0) or np.any(text_norm == 0):
        raise ClapAlignmentError("zero_norm_embedding")
    similarities = np.sum((audio / audio_norm) * (text / text_norm), axis=1)
    return [float(value) for value in similarities]


def run_alignment(arguments: argparse.Namespace) -> dict[str, object]:
    try:
        import laion_clap
        import numpy as np
        import torch
    except ImportError as error:
        raise ClapAlignmentError(f"missing_dependency:{error.name}") from error

    manifest = load_alignment_manifest(arguments.manifest)
    if not arguments.checkpoint.is_file():
        raise ClapAlignmentError("missing_checkpoint")
    model = laion_clap.CLAP_Module(enable_fusion=arguments.enable_fusion, device=arguments.device)
    # PyTorch 2.6+ defaults to weights-only checkpoint loading. The pinned
    # official CLAP file contains a NumPy scalar in its metadata, so allow only
    # that specific non-Tensor type instead of disabling safe loading globally.
    numpy_scalar = np._core.multiarray.scalar
    safe_checkpoint_types = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_checkpoint_types):
        model.load_ckpt(ckpt=str(arguments.checkpoint), verbose=False)
    audio_paths = [str(item.audio_path) for item in manifest.cases]
    prompts = [item.prompt for item in manifest.cases]
    audio_embeddings = model.get_audio_embedding_from_filelist(audio_paths, use_tensor=False)
    text_embeddings = model.get_text_embedding(prompts, use_tensor=False)
    similarities = _normalized_diagonal(audio_embeddings, text_embeddings)
    results = [
        {
            "case_id": item.case_id,
            "audio_path": str(item.audio_path),
            "audio_sha256": file_sha256(item.audio_path),
            "cosine_similarity": score,
        }
        for item, score in zip(manifest.cases, similarities, strict=True)
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": "laion_clap_prompt_audio_alignment",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"name": manifest.name, "path": str(manifest.path), "sha256": manifest.sha256},
        "checkpoint": {"path": str(arguments.checkpoint), "sha256": file_sha256(arguments.checkpoint)},
        "environment": {
            "device": arguments.device,
            "fusion_enabled": arguments.enable_fusion,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "case_count": len(results),
        "mean_cosine_similarity": float(np.mean(similarities)),
        "results": results,
        "passed": True,
        "scope": "prompt_audio_alignment_only",
    }


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_alignment(arguments)
        write_json_atomic(arguments.output, report)
    except (ClapAlignmentError, OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 2
    print(message("clap_completed", count=report["case_count"], report=arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
