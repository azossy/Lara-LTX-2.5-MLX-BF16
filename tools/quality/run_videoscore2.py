#!/usr/bin/env python3
"""Run pinned VideoScore2 inference reproducibly over a video manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
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
DEFAULT_INFER_FPS = 2.0
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 25_082_110
DEFAULT_VIDEO_READER = "decord"
SCORE_LABELS = (
    "visual quality",
    "text-to-video alignment",
    "physical/common-sense consistency",
)
QUERY_TEMPLATE = Template(
    """
You are an expert for evaluating AI-generated videos from three dimensions:
(1) visual quality \N{EN DASH} clarity, smoothness, artifacts;
(2) text-to-video alignment \N{EN DASH} fidelity to the prompt;
(3) physical/common-sense consistency \N{EN DASH} naturalness and physics plausibility.

Video prompt: $t2v_prompt

Please output in this format:
visual quality: <v_score>;
text-to-video alignment: <t_score>,
physical/common-sense consistency: <p_score>
"""
)


class VideoScoreError(ValueError):
    """Stable quality-tool failure with a documented recovery action."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"[{ERROR_CODE}] {message('videoscore_failed', reason=reason)}")


@dataclass(frozen=True)
class VideoScoreCase:
    case_id: str
    video_path: Path
    prompt: str


@dataclass(frozen=True)
class VideoScoreManifest:
    name: str
    path: Path
    sha256: str
    cases: tuple[VideoScoreCase, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--infer-fps", type=float, default=DEFAULT_INFER_FPS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--video-reader", default=DEFAULT_VIDEO_READER)
    return parser.parse_args()


def load_manifest(path: Path) -> VideoScoreManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if int(raw["schema_version"]) != MANIFEST_SCHEMA_VERSION:
            raise VideoScoreError("unsupported_manifest_schema")
        name = str(raw["name"]).strip()
        cases = tuple(
            VideoScoreCase(
                case_id=str(item["id"]).strip(),
                video_path=Path(str(item["video_path"])),
                prompt=str(item["prompt"]).strip(),
            )
            for item in raw["cases"]
        )
    except VideoScoreError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise VideoScoreError("unreadable_manifest") from error
    if not name or not cases:
        raise VideoScoreError("empty_manifest")
    identifiers = [item.case_id for item in cases]
    if any(not item.case_id or not item.prompt for item in cases):
        raise VideoScoreError("empty_case_field")
    if len(identifiers) != len(set(identifiers)):
        raise VideoScoreError("duplicate_case_id")
    missing = [str(item.video_path) for item in cases if not item.video_path.is_file()]
    if missing:
        raise VideoScoreError(f"missing_video:{missing[0]}")
    return VideoScoreManifest(name=name, path=path, sha256=file_sha256(path), cases=cases)


def _final_answer_section(output_text: str) -> str:
    """Exclude chain-of-thought text when the model emits a tagged rationale."""

    marker = "</think>"
    if marker in output_text.lower():
        marker_index = output_text.lower().rfind(marker)
        return output_text[marker_index + len(marker) :]
    return output_text


def _score_pattern(label: str) -> re.Pattern[str]:
    flexible_label = re.escape(label).replace(r"\ ", r"\s+")
    description_dashes = re.escape("\N{EN DASH}\N{EM DASH}-")
    return re.compile(
        rf"(?:\(\s*\d+\s*\)\s*)?{flexible_label}"
        rf"(?:\s*[{description_dashes}]\s*[^:\n]+)?\s*:\s*([1-5])\b",
        flags=re.IGNORECASE,
    )


def parse_hard_scores(output_text: str) -> tuple[int | None, int | None, int | None]:
    answer = _final_answer_section(output_text)
    matches = tuple(_score_pattern(label).search(answer) for label in SCORE_LABELS)
    if any(match is None for match in matches):
        return None, None, None
    return tuple(int(match.group(1)) for match in matches if match is not None)


def _find_score_token_index(label: str, tokenizer: Any, generated_ids: list[int]) -> int:
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    answer = _final_answer_section(generated_text)
    answer_offset = len(generated_text) - len(answer)
    match = _score_pattern(label).search(answer)
    if not match:
        return -1
    target = generated_text[: answer_offset + match.end(1)]
    for index in range(len(generated_ids)):
        if tokenizer.decode(generated_ids[: index + 1], skip_special_tokens=False) == target:
            return index
    return -1


def _soft_score(hard_value: int | None, token_index: int, scores: Any, tokenizer: Any) -> float | None:
    if hard_value is None or token_index < 0:
        return None
    import numpy as np
    import torch

    logits = scores[token_index][0]
    candidates: list[tuple[int, float]] = []
    for score in range(1, 6):
        token_ids = tokenizer.encode(str(score), add_special_tokens=False)
        if len(token_ids) == 1:
            probability = float(np.exp(torch.log_softmax(logits, dim=-1)[token_ids[0]].item()))
            candidates.append((score, probability))
    if not candidates:
        return None
    probabilities = [item[1] for item in candidates]
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    best_index = probabilities.index(max(probabilities))
    return round(candidates[best_index][0] * probabilities[best_index] / total, 4)


def _reset_seed(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_evaluation(arguments: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", arguments.video_reader)
    try:
        import numpy as np
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
    except ImportError as error:
        raise VideoScoreError(f"missing_dependency:{error.name}") from error

    manifest = load_manifest(arguments.manifest)
    if arguments.infer_fps <= 0 or arguments.max_new_tokens <= 0 or arguments.temperature <= 0:
        raise VideoScoreError("invalid_inference_parameter")
    model = AutoModelForVision2Seq.from_pretrained(
        arguments.model,
        revision=arguments.model_revision,
        trust_remote_code=True,
    ).to(arguments.device)
    processor = AutoProcessor.from_pretrained(
        arguments.model,
        revision=arguments.model_revision,
        trust_remote_code=True,
    )
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        arguments.model,
        revision=arguments.model_revision,
        trust_remote_code=True,
        use_fast=False,
    )
    results: list[dict[str, object]] = []
    for item in manifest.cases:
        _reset_seed(arguments.seed, torch, np)
        user_prompt = QUERY_TEMPLATE.substitute(t2v_prompt=item.prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(item.video_path), "fps": arguments.infer_fps},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=arguments.infer_fps,
            padding=True,
            return_tensors="pt",
        ).to(arguments.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=arguments.max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=arguments.temperature,
        )
        input_length = inputs["input_ids"].shape[1]
        generated_ids = generated.sequences[0, input_length:].tolist()
        output_text = processor.batch_decode(
            generated.sequences[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        hard_scores = parse_hard_scores(output_text)
        soft_scores = tuple(
            _soft_score(
                hard_value,
                _find_score_token_index(label, tokenizer, generated_ids),
                generated.scores,
                tokenizer,
            )
            for label, hard_value in zip(SCORE_LABELS, hard_scores, strict=True)
        )
        results.append(
            {
                "case_id": item.case_id,
                "video_path": str(item.video_path),
                "video_sha256": file_sha256(item.video_path),
                "hard_scores": {
                    "visual_quality": hard_scores[0],
                    "text_alignment": hard_scores[1],
                    "physical_consistency": hard_scores[2],
                },
                "soft_scores": {
                    "visual_quality": soft_scores[0],
                    "text_alignment": soft_scores[1],
                    "physical_consistency": soft_scores[2],
                },
                "raw_model_output": output_text,
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": "videoscore2",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"name": manifest.name, "path": str(manifest.path), "sha256": manifest.sha256},
        "source_revision": arguments.source_revision,
        "model": {"repository": arguments.model, "revision": arguments.model_revision},
        "inference": {
            "device": arguments.device,
            "seed_reset_per_case": arguments.seed,
            "infer_fps": arguments.infer_fps,
            "max_new_tokens": arguments.max_new_tokens,
            "do_sample": True,
            "temperature": arguments.temperature,
            "video_reader": arguments.video_reader,
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "case_count": len(results),
        "results": results,
        "passed": all(all(value is not None for value in item["soft_scores"].values()) for item in results),
    }


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_evaluation(arguments)
        write_json_atomic(arguments.output, report)
        if not report["passed"]:
            raise VideoScoreError("unparseable_model_output")
    except (VideoScoreError, OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(message("videoscore_completed", count=report["case_count"], report=arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
