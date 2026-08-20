#!/usr/bin/env python3
"""Validate one-load resident transformer execution and stage LoRA transition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.transformer import ResidentCheckpointTransformerBlocks, TransformerStream
from lara_ltx.transformer.runtime import run_transformer_block_sequence

STAGE_ONE_LORA_STRENGTH = 0.25
STAGE_TWO_LORA_STRENGTH = 0.5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _stream(reference: dict[str, np.ndarray], prefix: str) -> TransformerStream:
    def array(suffix: str) -> mx.array:
        return mx.array(reference[f"{prefix}_{suffix}"], dtype=mx.bfloat16)

    return TransformerStream(
        x=array("input"),
        context=array("context"),
        timesteps=array("timesteps"),
        prompt_timestep=array("prompt_timestep"),
        cross_scale_shift_timestep=array("cross_scale_shift_timestep"),
        cross_gate_timestep=array("cross_gate_timestep"),
    )


def _output(value: mx.array) -> dict[str, object]:
    array = np.asarray(value.astype(mx.float32))
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _run(
    blocks: ResidentCheckpointTransformerBlocks,
    reference: dict[str, np.ndarray],
) -> tuple[dict[str, object], int]:
    mx.reset_peak_memory()
    result = run_transformer_block_sequence(_stream(reference, "video"), _stream(reference, "audio"), blocks)
    assert result.video is not None and result.audio is not None
    outputs = {"video": _output(result.video.x), "audio": _output(result.audio.x)}
    return outputs, int(mx.get_peak_memory())


def main() -> int:
    arguments = parse_arguments()
    with np.load(arguments.reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    blocks = ResidentCheckpointTransformerBlocks(
        checkpoint=arguments.transformer_checkpoint,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=STAGE_ONE_LORA_STRENGTH,
    )
    resident_bytes = int(mx.get_active_memory())
    stage_one, stage_one_peak = _run(blocks, reference)
    mx.reset_peak_memory()
    blocks.set_lora_strength(STAGE_TWO_LORA_STRENGTH)
    transition_peak = int(mx.get_peak_memory())
    stage_two, stage_two_peak = _run(blocks, reference)
    passed = all(bool(stream["finite"]) for stage in (stage_one, stage_two) for stream in stage.values())
    report = {
        "schema_version": 1,
        "component": "resident_av_transformer",
        "block_count": len(blocks.blocks),
        "resident_bytes": resident_bytes,
        "stage_one": {
            "lora_strength": STAGE_ONE_LORA_STRENGTH,
            "execution_peak_bytes": stage_one_peak,
            "outputs": stage_one,
        },
        "stage_transition_peak_bytes": transition_peak,
        "stage_two": {
            "lora_strength": STAGE_TWO_LORA_STRENGTH,
            "execution_peak_bytes": stage_two_peak,
            "outputs": stage_two,
        },
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
