#!/usr/bin/env python3
"""Run the checkpoint-backed resident AV denoiser across configured LoRA stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.models import build_lora_pairs
from lara_ltx.models.transformer_block import AUDIO_DIMENSION, VIDEO_DIMENSION
from lara_ltx.sampling import (
    MultiModalGuider,
    MultiModalGuiderParams,
    Res2sLatentState,
    build_guidance_batch_plan,
)
from lara_ltx.transformer import (
    DenoiserModalityConditioning,
    GuidedDenoiserConditioning,
    GuidedResidentAVDenoiser,
    ResidentAVDenoiser,
    ResidentCheckpointTransformerBlocks,
    load_production_transformer_input,
    load_production_transformer_output,
)
from lara_ltx.transformer.runtime import PRODUCTION_INPUT_CHANNELS

VIDEO_POSITION_AXIS_COUNT = 3
AUDIO_POSITION_AXIS_COUNT = 1


@dataclass(frozen=True)
class SyntheticDenoiserInputs:
    video: Res2sLatentState
    audio: Res2sLatentState
    video_conditioning: DenoiserModalityConditioning
    audio_conditioning: DenoiserModalityConditioning


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--input-mapping", required=True, type=Path)
    parser.add_argument("--output-mapping", required=True, type=Path)
    parser.add_argument("--block-count", required=True, type=int)
    parser.add_argument("--token-count", required=True, type=int)
    parser.add_argument("--context-token-count", required=True, type=int)
    parser.add_argument("--sigma", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage-lora-strength", required=True, action="append", type=float)
    parser.add_argument("--guidance-config", type=Path)
    parser.add_argument("--require-all-lora-pairs", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _positions(axis_count: int, token_count: int) -> mx.array:
    positions = np.zeros((1, axis_count, token_count, 2), dtype=np.float32)
    positions[..., 1] = 1.0
    return mx.array(positions)


def _state(token_count: int) -> Res2sLatentState:
    latent = mx.random.normal((1, token_count, PRODUCTION_INPUT_CHANNELS), dtype=mx.bfloat16)
    mask = mx.ones((1, token_count, 1), dtype=mx.bfloat16)
    mx.eval(latent, mask)
    return Res2sLatentState(latent=latent, denoise_mask=mask, clean_latent=latent)


def _conditioning(
    token_count: int,
    context_token_count: int,
    axis_count: int,
    context_dimension: int,
) -> DenoiserModalityConditioning:
    context = mx.random.normal((1, context_token_count, context_dimension), dtype=mx.bfloat16)
    mx.eval(context)
    return DenoiserModalityConditioning(
        positions=_positions(axis_count, token_count),
        context=context,
        context_mask=mx.ones((1, context_token_count), dtype=mx.int32),
        attention_mask=mx.ones((1, token_count, token_count), dtype=mx.float32),
        keyframes_mask=(
            mx.zeros((1, token_count, 1), dtype=mx.int32)
            if axis_count == VIDEO_POSITION_AXIS_COUNT
            else None
        ),
    )


def _inputs(token_count: int, context_token_count: int) -> SyntheticDenoiserInputs:
    return SyntheticDenoiserInputs(
        video=_state(token_count),
        audio=_state(token_count),
        video_conditioning=_conditioning(
            token_count,
            context_token_count,
            VIDEO_POSITION_AXIS_COUNT,
            VIDEO_DIMENSION,
        ),
        audio_conditioning=_conditioning(
            token_count,
            context_token_count,
            AUDIO_POSITION_AXIS_COUNT,
            AUDIO_DIMENSION,
        ),
    )


def _output_report(value: mx.array) -> dict[str, object]:
    array = np.asarray(value.astype(mx.float32))
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _load_guidance(
    path: Path | None,
) -> tuple[MultiModalGuider, MultiModalGuider, int, dict[str, object]] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-SAMPLING-002", details={"reason": "unreadable_guidance_config"}) from error
    video_payload = payload.get("video")
    audio_payload = payload.get("audio")
    step_index = payload.get("step_index")
    if not isinstance(video_payload, dict) or not isinstance(audio_payload, dict) or not isinstance(step_index, int):
        raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_config_structure"})

    def guider(values: dict[str, object]) -> MultiModalGuider:
        normalized = dict(values)
        blocks = normalized.get("stg_blocks")
        if isinstance(blocks, list):
            normalized["stg_blocks"] = tuple(blocks)
        try:
            return MultiModalGuider(MultiModalGuiderParams(**normalized))
        except TypeError as error:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_config_fields"}) from error

    video = guider(video_payload)
    audio = guider(audio_payload)
    return video, audio, step_index, payload


def main() -> int:
    arguments = parse_arguments()
    input_mapping = json.loads(arguments.input_mapping.read_text(encoding="utf-8"))
    output_mapping = json.loads(arguments.output_mapping.read_text(encoding="utf-8"))
    lora_pairs = build_lora_pairs(arguments.lora_checkpoint, arguments.transformer_checkpoint)
    guidance = _load_guidance(arguments.guidance_config)
    mx.random.seed(arguments.seed)
    inputs = _inputs(arguments.token_count, arguments.context_token_count)

    initial_strength = arguments.stage_lora_strength[0]
    blocks = ResidentCheckpointTransformerBlocks(
        checkpoint=arguments.transformer_checkpoint,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=initial_strength,
        block_count=arguments.block_count,
    )
    resident_bytes = int(mx.get_active_memory())
    fused_block_pairs = sum(record.fused_lora_pair_count for record in blocks.load_records)
    stages: list[dict[str, object]] = []
    for stage_index, strength in enumerate(arguments.stage_lora_strength):
        mx.reset_peak_memory()
        blocks.set_lora_strength(strength)
        transition_peak_bytes = int(mx.get_peak_memory())
        mx.reset_peak_memory()
        input_processor, fused_input_pairs = load_production_transformer_input(
            checkpoint=arguments.transformer_checkpoint,
            mapping=input_mapping,
            lora_checkpoint=arguments.lora_checkpoint,
            lora_strength=strength,
            lora_pairs=lora_pairs,
        )
        output_heads, fused_output_pairs = load_production_transformer_output(
            checkpoint=arguments.transformer_checkpoint,
            mapping=output_mapping,
            lora_checkpoint=arguments.lora_checkpoint,
            lora_strength=strength,
            lora_pairs=lora_pairs,
        )
        if guidance is None:
            denoiser = ResidentAVDenoiser(
                input_processor=input_processor,
                blocks=blocks,
                expected_block_count=arguments.block_count,
                output_heads=output_heads,
                video_conditioning=inputs.video_conditioning,
                audio_conditioning=inputs.audio_conditioning,
            )
            guidance_passes = None
        else:
            video_guider, audio_guider, step_index, _ = guidance
            denoiser = GuidedResidentAVDenoiser(
                input_processor=input_processor,
                blocks=blocks,
                expected_block_count=arguments.block_count,
                output_heads=output_heads,
                video_conditioning=GuidedDenoiserConditioning(
                    conditioned=inputs.video_conditioning,
                    unconditioned=_conditioning(
                        arguments.token_count,
                        arguments.context_token_count,
                        VIDEO_POSITION_AXIS_COUNT,
                        VIDEO_DIMENSION,
                    ),
                ),
                audio_conditioning=GuidedDenoiserConditioning(
                    conditioned=inputs.audio_conditioning,
                    unconditioned=_conditioning(
                        arguments.token_count,
                        arguments.context_token_count,
                        AUDIO_POSITION_AXIS_COUNT,
                        AUDIO_DIMENSION,
                    ),
                ),
                video_guider=video_guider,
                audio_guider=audio_guider,
            )
            denoiser.set_step_index(step_index)
            guidance_passes = list(build_guidance_batch_plan(video_guider, audio_guider).pass_names)
        video, audio = denoiser(inputs.video, inputs.audio, arguments.sigma)
        assert video is not None and audio is not None
        stage = {
            "index": stage_index,
            "lora_strength": strength,
            "transition_peak_bytes": transition_peak_bytes,
            "execution_peak_bytes": int(mx.get_peak_memory()),
            "fused_input_pairs": fused_input_pairs,
            "fused_output_pairs": fused_output_pairs,
            "guidance_passes": guidance_passes,
            "outputs": {
                "video": _output_report(video),
                "audio": _output_report(audio),
            },
        }
        stages.append(stage)
        del denoiser, input_processor, output_heads, video, audio
        mx.clear_cache()

    finite = all(
        bool(output["finite"])
        for stage in stages
        for output in stage["outputs"].values()
    )
    covered_lora_pairs = fused_block_pairs + int(stages[0]["fused_input_pairs"]) + int(
        stages[0]["fused_output_pairs"]
    )
    all_lora_pairs_covered = covered_lora_pairs == len(lora_pairs)
    passed = (
        finite
        and len(blocks.blocks) == arguments.block_count
        and (all_lora_pairs_covered or not arguments.require_all_lora_pairs)
    )
    report = {
        "schema_version": 1,
        "component": "integrated_resident_av_denoiser",
        "configuration": {
            "block_count": arguments.block_count,
            "token_count": arguments.token_count,
            "context_token_count": arguments.context_token_count,
            "sigma": arguments.sigma,
            "seed": arguments.seed,
            "require_all_lora_pairs": arguments.require_all_lora_pairs,
            "guidance_config": guidance[3] if guidance is not None else None,
        },
        "resident_bytes": resident_bytes,
        "fused_block_pairs": fused_block_pairs,
        "covered_lora_pairs": covered_lora_pairs,
        "available_lora_pairs": len(lora_pairs),
        "all_lora_pairs_covered": all_lora_pairs_covered,
        "stages": stages,
        "passed": passed,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
