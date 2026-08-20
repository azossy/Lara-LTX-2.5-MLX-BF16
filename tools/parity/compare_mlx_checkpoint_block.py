#!/usr/bin/env python3
"""Compare MLX block-0 output from official mapped weights with CUDA golden data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.models import iter_component_weight_batches, transformer_block_target_shapes
from lara_ltx.parity.metrics import compare_tensors
from lara_ltx.transformer.blocks import AVTransformerBlock, TransformerStream, VideoTransformerConfig

VIDEO_DIMENSION = 4096
AUDIO_DIMENSION = 2048
ATTENTION_HEAD_COUNT = 32
VIDEO_ATTENTION_HEAD_DIMENSION = 128
AUDIO_ATTENTION_HEAD_DIMENSION = 64
MAX_NORMALIZED_RMSE = 2e-2
MIN_COSINE_SIMILARITY = 0.9999


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _array(reference: dict[str, np.ndarray], name: str) -> mx.array:
    return mx.array(reference[name], dtype=mx.bfloat16)


def _stream(reference: dict[str, np.ndarray], prefix: str) -> TransformerStream:
    return TransformerStream(
        x=_array(reference, f"{prefix}_input"),
        context=_array(reference, f"{prefix}_context"),
        timesteps=_array(reference, f"{prefix}_timesteps"),
        prompt_timestep=_array(reference, f"{prefix}_prompt_timestep"),
        cross_scale_shift_timestep=_array(reference, f"{prefix}_cross_scale_shift_timestep"),
        cross_gate_timestep=_array(reference, f"{prefix}_cross_gate_timestep"),
    )


def main() -> int:
    arguments = parse_arguments()
    mapping = json.loads(arguments.mapping.read_text(encoding="utf-8"))
    rules = mapping["rules"]
    if not isinstance(rules, list):
        raise LaraError("LARA-MODEL-014", details={"source_key": str(arguments.mapping)})
    block = AVTransformerBlock(
        video=VideoTransformerConfig(
            dim=VIDEO_DIMENSION,
            heads=ATTENTION_HEAD_COUNT,
            head_dim=VIDEO_ATTENTION_HEAD_DIMENSION,
            context_dim=VIDEO_DIMENSION,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            ff_bias=False,
        ),
        audio=VideoTransformerConfig(
            dim=AUDIO_DIMENSION,
            heads=ATTENTION_HEAD_COUNT,
            head_dim=AUDIO_ATTENTION_HEAD_DIMENSION,
            context_dim=AUDIO_DIMENSION,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            ff_bias=True,
        ),
    )
    active_memory_before_loading = int(mx.get_active_memory())
    mx.reset_peak_memory()
    batches = iter_component_weight_batches(
        (arguments.transformer_checkpoint,),
        rules,
        expected_target_shapes=transformer_block_target_shapes(),
    )
    for batch in batches:
        block.load_weights(batch)
        mx.eval(*[value for _, value in batch])
    weight_load_memory = {
        "active_before_bytes": active_memory_before_loading,
        "active_after_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }

    with np.load(arguments.cuda_reference) as reference:
        values = {name: reference[name] for name in reference.files}
    video_output, audio_output = block(
        _stream(values, "video"),
        _stream(values, "audio"),
    )
    assert video_output is not None and audio_output is not None
    mx.eval(video_output, audio_output)
    video_metrics = compare_tensors(
        "video_output",
        values["video_output"],
        np.asarray(video_output.astype(mx.float32)),
    )
    audio_metrics = compare_tensors(
        "audio_output",
        values["audio_output"],
        np.asarray(audio_output.astype(mx.float32)),
    )
    stream_metrics = (video_metrics, audio_metrics)
    passed = all(
        result.nan_count == 0
        and result.inf_count == 0
        and result.normalized_rmse <= MAX_NORMALIZED_RMSE
        and result.cosine_similarity >= MIN_COSINE_SIMILARITY
        for result in stream_metrics
    )
    report = {
        "schema_version": 1,
        "acceptance_criteria": {
            "max_normalized_rmse": MAX_NORMALIZED_RMSE,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "requires_finite_output": True,
        },
        "video": video_metrics.to_dict(),
        "audio": audio_metrics.to_dict(),
        "weight_load_memory": weight_load_memory,
        "passed": passed,
    }
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
