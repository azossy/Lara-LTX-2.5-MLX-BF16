#!/usr/bin/env python3
"""Run one configured checkpoint-backed HQ Stage-1/Stage-2 sampling smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors
from lara_ltx.pipeline import (
    CheckpointStageModuleLoader,
    TwoStageContexts,
    TwoStageSamplingConfig,
    TwoStageSamplingRuntime,
)
from lara_ltx.sampling import (
    AudioLatentLayout,
    MultiModalGuider,
    MultiModalGuiderParams,
    Res2sLatentState,
    Res2sSampler,
    VideoLatentLayout,
    apply_gaussian_noise,
    create_audio_state,
    create_video_state,
    unpatchify_audio,
    unpatchify_video,
)
from lara_ltx.transformer import ResidentCheckpointTransformerBlocks
from lara_ltx.video_vae import load_spatial_video_upscaler

CONFIG_SCHEMA_VERSION = 1
NOISE_SEQUENCE_LENGTH = 4


@dataclass(frozen=True)
class CapturedNoise:
    value: mx.array
    expected_scale: float


class CapturedNoiseSequence:
    """Replay captured Gaussian draws while allowing newly sampled latent inputs."""

    def __init__(self, entries: tuple[CapturedNoise, ...]) -> None:
        if len(entries) != NOISE_SEQUENCE_LENGTH:
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_captured_noise_count"})
        self.entries = entries
        self.index = 0

    def __call__(self, state: Res2sLatentState, noise_scale: float) -> Res2sLatentState:
        if self.index >= len(self.entries):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "captured_noise_exhausted"})
        entry = self.entries[self.index]
        self.index += 1
        if not math.isclose(noise_scale, entry.expected_scale, rel_tol=0.0, abs_tol=1e-7):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "captured_noise_scale_mismatch"})
        return apply_gaussian_noise(state, entry.value, noise_scale)

    def assert_exhausted(self) -> None:
        if self.index != len(self.entries):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "captured_noise_not_consumed"})


class CapturedSdeNoiseSequence:
    """Replay the normalized CUDA SDE tensors in official stream/modality order."""

    _modalities = ("video", "audio")

    def __init__(self, reference: dict[str, np.ndarray], *, stage: str, step_count: int) -> None:
        self._reference = reference
        self._stage = stage
        self._step_count = step_count
        self._stream_counts = {"substep": 0, "step": 0}

    def __call__(self, value: mx.array, stream: str) -> mx.array:
        if stream not in self._stream_counts:
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_sde_noise_stream"})
        call_index = self._stream_counts[stream]
        self._stream_counts[stream] += 1
        modality = self._modalities[call_index % len(self._modalities)]
        step_index = call_index // len(self._modalities)
        key = f"{self._stage}_sde_{stream}_{step_index:02d}_{modality}"
        noise = self._reference.get(key)
        if noise is None or tuple(noise.shape) != tuple(value.shape):
            raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_{key}"})
        return mx.array(noise, dtype=mx.float32)

    def assert_exhausted(self) -> None:
        expected = self._step_count * len(self._modalities)
        if any(count != expected for count in self._stream_counts.values()):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "captured_sde_noise_not_consumed"})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--upscaler-checkpoint", required=True, type=Path)
    parser.add_argument("--video-vae-checkpoint", required=True, type=Path)
    parser.add_argument("--input-mapping", required=True, type=Path)
    parser.add_argument("--output-mapping", required=True, type=Path)
    parser.add_argument("--upscaler-mapping", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unreadable_sampling_configuration"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_sampling_configuration"})
    return value


def _required(config: dict[str, Any], key: str, expected: type) -> Any:
    value = config.get(key)
    if not isinstance(value, expected) or (expected in (int, float) and isinstance(value, bool)):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_config_{key}"})
    return value


def _schedule(reference: dict[str, np.ndarray], keys: dict[str, str], config: dict[str, Any], stage: str) -> mx.array:
    source = reference[keys[f"{stage}_sigmas"]]
    indices = _required(config, f"{stage}_schedule_indices", list)
    try:
        selected = np.asarray([source[int(index)] for index in indices], dtype=np.float32)
    except (IndexError, TypeError, ValueError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_{stage}_schedule_indices"}) from error
    return mx.array(selected, dtype=mx.float32)


def _video_layout(shape_source: np.ndarray, frame_rate: float) -> VideoLatentLayout:
    batch, channels, frames, height, width = shape_source.shape
    return VideoLatentLayout(
        batch=int(batch),
        channels=int(channels),
        frames=int(frames),
        height=int(height),
        width=int(width),
        frame_rate=frame_rate,
    )


def _audio_layout(shape_source: np.ndarray) -> AudioLatentLayout:
    batch, channels, frames, mel_bins = shape_source.shape
    return AudioLatentLayout(
        batch=int(batch),
        channels=int(channels),
        frames=int(frames),
        mel_bins=int(mel_bins),
    )


def _guider(config: dict[str, Any], key: str) -> MultiModalGuider:
    values = dict(_required(config, key, dict))
    blocks = values.get("stg_blocks")
    if isinstance(blocks, list):
        values["stg_blocks"] = tuple(blocks)
    try:
        return MultiModalGuider(MultiModalGuiderParams(**values))
    except TypeError as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_{key}"}) from error


def _sampler(config: dict[str, Any], key: str, *, noise_fn: CapturedSdeNoiseSequence) -> Res2sSampler:
    values = dict(_required(config, key, dict))
    values["noise_fn"] = noise_fn
    try:
        return Res2sSampler(**values)
    except TypeError as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_{key}"}) from error


def _captured_noise(input_value: np.ndarray, output_value: np.ndarray, scale: float) -> CapturedNoise:
    if not 0.0 < scale <= 1.0 or input_value.shape != output_value.shape:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_captured_noise_boundary"})
    noise = (output_value.astype(np.float32) - (1.0 - scale) * input_value.astype(np.float32)) / scale
    return CapturedNoise(mx.array(noise, dtype=mx.bfloat16), scale)


def _output_report(value: mx.array) -> dict[str, object]:
    array = np.asarray(value.astype(mx.float32))
    return {
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_sampling_config_schema"})
    keys = _required(config, "reference_keys", dict)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in keys.items()):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_reference_keys"})
    with np.load(arguments.cuda_reference) as archive:
        reference = {name: archive[name] for name in archive.files}
    if not set(keys.values()).issubset(reference):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "missing_reference_tensor"})

    frame_rate = float(_required(config, "frame_rate", (int, float)))
    block_count = int(_required(config, "block_count", int))
    stage_one_strength = float(_required(config, "stage_one_lora_strength", (int, float)))
    stage_two_strength = float(_required(config, "stage_two_lora_strength", (int, float)))
    stage_one_sigmas = _schedule(reference, keys, config, "stage_one")
    stage_two_sigmas = _schedule(reference, keys, config, "stage_two")
    stage_one_sde_noise = CapturedSdeNoiseSequence(
        reference,
        stage="stage_1",
        step_count=stage_one_sigmas.shape[0] - 1,
    )
    stage_two_sde_noise = CapturedSdeNoiseSequence(
        reference,
        stage="stage_2",
        step_count=stage_two_sigmas.shape[0] - 1,
    )
    stage_one_scale = float(stage_one_sigmas[0].item())
    stage_two_scale = float(stage_two_sigmas[0].item())
    stage_one_video_layout = _video_layout(reference[keys["stage_one_video_layout"]], frame_rate)
    stage_two_video_layout = _video_layout(reference[keys["stage_two_video_layout"]], frame_rate)
    audio_layout = _audio_layout(reference[keys["stage_one_audio_layout"]])
    initial_video_tokens = mx.array(reference[keys["stage_one_video_input"]], dtype=mx.bfloat16)
    initial_audio_tokens = mx.array(reference[keys["stage_one_audio_input"]], dtype=mx.bfloat16)
    initial_video = unpatchify_video(initial_video_tokens, stage_one_video_layout)
    initial_audio = unpatchify_audio(initial_audio_tokens, audio_layout)
    noiser = CapturedNoiseSequence(
        (
            _captured_noise(
                reference[keys["stage_one_video_input"]],
                reference[keys["stage_one_video_output"]],
                stage_one_scale,
            ),
            _captured_noise(
                reference[keys["stage_one_audio_input"]],
                reference[keys["stage_one_audio_output"]],
                stage_one_scale,
            ),
            _captured_noise(
                reference[keys["stage_two_video_input"]],
                reference[keys["stage_two_video_output"]],
                stage_two_scale,
            ),
            _captured_noise(
                reference[keys["stage_two_audio_input"]],
                reference[keys["stage_two_audio_output"]],
                stage_two_scale,
            ),
        )
    )
    input_mapping = _read_json(arguments.input_mapping)
    output_mapping = _read_json(arguments.output_mapping)
    upscaler_mapping = _read_json(arguments.upscaler_mapping)
    blocks = ResidentCheckpointTransformerBlocks(
        checkpoint=arguments.transformer_checkpoint,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=stage_one_strength,
        block_count=block_count,
    )
    resident_bytes = int(mx.get_active_memory())
    upscaler = load_spatial_video_upscaler(
        upscaler_checkpoint=arguments.upscaler_checkpoint,
        video_vae_checkpoint=arguments.video_vae_checkpoint,
        mapping=upscaler_mapping,
    )
    module_loader = CheckpointStageModuleLoader(
        transformer_checkpoint=arguments.transformer_checkpoint,
        lora_checkpoint=arguments.lora_checkpoint,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        lora_pairs=blocks.lora_pairs,
    )
    runtime = TwoStageSamplingRuntime(
        blocks=blocks,
        expected_block_count=block_count,
        stage_module_loader=module_loader,
        set_lora_strength=blocks.set_lora_strength,
        upscaler=upscaler,
        initial_noiser=noiser,
        stage_one_sampler=_sampler(config, "stage_one_sampler", noise_fn=stage_one_sde_noise),
        stage_two_sampler=_sampler(config, "stage_two_sampler", noise_fn=stage_two_sde_noise),
        config=TwoStageSamplingConfig(stage_one_strength, stage_two_strength),
    )
    mx.reset_peak_memory()
    result = runtime.run(
        stage_one_video=create_video_state(stage_one_video_layout, initial_latent=initial_video),
        stage_one_audio=create_audio_state(audio_layout, initial_latent=initial_audio),
        stage_one_video_layout=stage_one_video_layout,
        stage_two_video_layout=stage_two_video_layout,
        audio_layout=audio_layout,
        contexts=TwoStageContexts(
            video_positive=mx.array(reference[keys["video_positive"]], dtype=mx.bfloat16),
            video_negative=mx.array(reference[keys["video_negative"]], dtype=mx.bfloat16),
            audio_positive=mx.array(reference[keys["audio_positive"]], dtype=mx.bfloat16),
            audio_negative=mx.array(reference[keys["audio_negative"]], dtype=mx.bfloat16),
        ),
        video_guider=_guider(config, "video_guidance"),
        audio_guider=_guider(config, "audio_guidance"),
        stage_one_sigmas=stage_one_sigmas,
        stage_two_sigmas=stage_two_sigmas,
    )
    noiser.assert_exhausted()
    stage_one_sde_noise.assert_exhausted()
    stage_two_sde_noise.assert_exhausted()
    outputs = {"video": _output_report(result.video), "audio": _output_report(result.audio)}
    comparisons = {
        "video": compare_tensors(
            "stage_two_video_latent",
            reference[keys["final_video"]].astype(np.float32),
            np.asarray(result.video.astype(mx.float32)),
        ).to_dict(),
        "audio": compare_tensors(
            "stage_one_audio_latent",
            reference[keys["final_audio"]].astype(np.float32),
            np.asarray(result.audio.astype(mx.float32)),
        ).to_dict(),
    }
    acceptance = _required(config, "acceptance", dict)
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    acceptance_checks = {
        modality: metrics["normalized_rmse"] <= maximum_nrmse
        and metrics["cosine_similarity"] >= minimum_cosine
        for modality, metrics in comparisons.items()
    }
    passed = all(bool(output["finite"]) for output in outputs.values()) and all(acceptance_checks.values())
    report = {
        "schema_version": 1,
        "component": "checkpoint_two_stage_sampling_smoke",
        "configuration": config,
        "resident_bytes": resident_bytes,
        "peak_bytes": int(mx.get_peak_memory()),
        "resident_block_count": len(blocks.blocks),
        "final_lora_strength": blocks.lora_strength,
        "outputs": outputs,
        "comparisons": comparisons,
        "acceptance": {
            "thresholds": acceptance,
            "checks": acceptance_checks,
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
