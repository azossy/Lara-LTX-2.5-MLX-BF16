#!/usr/bin/env python3
"""Replay CUDA denoiser outputs through the MLX res_2s sampler only."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors
from lara_ltx.sampling import Res2sLatentState, Res2sSampler

CONFIG_SCHEMA_VERSION = 1
MODALITIES = ("video", "audio")
STAGE_CONFIGURATIONS = (("stage_1", "stage_one"), ("stage_2", "stage_two"))
SAMPLER_FIELDS = ("eta", "bongmath", "bongmath_max_iterations", "noise_seed", "noise_seed_substep")
SIGMA_ABSOLUTE_TOLERANCE = 1e-7


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-reference", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unreadable_sampler_replay_config"}) from error
    if not isinstance(value, dict) or value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_sampler_replay_config"})
    return value


def _required(value: dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    result = value.get(key)
    if not isinstance(result, expected) or (expected in (int, float) and isinstance(result, bool)):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_sampler_replay_field:{key}"})
    return result


def _tensor(reference: dict[str, np.ndarray], key: str) -> np.ndarray:
    value = reference.get(key)
    if value is None:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_sampler_replay_tensor:{key}"})
    return value


def _schedule(reference: dict[str, np.ndarray], config: dict[str, Any], stage_name: str) -> mx.array:
    references = _required(config, "reference_keys", dict)
    source_key = _required(references, f"{stage_name}_sigmas", str)
    source = reference.get(source_key)
    indices = _required(config, f"{stage_name}_schedule_indices", list)
    if source is None:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_sampler_replay_tensor:{source_key}"})
    try:
        selected = np.asarray([source[int(index)] for index in indices], dtype=np.float32)
    except (IndexError, TypeError, ValueError) as error:
        raise LaraError(
            "LARA-RUNTIME-008",
            details={"reason": f"invalid_sampler_replay_schedule:{stage_name}"},
        ) from error
    if selected.ndim != 1 or selected.size < 2:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_sampler_replay_schedule:{stage_name}"})
    return mx.array(selected, dtype=mx.float32)


def _expected_denoiser_calls(schedule: mx.array) -> int:
    values = np.asarray(schedule.astype(mx.float32))
    full_step_count = values.size - 1
    return full_step_count * 2 + int(values[-1] == 0.0)


def _state(value: np.ndarray) -> Res2sLatentState:
    latent = mx.array(value, dtype=mx.bfloat16)
    mask = mx.ones((*latent.shape[:-1], 1), dtype=mx.float32)
    mx.eval(latent, mask)
    return Res2sLatentState(latent=latent, denoise_mask=mask, clean_latent=latent)


@dataclass
class CapturedSdeNoise:
    reference: dict[str, np.ndarray]
    stage: str
    step_count: int

    def __post_init__(self) -> None:
        self.counts = {"substep": 0, "step": 0}

    def __call__(self, value: mx.array, stream: str) -> mx.array:
        if stream not in self.counts:
            raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_sampler_replay_stream:{stream}"})
        call_index = self.counts[stream]
        self.counts[stream] += 1
        modality = MODALITIES[call_index % len(MODALITIES)]
        step_index = call_index // len(MODALITIES)
        key = f"{self.stage}_sde_{stream}_{step_index:02d}_{modality}"
        noise = self.reference.get(key)
        if noise is None or tuple(noise.shape) != tuple(value.shape):
            raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_sampler_replay_tensor:{key}"})
        return mx.array(noise, dtype=mx.float32)

    def assert_exhausted(self) -> None:
        expected = self.step_count * len(MODALITIES)
        if any(count != expected for count in self.counts.values()):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "sampler_replay_noise_not_exhausted"})


class CapturedDenoiser:
    def __init__(self, reference: dict[str, np.ndarray], stage: str, expected_calls: int) -> None:
        self.reference = reference
        self.stage = stage
        self.expected_calls = expected_calls
        self.call_index = 0
        self.comparisons: list[dict[str, Any]] = []
        self.step_indices: list[int] = []

    def set_step_index(self, step_index: int) -> None:
        self.step_indices.append(step_index)

    def _key(self, modality: str, boundary: str) -> str:
        return f"{self.stage}_denoiser_call_{self.call_index:02d}_{modality}_{boundary}"

    def __call__(
        self,
        video: Res2sLatentState | None,
        audio: Res2sLatentState | None,
        sigma: float,
    ) -> tuple[mx.array | None, mx.array | None]:
        sigma_key = f"{self.stage}_denoiser_call_{self.call_index:02d}_sigma"
        captured_sigma = self.reference.get(sigma_key)
        if captured_sigma is None or not math.isclose(
            sigma,
            float(np.asarray(captured_sigma).item()),
            rel_tol=0.0,
            abs_tol=SIGMA_ABSOLUTE_TOLERANCE,
        ):
            raise LaraError("LARA-RUNTIME-008", details={"reason": f"sampler_replay_sigma_mismatch:{sigma_key}"})
        outputs: list[mx.array | None] = []
        for modality, state in zip(MODALITIES, (video, audio), strict=True):
            if state is None:
                outputs.append(None)
                continue
            input_key = self._key(modality, "input")
            output_key = self._key(modality, "output")
            captured_input = self.reference.get(input_key)
            captured_output = self.reference.get(output_key)
            if captured_input is None or captured_output is None:
                missing = input_key if captured_input is None else output_key
                raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_sampler_replay_tensor:{missing}"})
            metrics = compare_tensors(
                input_key,
                captured_input.astype(np.float32),
                np.asarray(state.latent.astype(mx.float32)),
            )
            self.comparisons.append(
                {
                    "call_index": self.call_index,
                    "modality": modality,
                    "metrics": metrics.to_dict(),
                }
            )
            outputs.append(mx.array(captured_output, dtype=mx.bfloat16))
        self.call_index += 1
        return outputs[0], outputs[1]

    def assert_exhausted(self) -> None:
        if self.call_index != self.expected_calls:
            raise LaraError("LARA-RUNTIME-008", details={"reason": "sampler_replay_denoiser_not_exhausted"})


def _sampler(config: dict[str, Any], key: str, noise: CapturedSdeNoise) -> Res2sSampler:
    values = _required(config, key, dict)
    arguments = {name: values[name] for name in SAMPLER_FIELDS if name in values}
    try:
        return Res2sSampler(**arguments, noise_fn=noise)
    except (TypeError, ValueError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_sampler_replay_settings:{key}"}) from error


def replay_stage(
    reference: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    stage: str,
    config_prefix: str,
) -> dict[str, Any]:
    schedule = _schedule(reference, config, config_prefix)
    expected_calls = _expected_denoiser_calls(schedule)
    denoiser = CapturedDenoiser(reference, stage, expected_calls)
    noise = CapturedSdeNoise(reference, stage, schedule.shape[0] - 1)
    states = {
        modality: _state(_tensor(reference, f"{stage}_denoiser_call_00_{modality}_input")) for modality in MODALITIES
    }
    sampler = _sampler(config, f"{config_prefix}_sampler", noise)
    video, audio = sampler.sample(schedule, states["video"], states["audio"], denoiser)
    if video is None or audio is None:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_sampler_replay_output:{stage}"})
    denoiser.assert_exhausted()
    noise.assert_exhausted()
    return {
        "stage": stage,
        "schedule_length": int(schedule.shape[0]),
        "denoiser_call_count": expected_calls,
        "step_indices": denoiser.step_indices,
        "comparisons": denoiser.comparisons,
    }


def build_report(reference: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    stages = [
        replay_stage(reference, config, stage=stage, config_prefix=config_prefix)
        for stage, config_prefix in STAGE_CONFIGURATIONS
    ]
    acceptance = _required(config, "acceptance", dict)
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    comparisons = [comparison for stage in stages for comparison in stage["comparisons"]]
    for comparison in comparisons:
        metrics = comparison["metrics"]
        comparison["passed"] = (
            metrics["nan_count"] == 0
            and metrics["inf_count"] == 0
            and metrics["normalized_rmse"] <= maximum_nrmse
            and metrics["cosine_similarity"] >= minimum_cosine
        )
    first_failure = next((comparison for comparison in comparisons if not comparison["passed"]), None)
    return {
        "schema_version": 1,
        "component": "cuda_output_injected_sampler_replay",
        "acceptance": acceptance,
        "stages": stages,
        "first_failure": first_failure,
        "passed": first_failure is None,
    }


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    try:
        with np.load(arguments.cuda_reference) as archive:
            reference = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unreadable_sampler_replay_reference"}) from error
    report = build_report(reference, config)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
