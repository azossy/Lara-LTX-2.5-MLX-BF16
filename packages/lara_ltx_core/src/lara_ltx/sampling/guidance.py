"""MLX classifier-free, spatio-temporal and modality guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx

from lara_ltx.errors import LaraError

from .perturbations import Perturbation, PerturbationConfig, PerturbationType

CONDITIONED_PASS = "cond"
UNCONDITIONED_PASS = "uncond"
PERTURBED_PASS = "ptb"
ISOLATED_MODALITY_PASS = "mod"


@dataclass(frozen=True)
class MultiModalGuiderParams:
    cfg_scale: float = 1.0
    stg_scale: float = 0.0
    stg_blocks: tuple[int, ...] = field(default_factory=tuple)
    rescale_scale: float = 0.0
    modality_scale: float = 1.0
    skip_step: int = 0

    def __post_init__(self) -> None:
        scales = (self.cfg_scale, self.stg_scale, self.rescale_scale, self.modality_scale)
        if (
            not all(math.isfinite(value) for value in scales)
            or not 0.0 <= self.rescale_scale <= 1.0
            or not isinstance(self.skip_step, int)
            or isinstance(self.skip_step, bool)
            or self.skip_step < 0
            or any(not isinstance(block, int) or isinstance(block, bool) or block < 0 for block in self.stg_blocks)
        ):
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_configuration"})


@dataclass(frozen=True)
class MultiModalGuider:
    params: MultiModalGuiderParams

    def _required(self, value: mx.array | None, *, enabled: bool, name: str, cond: mx.array) -> mx.array:
        if value is None and enabled:
            raise LaraError("LARA-SAMPLING-002", details={"reason": f"missing_{name}_prediction"})
        return cond if value is None else value

    def calculate(
        self,
        cond: mx.array,
        *,
        uncond: mx.array | None = None,
        perturbed: mx.array | None = None,
        isolated: mx.array | None = None,
    ) -> mx.array:
        if cond.ndim < 2:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_conditioned_prediction"})
        uncond = self._required(uncond, enabled=self.do_unconditional_generation(), name="unconditional", cond=cond)
        perturbed = self._required(perturbed, enabled=self.do_perturbed_generation(), name="perturbed", cond=cond)
        isolated = self._required(isolated, enabled=self.do_isolated_modality_generation(), name="isolated", cond=cond)
        dtype = cond.dtype
        cond_f = cond.astype(mx.float32)
        prediction = (
            cond_f
            + (self.params.cfg_scale - 1.0) * (cond_f - uncond.astype(mx.float32))
            + self.params.stg_scale * (cond_f - perturbed.astype(mx.float32))
            + (self.params.modality_scale - 1.0) * (cond_f - isolated.astype(mx.float32))
        )
        if self.params.rescale_scale:
            factor = mx.std(cond_f, ddof=1) / mx.std(prediction, ddof=1)
            factor = self.params.rescale_scale * factor + (1.0 - self.params.rescale_scale)
            prediction = prediction * factor
        return prediction.astype(dtype)

    def do_unconditional_generation(self) -> bool:
        return not math.isclose(self.params.cfg_scale, 1.0)

    def do_perturbed_generation(self) -> bool:
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step: int) -> bool:
        return self.params.skip_step != 0 and step % (self.params.skip_step + 1) != 0


@dataclass(frozen=True)
class GuidanceBatchPlan:
    pass_names: tuple[str, ...]
    perturbations: tuple[PerturbationConfig, ...]

    def split(self, outputs: mx.array, *, original_batch_size: int) -> dict[str, mx.array]:
        if original_batch_size <= 0 or outputs.shape[0] != len(self.pass_names) * original_batch_size:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_output_batch"})
        return {
            name: outputs[index * original_batch_size : (index + 1) * original_batch_size]
            for index, name in enumerate(self.pass_names)
        }

    def repeated_perturbations(self, *, original_batch_size: int) -> tuple[PerturbationConfig, ...]:
        if original_batch_size <= 0:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_batch_size"})
        return tuple(config for config in self.perturbations for _ in range(original_batch_size))


def build_guidance_batch_plan(
    video: MultiModalGuider,
    audio: MultiModalGuider,
    *,
    force_unconditional: bool = False,
) -> GuidanceBatchPlan:
    names = [CONDITIONED_PASS]
    configs = [PerturbationConfig()]
    if video.do_unconditional_generation() or audio.do_unconditional_generation() or force_unconditional:
        names.append(UNCONDITIONED_PASS)
        configs.append(PerturbationConfig())
    stg: list[Perturbation] = []
    if video.do_perturbed_generation():
        stg.append(
            Perturbation(
                PerturbationType.SKIP_VIDEO_SELF_ATTN,
                blocks=video.params.stg_blocks,
            )
        )
    if audio.do_perturbed_generation():
        stg.append(
            Perturbation(
                PerturbationType.SKIP_AUDIO_SELF_ATTN,
                blocks=audio.params.stg_blocks,
            )
        )
    if stg:
        names.append(PERTURBED_PASS)
        configs.append(PerturbationConfig(tuple(stg)))
    if video.do_isolated_modality_generation() or audio.do_isolated_modality_generation():
        names.append(ISOLATED_MODALITY_PASS)
        configs.append(
            PerturbationConfig(
                (
                    Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                    Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                )
            )
        )
    return GuidanceBatchPlan(tuple(names), tuple(configs))


def calculate_guided_output(
    guider: MultiModalGuider,
    plan: GuidanceBatchPlan,
    outputs: mx.array,
    *,
    original_batch_size: int,
) -> mx.array:
    split = plan.split(outputs, original_batch_size=original_batch_size)
    return guider.calculate(
        split[CONDITIONED_PASS],
        uncond=split.get(UNCONDITIONED_PASS),
        perturbed=split.get(PERTURBED_PASS),
        isolated=split.get(ISOLATED_MODALITY_PASS),
    )
