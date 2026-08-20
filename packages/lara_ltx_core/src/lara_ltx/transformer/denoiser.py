"""Integrated resident AV velocity and denoised-output model."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import mlx.core as mx

from lara_ltx.errors import LaraError

from .blocks import AVTransformerBlock
from .input import AVTransformerInputPreprocessor, PreparedTransformerInput, TransformerModalityInput
from .output import AVTransformerOutput
from .runtime import run_transformer_block_sequence

if TYPE_CHECKING:
    from lara_ltx.sampling.guidance import GuidanceBatchPlan, MultiModalGuider
    from lara_ltx.sampling.perturbations import BatchedPerturbationConfig
    from lara_ltx.sampling.res2s import Res2sLatentState


@dataclass(frozen=True)
class DenoiserModalityConditioning:
    """Static per-modality inputs shared across denoising evaluations."""

    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: mx.array | None = None
    attention_mask: mx.array | None = None
    keyframes_mask: mx.array | None = None


@dataclass(frozen=True)
class GuidedDenoiserConditioning:
    """Conditioned and optional unconditional inputs for one modality."""

    conditioned: DenoiserModalityConditioning
    unconditioned: DenoiserModalityConditioning | None = None


def _token_mask(state: Res2sLatentState) -> mx.array:
    mask = state.denoise_mask
    while mask.ndim > 2 and mask.shape[-1] == 1:
        mask = mx.squeeze(mask, axis=-1)
    if mask.ndim != 2 or mask.shape != state.latent.shape[:2]:
        raise LaraError("LARA-TENSOR-029", details={"reason": "invalid_denoise_mask"})
    return mask


def _bind_modality(
    state: Res2sLatentState,
    conditioning: DenoiserModalityConditioning,
    sigma: float,
) -> TransformerModalityInput:
    if (
        state.latent.ndim != 3
        or conditioning.context.ndim != 3
        or conditioning.context.shape[0] != state.latent.shape[0]
    ):
        raise LaraError("LARA-TENSOR-029", details={"reason": "invalid_latent_or_context"})
    sigma_array = mx.full((state.latent.shape[0],), sigma, dtype=state.latent.dtype)
    timesteps = _token_mask(state).astype(state.latent.dtype) * sigma_array[:, None]
    return TransformerModalityInput(
        latent=state.latent,
        sigma=sigma_array,
        timesteps=timesteps,
        positions=conditioning.positions,
        context=conditioning.context,
        enabled=conditioning.enabled,
        context_mask=conditioning.context_mask,
        attention_mask=conditioning.attention_mask,
        keyframes_mask=conditioning.keyframes_mask,
    )


def _materialize_prepared(prepared: PreparedTransformerInput | None) -> None:
    if prepared is None:
        return
    stream = prepared.stream
    values = [stream.x, stream.context, stream.timesteps, prepared.embedded_timestep]
    for optional in (
        stream.prompt_timestep,
        stream.cross_scale_shift_timestep,
        stream.cross_gate_timestep,
        stream.context_mask,
        stream.self_attention_mask,
    ):
        if optional is not None:
            values.append(optional)
    for positional in (stream.positional_embeddings, stream.cross_positional_embeddings):
        if positional is not None:
            values.extend(positional)
    mx.eval(*values)


class ResidentAVDenoiser:
    """One in-process input-to-resident-blocks-to-output denoiser."""

    def __init__(
        self,
        *,
        input_processor: AVTransformerInputPreprocessor,
        blocks: Iterable[tuple[int, AVTransformerBlock]],
        expected_block_count: int,
        output_heads: AVTransformerOutput,
        video_conditioning: DenoiserModalityConditioning | None,
        audio_conditioning: DenoiserModalityConditioning | None,
    ) -> None:
        if video_conditioning is None and audio_conditioning is None:
            raise LaraError("LARA-TENSOR-029", details={"reason": "missing_conditioning"})
        if expected_block_count <= 0:
            raise LaraError("LARA-TENSOR-029", details={"reason": "invalid_expected_block_count"})
        self.input_processor = input_processor
        self.blocks = blocks
        self.expected_block_count = expected_block_count
        self.output_heads = output_heads
        self.video_conditioning = video_conditioning
        self.audio_conditioning = audio_conditioning

    def __call__(
        self,
        video: Res2sLatentState | None,
        audio: Res2sLatentState | None,
        sigma: float,
        *,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[mx.array | None, mx.array | None]:
        if not math.isfinite(sigma) or sigma < 0:
            raise LaraError("LARA-TENSOR-029", details={"reason": "invalid_sigma"})
        if (video is None) != (self.video_conditioning is None) or (audio is None) != (self.audio_conditioning is None):
            raise LaraError("LARA-TENSOR-029", details={"reason": "modality_conditioning_mismatch"})
        video_input = _bind_modality(video, self.video_conditioning, sigma) if video is not None else None
        audio_input = _bind_modality(audio, self.audio_conditioning, sigma) if audio is not None else None
        prepared_video, prepared_audio = self.input_processor.prepare(video_input, audio_input)
        _materialize_prepared(prepared_video)
        _materialize_prepared(prepared_audio)
        sequence = run_transformer_block_sequence(
            prepared_video.stream if prepared_video is not None else None,
            prepared_audio.stream if prepared_audio is not None else None,
            self.blocks,
            perturbations=perturbations,
        )
        if sequence.completed_block_count != self.expected_block_count:
            raise LaraError(
                "LARA-TENSOR-029",
                details={
                    "reason": (
                        f"block_count_mismatch:expected={self.expected_block_count}:"
                        f"actual={sequence.completed_block_count}"
                    )
                },
            )
        video_velocity = (
            self.output_heads.video(sequence.video.x, prepared_video.embedded_timestep)
            if sequence.video is not None and prepared_video is not None
            else None
        )
        audio_velocity = (
            self.output_heads.audio(sequence.audio.x, prepared_audio.embedded_timestep)
            if sequence.audio is not None and prepared_audio is not None
            else None
        )
        video_denoised = (
            video.latent.astype(mx.float32)
            - video_velocity.astype(mx.float32) * video_input.timesteps.astype(mx.float32)[..., None]
            if video is not None and video_velocity is not None and video_input is not None
            else None
        )
        audio_denoised = (
            audio.latent.astype(mx.float32)
            - audio_velocity.astype(mx.float32) * audio_input.timesteps.astype(mx.float32)[..., None]
            if audio is not None and audio_velocity is not None and audio_input is not None
            else None
        )
        video_denoised = (
            video_denoised.astype(video.latent.dtype) if video_denoised is not None and video is not None else None
        )
        audio_denoised = (
            audio_denoised.astype(audio.latent.dtype) if audio_denoised is not None and audio is not None else None
        )
        outputs = tuple(value for value in (video_denoised, audio_denoised) if value is not None)
        mx.eval(*outputs)
        return video_denoised, audio_denoised


def _repeat_batch(value: mx.array, count: int) -> mx.array:
    return mx.concatenate(tuple(value for _ in range(count)), axis=0)


def _repeat_state(state: Res2sLatentState, pass_count: int) -> Res2sLatentState:
    from lara_ltx.sampling.res2s import Res2sLatentState

    return Res2sLatentState(
        latent=_repeat_batch(state.latent, pass_count),
        denoise_mask=_repeat_batch(state.denoise_mask, pass_count),
        clean_latent=_repeat_batch(state.clean_latent, pass_count),
    )


def _concatenate_optional(values: tuple[mx.array | None, ...], *, field: str) -> mx.array | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise LaraError("LARA-SAMPLING-002", details={"reason": f"mixed_guidance_{field}"})
    arrays = tuple(value for value in values if value is not None)
    if len({tuple(value.shape[1:]) for value in arrays}) != 1:
        raise LaraError("LARA-SAMPLING-002", details={"reason": f"mismatched_guidance_{field}"})
    return mx.concatenate(arrays, axis=0)


def _batch_conditioning(
    conditioning: GuidedDenoiserConditioning,
    plan: GuidanceBatchPlan,
    *,
    original_batch_size: int,
    needs_unconditional: bool,
) -> DenoiserModalityConditioning:
    from lara_ltx.sampling.guidance import UNCONDITIONED_PASS

    selected: list[DenoiserModalityConditioning] = []
    for pass_name in plan.pass_names:
        current = (
            conditioning.unconditioned
            if pass_name == UNCONDITIONED_PASS and needs_unconditional
            else conditioning.conditioned
        )
        if current is None:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "missing_unconditional_conditioning"})
        if not current.enabled or current.context.shape[0] != original_batch_size:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_conditioning_batch"})
        selected.append(current)
    return DenoiserModalityConditioning(
        positions=_concatenate_optional(tuple(value.positions for value in selected), field="positions"),
        context=_concatenate_optional(tuple(value.context for value in selected), field="context"),
        context_mask=_concatenate_optional(tuple(value.context_mask for value in selected), field="context_mask"),
        attention_mask=_concatenate_optional(
            tuple(value.attention_mask for value in selected),
            field="attention_mask",
        ),
        keyframes_mask=_concatenate_optional(
            tuple(value.keyframes_mask for value in selected),
            field="keyframes_mask",
        ),
    )


class GuidedResidentAVDenoiser:
    """Batch CFG/STG/AV-isolation passes around one resident denoiser call."""

    def __init__(
        self,
        *,
        input_processor: AVTransformerInputPreprocessor,
        blocks: Iterable[tuple[int, AVTransformerBlock]],
        expected_block_count: int,
        output_heads: AVTransformerOutput,
        video_conditioning: GuidedDenoiserConditioning | None,
        audio_conditioning: GuidedDenoiserConditioning | None,
        video_guider: MultiModalGuider,
        audio_guider: MultiModalGuider,
    ) -> None:
        if video_conditioning is None and audio_conditioning is None:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "missing_guidance_conditioning"})
        self.input_processor = input_processor
        self.blocks = blocks
        self.expected_block_count = expected_block_count
        self.output_heads = output_heads
        self.video_conditioning = video_conditioning
        self.audio_conditioning = audio_conditioning
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self.step_index = 0
        self._last_denoised_video: mx.array | None = None
        self._last_denoised_audio: mx.array | None = None

    def set_step_index(self, step_index: int) -> None:
        if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_step_index"})
        self.step_index = step_index

    def reset_guidance_state(self) -> None:
        self.step_index = 0
        self._last_denoised_video = None
        self._last_denoised_audio = None

    @staticmethod
    def _original_batch_size(
        video: Res2sLatentState | None,
        audio: Res2sLatentState | None,
    ) -> int:
        sizes = tuple(state.latent.shape[0] for state in (video, audio) if state is not None)
        if not sizes or len(set(sizes)) != 1 or sizes[0] <= 0:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_guidance_latent_batch"})
        return sizes[0]

    def __call__(
        self,
        video: Res2sLatentState | None,
        audio: Res2sLatentState | None,
        sigma: float,
    ) -> tuple[mx.array | None, mx.array | None]:
        from lara_ltx.sampling.guidance import build_guidance_batch_plan, calculate_guided_output
        from lara_ltx.sampling.perturbations import BatchedPerturbationConfig

        if (video is None) != (self.video_conditioning is None) or (audio is None) != (self.audio_conditioning is None):
            raise LaraError("LARA-SAMPLING-002", details={"reason": "guidance_modality_mismatch"})
        original_batch_size = self._original_batch_size(video, audio)
        video_skipped = video is not None and self.video_guider.should_skip_step(self.step_index)
        audio_skipped = audio is not None and self.audio_guider.should_skip_step(self.step_index)
        if (video is None or video_skipped) and (audio is None or audio_skipped):
            if (video is not None and self._last_denoised_video is None) or (
                audio is not None and self._last_denoised_audio is None
            ):
                raise LaraError("LARA-SAMPLING-002", details={"reason": "missing_skipped_guidance_prediction"})
            return self._last_denoised_video, self._last_denoised_audio
        plan = build_guidance_batch_plan(self.video_guider, self.audio_guider)
        pass_count = len(plan.pass_names)
        batched_video = _repeat_state(video, pass_count) if video is not None else None
        batched_audio = _repeat_state(audio, pass_count) if audio is not None else None
        video_conditioning = (
            replace(
                _batch_conditioning(
                    self.video_conditioning,
                    plan,
                    original_batch_size=original_batch_size,
                    needs_unconditional=self.video_guider.do_unconditional_generation(),
                ),
                enabled=not video_skipped,
            )
            if self.video_conditioning is not None
            else None
        )
        audio_conditioning = (
            replace(
                _batch_conditioning(
                    self.audio_conditioning,
                    plan,
                    original_batch_size=original_batch_size,
                    needs_unconditional=self.audio_guider.do_unconditional_generation(),
                ),
                enabled=not audio_skipped,
            )
            if self.audio_conditioning is not None
            else None
        )
        latent_dtype = video.latent.dtype if video is not None else audio.latent.dtype
        perturbations = BatchedPerturbationConfig(
            plan.repeated_perturbations(original_batch_size=original_batch_size),
            num_blocks=self.expected_block_count,
            dtype=latent_dtype,
        )
        denoiser = ResidentAVDenoiser(
            input_processor=self.input_processor,
            blocks=self.blocks,
            expected_block_count=self.expected_block_count,
            output_heads=self.output_heads,
            video_conditioning=video_conditioning,
            audio_conditioning=audio_conditioning,
        )
        video_outputs, audio_outputs = denoiser(
            batched_video,
            batched_audio,
            sigma,
            perturbations=perturbations,
        )
        guided_video = (
            self._last_denoised_video
            if video_skipped
            else calculate_guided_output(
                self.video_guider,
                plan,
                video_outputs,
                original_batch_size=original_batch_size,
            )
            if video_outputs is not None
            else None
        )
        guided_audio = (
            self._last_denoised_audio
            if audio_skipped
            else calculate_guided_output(
                self.audio_guider,
                plan,
                audio_outputs,
                original_batch_size=original_batch_size,
            )
            if audio_outputs is not None
            else None
        )
        materialized = tuple(value for value in (guided_video, guided_audio) if value is not None)
        mx.eval(*materialized)
        if not video_skipped:
            self._last_denoised_video = guided_video
        if not audio_skipped:
            self._last_denoised_audio = guided_audio
        return guided_video, guided_audio
