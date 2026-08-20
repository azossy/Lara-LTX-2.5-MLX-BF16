"""Integrated resident AV velocity and denoised-output model."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

from lara_ltx.errors import LaraError

from .blocks import AVTransformerBlock
from .input import AVTransformerInputPreprocessor, PreparedTransformerInput, TransformerModalityInput
from .output import AVTransformerOutput
from .runtime import run_transformer_block_sequence

if TYPE_CHECKING:
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
        or not conditioning.enabled
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
        if (video is None) != (self.video_conditioning is None) or (audio is None) != (
            self.audio_conditioning is None
        ):
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
            video_denoised.astype(video.latent.dtype)
            if video_denoised is not None and video is not None
            else None
        )
        audio_denoised = (
            audio_denoised.astype(audio.latent.dtype)
            if audio_denoised is not None and audio is not None
            else None
        )
        outputs = tuple(value for value in (video_denoised, audio_denoised) if value is not None)
        mx.eval(*outputs)
        return video_denoised, audio_denoised
