"""HQ two-stage sampling and resident-transformer lifecycle."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx

from lara_ltx.errors import LaraError
from lara_ltx.models import LoraPair, build_lora_pairs
from lara_ltx.sampling import (
    AudioLatentLayout,
    MultiModalGuider,
    PatchifiedLatentState,
    Res2sLatentState,
    Res2sSampler,
    VideoLatentLayout,
    prepare_stage_two_states,
    unpatchify_audio,
    unpatchify_video,
)
from lara_ltx.transformer import (
    AVTransformerBlock,
    AVTransformerInputPreprocessor,
    AVTransformerOutput,
    DenoiserModalityConditioning,
    GuidedDenoiserConditioning,
    GuidedResidentAVDenoiser,
    ResidentAVDenoiser,
    load_production_transformer_input,
    load_production_transformer_output,
)

from .profiling import RuntimePhase, RuntimeProfiler

StageModuleLoader = Callable[[float], tuple[AVTransformerInputPreprocessor, AVTransformerOutput]]
StageStrengthSetter = Callable[[float], None]
InitialNoiser = Callable[[Res2sLatentState, float], Res2sLatentState]
Upscaler = Callable[[mx.array], mx.array]
DenoiserObserver = Callable[
    [
        str,
        int,
        float,
        Res2sLatentState | None,
        Res2sLatentState | None,
        mx.array | None,
        mx.array | None,
    ],
    None,
]


class _ObservedDenoiser:
    """Transparent diagnostic wrapper that does not alter denoiser execution."""

    def __init__(self, denoiser: object, stage: str, observer: DenoiserObserver) -> None:
        self._denoiser = denoiser
        self._stage = stage
        self._observer = observer
        self._call_index = 0

    def set_step_index(self, step_index: int) -> None:
        callback = getattr(self._denoiser, "set_step_index", None)
        if callable(callback):
            callback(step_index)

    def __call__(
        self,
        video: Res2sLatentState | None,
        audio: Res2sLatentState | None,
        sigma: float,
    ) -> tuple[mx.array | None, mx.array | None]:
        callback = self._denoiser
        if not callable(callback):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_observed_denoiser"})
        video_output, audio_output = callback(video, audio, sigma)
        self._observer(
            self._stage,
            self._call_index,
            sigma,
            video,
            audio,
            video_output,
            audio_output,
        )
        self._call_index += 1
        return video_output, audio_output


@dataclass(frozen=True)
class TwoStageSamplingConfig:
    stage_one_lora_strength: float
    stage_two_lora_strength: float

    def __post_init__(self) -> None:
        strengths = (self.stage_one_lora_strength, self.stage_two_lora_strength)
        if any(not math.isfinite(value) or value < 0 for value in strengths):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_stage_lora_strength"})


@dataclass(frozen=True)
class TwoStageContexts:
    video_positive: mx.array
    audio_positive: mx.array
    video_negative: mx.array | None = None
    audio_negative: mx.array | None = None
    video_context_mask: mx.array | None = None
    audio_context_mask: mx.array | None = None


@dataclass(frozen=True)
class TwoStageLatentResult:
    video: mx.array
    audio: mx.array
    stage_one_video: Res2sLatentState
    stage_one_audio: Res2sLatentState
    stage_two_video: Res2sLatentState


@dataclass(frozen=True)
class TwoStageSamplingRequest:
    stage_one_video: PatchifiedLatentState
    stage_one_audio: PatchifiedLatentState
    stage_one_video_layout: VideoLatentLayout
    stage_two_video_layout: VideoLatentLayout
    audio_layout: AudioLatentLayout
    contexts: TwoStageContexts
    video_guider: MultiModalGuider
    audio_guider: MultiModalGuider
    stage_one_sigmas: mx.array
    stage_two_sigmas: mx.array


class CheckpointStageModuleLoader:
    """Load stage-local transformer input/output modules from reviewed mappings."""

    def __init__(
        self,
        *,
        transformer_checkpoint: Path,
        lora_checkpoint: Path,
        input_mapping: dict[str, object],
        output_mapping: dict[str, object],
        lora_pairs: tuple[LoraPair, ...] | None = None,
    ) -> None:
        self.transformer_checkpoint = transformer_checkpoint
        self.lora_checkpoint = lora_checkpoint
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.lora_pairs = lora_pairs or build_lora_pairs(lora_checkpoint, transformer_checkpoint)

    def __call__(self, strength: float) -> tuple[AVTransformerInputPreprocessor, AVTransformerOutput]:
        input_processor, _ = load_production_transformer_input(
            checkpoint=self.transformer_checkpoint,
            mapping=self.input_mapping,
            lora_checkpoint=self.lora_checkpoint,
            lora_strength=strength,
            lora_pairs=self.lora_pairs,
        )
        output_heads, _ = load_production_transformer_output(
            checkpoint=self.transformer_checkpoint,
            mapping=self.output_mapping,
            lora_checkpoint=self.lora_checkpoint,
            lora_strength=strength,
            lora_pairs=self.lora_pairs,
        )
        return input_processor, output_heads


def _conditioning(
    bundle: PatchifiedLatentState,
    context: mx.array,
    context_mask: mx.array | None,
) -> DenoiserModalityConditioning:
    return DenoiserModalityConditioning(
        positions=bundle.positions,
        context=context,
        context_mask=context_mask,
        attention_mask=bundle.attention_mask,
        keyframes_mask=bundle.keyframes_mask,
    )


def _guided_conditioning(
    bundle: PatchifiedLatentState,
    positive: mx.array,
    negative: mx.array | None,
    context_mask: mx.array | None,
) -> GuidedDenoiserConditioning:
    conditioned = _conditioning(bundle, positive, context_mask)
    unconditioned = _conditioning(bundle, negative, context_mask) if negative is not None else None
    return GuidedDenoiserConditioning(conditioned=conditioned, unconditioned=unconditioned)


class TwoStageSamplingRuntime:
    """Run the pinned HQ stage sequence while retaining one transformer block state."""

    def __init__(
        self,
        *,
        blocks: Iterable[tuple[int, AVTransformerBlock]],
        expected_block_count: int,
        stage_module_loader: StageModuleLoader,
        set_lora_strength: StageStrengthSetter,
        upscaler: Upscaler,
        initial_noiser: InitialNoiser,
        stage_one_sampler: Res2sSampler,
        stage_two_sampler: Res2sSampler,
        config: TwoStageSamplingConfig,
        denoiser_observer: DenoiserObserver | None = None,
        profiler: RuntimeProfiler | None = None,
    ) -> None:
        resident_blocks = tuple(blocks)
        if expected_block_count <= 0 or len(resident_blocks) != expected_block_count:
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_stage_block_count"})
        self.blocks = resident_blocks
        self.expected_block_count = expected_block_count
        self.stage_module_loader = stage_module_loader
        self.set_lora_strength = set_lora_strength
        self.upscaler = upscaler
        self.initial_noiser = initial_noiser
        self.stage_one_sampler = stage_one_sampler
        self.stage_two_sampler = stage_two_sampler
        self.config = config
        self.denoiser_observer = denoiser_observer
        self.profiler = profiler or RuntimeProfiler()

    @staticmethod
    def _first_sigma(sigmas: mx.array, stage: str) -> float:
        if sigmas.ndim != 1 or sigmas.shape[0] < 2:
            raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_{stage}_sigmas"})
        sigma = float(sigmas[0].item())
        if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
            raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_{stage}_initial_sigma"})
        return sigma

    def run(
        self,
        *,
        stage_one_video: PatchifiedLatentState,
        stage_one_audio: PatchifiedLatentState,
        stage_one_video_layout: VideoLatentLayout,
        stage_two_video_layout: VideoLatentLayout,
        audio_layout: AudioLatentLayout,
        contexts: TwoStageContexts,
        video_guider: MultiModalGuider,
        audio_guider: MultiModalGuider,
        stage_one_sigmas: mx.array,
        stage_two_sigmas: mx.array,
    ) -> TwoStageLatentResult:
        stage_one_sigma = self._first_sigma(stage_one_sigmas, "stage_one")
        stage_two_sigma = self._first_sigma(stage_two_sigmas, "stage_two")
        initial_video = replace(stage_one_video, state=self.initial_noiser(stage_one_video.state, stage_one_sigma))
        initial_audio = replace(stage_one_audio, state=self.initial_noiser(stage_one_audio.state, stage_one_sigma))
        with self.profiler.measure(RuntimePhase.SAMPLING_STAGE_ONE):
            self.set_lora_strength(self.config.stage_one_lora_strength)
            stage_one_input, stage_one_output = self.stage_module_loader(self.config.stage_one_lora_strength)
            stage_one_denoiser = GuidedResidentAVDenoiser(
                input_processor=stage_one_input,
                blocks=self.blocks,
                expected_block_count=self.expected_block_count,
                output_heads=stage_one_output,
                video_conditioning=_guided_conditioning(
                    initial_video,
                    contexts.video_positive,
                    contexts.video_negative,
                    contexts.video_context_mask,
                ),
                audio_conditioning=_guided_conditioning(
                    initial_audio,
                    contexts.audio_positive,
                    contexts.audio_negative,
                    contexts.audio_context_mask,
                ),
                video_guider=video_guider,
                audio_guider=audio_guider,
            )
            if self.denoiser_observer is not None:
                stage_one_denoiser = _ObservedDenoiser(
                    stage_one_denoiser,
                    "stage_1",
                    self.denoiser_observer,
                )
            sampled_video, sampled_audio = self.stage_one_sampler.sample(
                stage_one_sigmas,
                initial_video.state,
                initial_audio.state,
                stage_one_denoiser,
            )
            if sampled_video is None or sampled_audio is None:
                raise LaraError("LARA-RUNTIME-008", details={"reason": "missing_stage_one_output"})
            del stage_one_denoiser, stage_one_input, stage_one_output
            mx.clear_cache()

        with self.profiler.measure(RuntimePhase.LATENT_UPSCALE):
            stage_two_video, stage_two_audio = prepare_stage_two_states(
                sampled_video,
                sampled_audio,
                stage_one_video_layout=stage_one_video_layout,
                stage_two_video_layout=stage_two_video_layout,
                audio_layout=audio_layout,
                upscaler=self.upscaler,
                noiser=self.initial_noiser,
                noise_scale=stage_two_sigma,
            )
            mx.eval(stage_two_video.state.latent, stage_two_audio.state.latent)

        with self.profiler.measure(RuntimePhase.SAMPLING_STAGE_TWO):
            self.set_lora_strength(self.config.stage_two_lora_strength)
            stage_two_input, stage_two_output = self.stage_module_loader(self.config.stage_two_lora_strength)
            stage_two_denoiser = ResidentAVDenoiser(
                input_processor=stage_two_input,
                blocks=self.blocks,
                expected_block_count=self.expected_block_count,
                output_heads=stage_two_output,
                video_conditioning=_conditioning(
                    stage_two_video,
                    contexts.video_positive,
                    contexts.video_context_mask,
                ),
                audio_conditioning=_conditioning(
                    stage_two_audio,
                    contexts.audio_positive,
                    contexts.audio_context_mask,
                ),
            )
            if self.denoiser_observer is not None:
                stage_two_denoiser = _ObservedDenoiser(
                    stage_two_denoiser,
                    "stage_2",
                    self.denoiser_observer,
                )
            refined_video, _ = self.stage_two_sampler.sample(
                stage_two_sigmas,
                stage_two_video.state,
                stage_two_audio.state,
                stage_two_denoiser,
            )
            if refined_video is None:
                raise LaraError("LARA-RUNTIME-008", details={"reason": "missing_stage_two_video"})
            del stage_two_denoiser, stage_two_input, stage_two_output
            mx.clear_cache()
        video = unpatchify_video(refined_video.latent, stage_two_video_layout)
        audio = unpatchify_audio(sampled_audio.latent, audio_layout)
        mx.eval(video, audio)
        return TwoStageLatentResult(
            video=video,
            audio=audio,
            stage_one_video=sampled_video,
            stage_one_audio=sampled_audio,
            stage_two_video=refined_video,
        )

    def run_request(self, request: TwoStageSamplingRequest) -> TwoStageLatentResult:
        """Run a reusable typed request for top-level lifecycle composition."""

        return self.run(
            stage_one_video=request.stage_one_video,
            stage_one_audio=request.stage_one_audio,
            stage_one_video_layout=request.stage_one_video_layout,
            stage_two_video_layout=request.stage_two_video_layout,
            audio_layout=request.audio_layout,
            contexts=request.contexts,
            video_guider=request.video_guider,
            audio_guider=request.audio_guider,
            stage_one_sigmas=request.stage_one_sigmas,
            stage_two_sigmas=request.stage_two_sigmas,
        )
