"""Typed, package-owned configuration for the public local pipeline."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from lara_ltx.errors import LaraError
from lara_ltx.media import MediaEncodingConfig
from lara_ltx.sampling import MultiModalGuiderParams

PROFILE_RESOURCE_PACKAGE = "lara_ltx.resources"
DEFAULT_PROFILE_NAME = "hq.toml"


def _required(value: dict[str, Any], key: str) -> Any:
    try:
        return value[key]
    except KeyError as error:
        raise LaraError("LARA-PIPELINE-001", details={"reason": f"missing_{key}"}) from error


def _section(value: dict[str, Any], key: str) -> dict[str, Any]:
    section = _required(value, key)
    if not isinstance(section, dict):
        raise LaraError("LARA-PIPELINE-001", details={"reason": f"invalid_{key}"})
    return section


def _guidance(value: dict[str, Any]) -> MultiModalGuiderParams:
    try:
        return MultiModalGuiderParams(
            cfg_scale=float(_required(value, "cfg_scale")),
            stg_scale=float(_required(value, "stg_scale")),
            stg_blocks=tuple(int(item) for item in _required(value, "stg_blocks")),
            rescale_scale=float(_required(value, "rescale_scale")),
            modality_scale=float(_required(value, "modality_scale")),
            skip_step=int(_required(value, "skip_step")),
        )
    except (TypeError, ValueError) as error:
        raise LaraError("LARA-PIPELINE-001", details={"reason": "invalid_guidance"}) from error


@dataclass(frozen=True)
class CheckpointPaths:
    transformer: Path
    text_encoder: Path
    video_vae: Path
    audio_vae: Path
    spatial_upscaler: Path
    distilled_lora: Path


@dataclass(frozen=True)
class ModelProfile:
    repository_id: str
    revision: str
    relative_files: dict[str, str]

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        return tuple(self.relative_files.values())

    def resolve(self, root: Path) -> CheckpointPaths:
        values = {name: root / relative for name, relative in self.relative_files.items()}
        missing = [str(path) for path in values.values() if not path.is_file()]
        if missing:
            raise LaraError("LARA-PIPELINE-002", details={"path": missing[0]})
        return CheckpointPaths(**values)


@dataclass(frozen=True)
class SchedulerProfile:
    max_shift: float
    base_shift: float
    stretch: bool
    terminal: float


@dataclass(frozen=True)
class SamplerProfile:
    eta: float
    bongmath: bool
    bongmath_max_iterations: int
    substep_seed_offset: int


@dataclass(frozen=True)
class GenerationProfile:
    negative_prompt: str
    seed: int
    height: int
    width: int
    num_frames: int
    frame_rate: float
    text_max_length: int
    num_inference_steps: int
    block_count: int
    video_latent_channels: int
    audio_latent_channels: int
    audio_mel_bins: int
    audio_model_sample_rate: int
    audio_hop_length: int
    audio_latent_downsample_factor: int
    stage_one_lora_strength: float
    stage_two_lora_strength: float
    stage_two_sigmas: tuple[float, ...]
    scheduler: SchedulerProfile
    sampler: SamplerProfile
    video_guidance: MultiModalGuiderParams
    audio_guidance: MultiModalGuiderParams

    def with_overrides(
        self,
        *,
        seed: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        negative_prompt: str | None = None,
    ) -> "GenerationProfile":
        updated = replace(
            self,
            seed=self.seed if seed is None else seed,
            height=self.height if height is None else height,
            width=self.width if width is None else width,
            num_frames=self.num_frames if num_frames is None else num_frames,
            frame_rate=self.frame_rate if frame_rate is None else frame_rate,
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            negative_prompt=self.negative_prompt if negative_prompt is None else negative_prompt,
        )
        updated.validate()
        return updated

    def validate(self) -> None:
        positive_ints = (
            self.height,
            self.width,
            self.num_frames,
            self.text_max_length,
            self.num_inference_steps,
            self.block_count,
            self.video_latent_channels,
            self.audio_latent_channels,
            self.audio_mel_bins,
            self.audio_model_sample_rate,
            self.audio_hop_length,
            self.audio_latent_downsample_factor,
        )
        if (
            any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in positive_ints)
            or self.height % 64
            or self.width % 64
            or (self.num_frames - 1) % 8
            or self.num_inference_steps < 2
            or self.frame_rate <= 0
            or not self.negative_prompt.strip()
        ):
            raise LaraError("LARA-PIPELINE-003", details={"reason": "invalid_generation_dimensions"})


@dataclass(frozen=True)
class DecodeProfile:
    video_noise_seed_offset: int
    video_timesteps: tuple[float, ...]
    video_activation_budget_bytes: int


@dataclass(frozen=True)
class DownloadProfile:
    cache_environment_variable: str
    token_environment_variable: str


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    model: ModelProfile
    generation: GenerationProfile
    decode: DecodeProfile
    media: MediaEncodingConfig
    download: DownloadProfile


def load_pipeline_profile(path: Path | None = None) -> PipelineProfile:
    """Load a user profile or the versioned package HQ profile."""

    try:
        if path is None:
            resource = files(PROFILE_RESOURCE_PACKAGE).joinpath(DEFAULT_PROFILE_NAME)
            raw = tomllib.loads(resource.read_text(encoding="utf-8"))
        else:
            with Path(path).expanduser().open("rb") as handle:
                raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise LaraError("LARA-PIPELINE-001", details={"reason": "unreadable_profile"}) from error
    profile = _section(raw, "profile")
    if _required(profile, "schema_version") != 1:
        raise LaraError("LARA-PIPELINE-001", details={"reason": "unsupported_profile_schema"})
    model = _section(raw, "model")
    generation = _section(raw, "generation")
    scheduler = _section(generation, "scheduler")
    sampler = _section(generation, "sampler")
    decode = _section(raw, "decode")
    media = _section(raw, "media")
    download = _section(raw, "download")
    model_keys = ("transformer", "text_encoder", "video_vae", "audio_vae", "spatial_upscaler", "distilled_lora")
    generation_profile = GenerationProfile(
        negative_prompt=str(_required(generation, "negative_prompt")),
        seed=int(_required(generation, "seed")),
        height=int(_required(generation, "height")),
        width=int(_required(generation, "width")),
        num_frames=int(_required(generation, "num_frames")),
        frame_rate=float(_required(generation, "frame_rate")),
        text_max_length=int(_required(generation, "text_max_length")),
        num_inference_steps=int(_required(generation, "num_inference_steps")),
        block_count=int(_required(generation, "block_count")),
        video_latent_channels=int(_required(generation, "video_latent_channels")),
        audio_latent_channels=int(_required(generation, "audio_latent_channels")),
        audio_mel_bins=int(_required(generation, "audio_mel_bins")),
        audio_model_sample_rate=int(_required(generation, "audio_model_sample_rate")),
        audio_hop_length=int(_required(generation, "audio_hop_length")),
        audio_latent_downsample_factor=int(_required(generation, "audio_latent_downsample_factor")),
        stage_one_lora_strength=float(_required(generation, "stage_one_lora_strength")),
        stage_two_lora_strength=float(_required(generation, "stage_two_lora_strength")),
        stage_two_sigmas=tuple(float(item) for item in _required(generation, "stage_two_sigmas")),
        scheduler=SchedulerProfile(
            max_shift=float(_required(scheduler, "max_shift")),
            base_shift=float(_required(scheduler, "base_shift")),
            stretch=bool(_required(scheduler, "stretch")),
            terminal=float(_required(scheduler, "terminal")),
        ),
        sampler=SamplerProfile(
            eta=float(_required(sampler, "eta")),
            bongmath=bool(_required(sampler, "bongmath")),
            bongmath_max_iterations=int(_required(sampler, "bongmath_max_iterations")),
            substep_seed_offset=int(_required(sampler, "substep_seed_offset")),
        ),
        video_guidance=_guidance(_section(generation, "video_guidance")),
        audio_guidance=_guidance(_section(generation, "audio_guidance")),
    )
    generation_profile.validate()
    return PipelineProfile(
        name=str(_required(profile, "name")),
        model=ModelProfile(
            repository_id=str(_required(model, "repository_id")),
            revision=str(_required(model, "revision")),
            relative_files={key: str(_required(model, key)) for key in model_keys},
        ),
        generation=generation_profile,
        decode=DecodeProfile(
            video_noise_seed_offset=int(_required(decode, "video_noise_seed_offset")),
            video_timesteps=tuple(float(item) for item in _required(decode, "video_timesteps")),
            video_activation_budget_bytes=int(_required(decode, "video_activation_budget_bytes")),
        ),
        media=MediaEncodingConfig(
            frame_rate=generation_profile.frame_rate,
            audio_sample_rate=int(_required(media, "audio_sample_rate")),
            video_codec=str(_required(media, "video_codec")),
            audio_codec=str(_required(media, "audio_codec")),
            pixel_format=str(_required(media, "pixel_format")),
            audio_bitrate=str(_required(media, "audio_bitrate")),
            crf=int(_required(media, "crf")),
            timeout_seconds=int(_required(media, "timeout_seconds")),
            ffmpeg_binary=os.getenv("LARA_FFMPEG_BINARY", str(_required(media, "ffmpeg_binary"))),
        ),
        download=DownloadProfile(
            cache_environment_variable=str(_required(download, "cache_environment_variable")),
            token_environment_variable=str(_required(download, "token_environment_variable")),
        ),
    )
