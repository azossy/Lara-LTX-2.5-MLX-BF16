"""Patchified video/audio latent layouts, Gaussian noising, and HQ stage transition."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import mlx.core as mx

from lara_ltx.errors import LaraError

from .res2s import Res2sLatentState

DEFAULT_VIDEO_TEMPORAL_SCALE = 8
DEFAULT_VIDEO_SPATIAL_SCALE = 32
DEFAULT_AUDIO_SAMPLE_RATE = 16_000
DEFAULT_AUDIO_HOP_LENGTH = 160
DEFAULT_AUDIO_LATENT_DOWNSAMPLE = 4
CAUSAL_POSITION_OFFSET = 1


@dataclass(frozen=True)
class VideoLatentLayout:
    batch: int
    channels: int
    frames: int
    height: int
    width: int
    frame_rate: float
    patch_size_t: int = 1
    patch_size_hw: int = 1
    temporal_scale: int = DEFAULT_VIDEO_TEMPORAL_SCALE
    height_scale: int = DEFAULT_VIDEO_SPATIAL_SCALE
    width_scale: int = DEFAULT_VIDEO_SPATIAL_SCALE
    causal: bool = True

    def __post_init__(self) -> None:
        integers = (
            self.batch,
            self.channels,
            self.frames,
            self.height,
            self.width,
            self.patch_size_t,
            self.patch_size_hw,
            self.temporal_scale,
            self.height_scale,
            self.width_scale,
        )
        if (
            any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in integers)
            or not math.isfinite(self.frame_rate)
            or self.frame_rate <= 0
            or self.frames % self.patch_size_t
            or self.height % self.patch_size_hw
            or self.width % self.patch_size_hw
        ):
            raise LaraError("LARA-TENSOR-030", details={"reason": "invalid_video_latent_layout"})

    @property
    def latent_shape(self) -> tuple[int, int, int, int, int]:
        return self.batch, self.channels, self.frames, self.height, self.width

    @property
    def token_count(self) -> int:
        return (
            self.frames
            * self.height
            * self.width
            // (self.patch_size_t * self.patch_size_hw * self.patch_size_hw)
        )

    @property
    def token_channels(self) -> int:
        return self.channels * self.patch_size_t * self.patch_size_hw * self.patch_size_hw


@dataclass(frozen=True)
class AudioLatentLayout:
    batch: int
    channels: int
    frames: int
    mel_bins: int
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    hop_length: int = DEFAULT_AUDIO_HOP_LENGTH
    latent_downsample_factor: int = DEFAULT_AUDIO_LATENT_DOWNSAMPLE
    causal: bool = True
    shift: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.batch,
            self.channels,
            self.frames,
            self.mel_bins,
            self.sample_rate,
            self.hop_length,
            self.latent_downsample_factor,
        )
        if (
            any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in positive)
            or not isinstance(self.shift, int)
            or isinstance(self.shift, bool)
        ):
            raise LaraError("LARA-TENSOR-030", details={"reason": "invalid_audio_latent_layout"})

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.channels, self.frames, self.mel_bins

    @property
    def token_count(self) -> int:
        return self.frames

    @property
    def token_channels(self) -> int:
        return self.channels * self.mel_bins


@dataclass(frozen=True)
class PatchifiedLatentState:
    state: Res2sLatentState
    positions: mx.array
    attention_mask: mx.array | None = None
    keyframes_mask: mx.array | None = None


def patchify_video(value: mx.array, layout: VideoLatentLayout) -> mx.array:
    if tuple(value.shape) != layout.latent_shape:
        raise LaraError("LARA-TENSOR-030", details={"reason": "video_latent_shape_mismatch"})
    batch, channels, frames, height, width = value.shape
    pt = layout.patch_size_t
    ph = layout.patch_size_hw
    patched = value.reshape(
        batch,
        channels,
        frames // pt,
        pt,
        height // ph,
        ph,
        width // ph,
        ph,
    )
    return mx.transpose(patched, (0, 2, 4, 6, 1, 3, 5, 7)).reshape(
        batch,
        layout.token_count,
        layout.token_channels,
    )


def unpatchify_video(value: mx.array, layout: VideoLatentLayout) -> mx.array:
    expected = (layout.batch, layout.token_count, layout.token_channels)
    if tuple(value.shape) != expected:
        raise LaraError("LARA-TENSOR-030", details={"reason": "video_token_shape_mismatch"})
    pt = layout.patch_size_t
    ph = layout.patch_size_hw
    grid_f = layout.frames // pt
    grid_h = layout.height // ph
    grid_w = layout.width // ph
    patched = value.reshape(layout.batch, grid_f, grid_h, grid_w, layout.channels, pt, ph, ph)
    return mx.transpose(patched, (0, 4, 1, 5, 2, 6, 3, 7)).reshape(layout.latent_shape)


def _axis_bounds(size: int, patch_size: int, scale: int) -> mx.array:
    starts = mx.arange(0, size, patch_size, dtype=mx.float32)
    return mx.stack((starts, starts + patch_size), axis=-1) * scale


def video_positions(layout: VideoLatentLayout) -> mx.array:
    temporal = _axis_bounds(layout.frames, layout.patch_size_t, layout.temporal_scale)
    if layout.causal:
        temporal = mx.maximum(temporal + CAUSAL_POSITION_OFFSET - layout.temporal_scale, 0)
    temporal = temporal / layout.frame_rate
    height = _axis_bounds(layout.height, layout.patch_size_hw, layout.height_scale)
    width = _axis_bounds(layout.width, layout.patch_size_hw, layout.width_scale)
    tf, hf, wf = mx.meshgrid(
        mx.arange(temporal.shape[0]),
        mx.arange(height.shape[0]),
        mx.arange(width.shape[0]),
        indexing="ij",
    )
    bounds = mx.stack((temporal[tf], height[hf], width[wf]), axis=0).reshape(3, layout.token_count, 2)
    return mx.broadcast_to(bounds[None], (layout.batch, 3, layout.token_count, 2))


def create_video_state(
    layout: VideoLatentLayout,
    *,
    initial_latent: mx.array | None = None,
    dtype: mx.Dtype = mx.bfloat16,
) -> PatchifiedLatentState:
    latent = initial_latent if initial_latent is not None else mx.zeros(layout.latent_shape, dtype=dtype)
    if tuple(latent.shape) != layout.latent_shape:
        raise LaraError("LARA-TENSOR-030", details={"reason": "video_initial_latent_shape_mismatch"})
    patched = patchify_video(latent, layout)
    mask = mx.ones((layout.batch, layout.token_count, 1), dtype=mx.float32)
    keyframes = mx.zeros(mask.shape, dtype=mx.float32)
    tokens_per_frame = layout.token_count // (layout.frames // layout.patch_size_t)
    keyframes[:, :tokens_per_frame] = 1.0
    state = Res2sLatentState(latent=patched, denoise_mask=mask, clean_latent=patched)
    mx.eval(patched, mask, keyframes)
    return PatchifiedLatentState(
        state=state,
        positions=video_positions(layout),
        keyframes_mask=keyframes,
    )


def patchify_audio(value: mx.array, layout: AudioLatentLayout) -> mx.array:
    if tuple(value.shape) != layout.latent_shape:
        raise LaraError("LARA-TENSOR-030", details={"reason": "audio_latent_shape_mismatch"})
    return mx.transpose(value, (0, 2, 1, 3)).reshape(
        layout.batch,
        layout.token_count,
        layout.token_channels,
    )


def unpatchify_audio(value: mx.array, layout: AudioLatentLayout) -> mx.array:
    expected = (layout.batch, layout.token_count, layout.token_channels)
    if tuple(value.shape) != expected:
        raise LaraError("LARA-TENSOR-030", details={"reason": "audio_token_shape_mismatch"})
    return mx.transpose(
        value.reshape(layout.batch, layout.frames, layout.channels, layout.mel_bins),
        (0, 2, 1, 3),
    )


def _audio_times(layout: AudioLatentLayout, offset: int) -> mx.array:
    indices = mx.arange(layout.shift + offset, layout.shift + offset + layout.frames, dtype=mx.float32)
    mel_frames = indices * layout.latent_downsample_factor
    if layout.causal:
        mel_frames = mx.maximum(
            mel_frames + CAUSAL_POSITION_OFFSET - layout.latent_downsample_factor,
            0,
        )
    return mel_frames * layout.hop_length / layout.sample_rate


def audio_positions(layout: AudioLatentLayout) -> mx.array:
    bounds = mx.stack((_audio_times(layout, 0), _audio_times(layout, 1)), axis=-1)
    return mx.broadcast_to(bounds[None, None], (layout.batch, 1, layout.frames, 2))


def create_audio_state(
    layout: AudioLatentLayout,
    *,
    initial_latent: mx.array | None = None,
    dtype: mx.Dtype = mx.bfloat16,
) -> PatchifiedLatentState:
    latent = initial_latent if initial_latent is not None else mx.zeros(layout.latent_shape, dtype=dtype)
    if tuple(latent.shape) != layout.latent_shape:
        raise LaraError("LARA-TENSOR-030", details={"reason": "audio_initial_latent_shape_mismatch"})
    patched = patchify_audio(latent, layout)
    mask = mx.ones((layout.batch, layout.token_count, 1), dtype=mx.float32)
    state = Res2sLatentState(latent=patched, denoise_mask=mask, clean_latent=patched)
    mx.eval(patched, mask)
    return PatchifiedLatentState(state=state, positions=audio_positions(layout))


def apply_gaussian_noise(
    state: Res2sLatentState,
    noise: mx.array,
    noise_scale: float,
) -> Res2sLatentState:
    if (
        tuple(noise.shape) != tuple(state.latent.shape)
        or not math.isfinite(noise_scale)
        or not 0.0 <= noise_scale <= 1.0
    ):
        raise LaraError("LARA-TENSOR-030", details={"reason": "invalid_gaussian_noise_request"})
    dtype = state.latent.dtype
    mixed = state.latent.astype(mx.float32) * (1.0 - noise_scale)
    mixed = mixed + noise.astype(mx.float32) * noise_scale
    mask = state.denoise_mask.astype(mx.float32)
    latent = state.clean_latent.astype(mx.float32) * (1.0 - mask) + mixed * mask
    latent = latent.astype(dtype)
    mx.eval(latent)
    return replace(state, latent=latent)


class GaussianNoiser:
    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise LaraError("LARA-TENSOR-030", details={"reason": "invalid_gaussian_noise_seed"})
        self._key = mx.random.key(seed)

    def __call__(self, state: Res2sLatentState, noise_scale: float) -> Res2sLatentState:
        next_key, draw_key = mx.random.split(self._key, num=2)
        self._key = next_key
        noise = mx.random.normal(state.latent.shape, key=draw_key, dtype=state.latent.dtype)
        return apply_gaussian_noise(state, noise, noise_scale)


def prepare_stage_two_states(
    stage_one_video: Res2sLatentState,
    stage_one_audio: Res2sLatentState,
    *,
    stage_one_video_layout: VideoLatentLayout,
    stage_two_video_layout: VideoLatentLayout,
    audio_layout: AudioLatentLayout,
    upscaler: Callable[[mx.array], mx.array],
    noiser: Callable[[Res2sLatentState, float], Res2sLatentState],
    noise_scale: float,
) -> tuple[PatchifiedLatentState, PatchifiedLatentState]:
    stage_one_video_latent = unpatchify_video(stage_one_video.latent, stage_one_video_layout)
    upscaled_video = upscaler(stage_one_video_latent)
    mx.eval(upscaled_video)
    video = create_video_state(stage_two_video_layout, initial_latent=upscaled_video)
    stage_one_audio_latent = unpatchify_audio(stage_one_audio.latent, audio_layout)
    audio = create_audio_state(audio_layout, initial_latent=stage_one_audio_latent)
    return (
        replace(video, state=noiser(video.state, noise_scale)),
        replace(audio, state=noiser(audio.state, noise_scale)),
    )
