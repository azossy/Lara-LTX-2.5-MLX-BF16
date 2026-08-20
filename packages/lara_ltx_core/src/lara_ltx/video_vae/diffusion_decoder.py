"""Checkpoint-compatible untiled production Diffusion VAE decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from lara_ltx.errors import LaraError
from lara_ltx.models.checkpoint import BF16_DTYPE
from lara_ltx.models.loading import iter_component_weight_batches
from lara_ltx.models.mapping import validate_mapping
from lara_ltx.transformer.attention import RMSNorm
from lara_ltx.transformer.timestep import PixArtAlphaCombinedTimestepSizeEmbeddings

from .diffusion_blocks import (
    DEFAULT_MASK_ELEMENT_BUDGET,
    CombinedDiffusionNABlock,
    CombinedDiffusionNABlockConfig,
    NABlock,
    NABlockConfig,
)
from .diffusion_layers import AdaLNZero, LinearPixelShuffleUpsample, LinearPixelShuffleUpsampleConfig
from .ops import PerChannelStatistics, patchify, unpatchify

MODEL_OUTPUT_V: Final = "v"
MODEL_OUTPUT_X0: Final = "x0"
DETERMINISTIC_STAGE_COUNT: Final = 4
STAGE_COUNT: Final = DETERMINISTIC_STAGE_COUNT + 1
DEFAULT_STAGE_CHANNELS: Final = (2048, 1024, 512, 512, 256)
DEFAULT_STAGE_DEPTHS: Final = (4, 6, 4, 2, 8)
DEFAULT_STAGE_KERNELS: Final = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11))
DEFAULT_UPSAMPLES: Final = (((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2))
DEFAULT_STAGE5_KERNEL: Final = (11, 11, 11)
DEFAULT_TIMESTEP_EMBEDDING_DIM: Final = 384
DEFAULT_INFERENCE_STEPS: Final = 1
DEFAULT_TIMESTEP_SCALE_MULTIPLIER: Final = 1000.0
BF16_BYTES: Final = 2
DETERMINISTIC_WORKING_SET_FACTOR: Final = 16
DIFFUSION_WORKING_SET_FACTOR: Final = 24
TILED_STAGE_COUNT: Final = 2
DIFFUSION_DECODER_COMPONENT: Final = "diffusion_video_decoder"
DIFFUSION_DECODER_IGNORED_SOURCES: Final = ({"source_key": "decoder.type_emb", "reason": "upstream_load_artifact"},)


def _bounded_halo_interval(
    start: int,
    end: int,
    *,
    halo: int,
    dimension: int,
    minimum_size: int,
) -> tuple[int, int]:
    outer_start = max(0, start - halo)
    outer_end = min(dimension, end + halo)
    deficit = minimum_size - (outer_end - outer_start)
    if deficit > 0:
        shift_left = min(outer_start, deficit)
        outer_start -= shift_left
        outer_end = min(dimension, outer_end + deficit - shift_left)
    return outer_start, outer_end


def _fit_tile_shape(
    dimensions: tuple[int, int, int],
    halo: tuple[int, int, int],
    *,
    channels: int,
    budget_bytes: int,
    working_set_factor: int,
) -> tuple[int, int, int]:
    tile = list(dimensions)

    def estimated_bytes() -> int:
        outer = [
            min(dimension, core + 2 * axis_halo)
            for dimension, core, axis_halo in zip(dimensions, tile, halo, strict=True)
        ]
        return math.prod(outer) * channels * BF16_BYTES * working_set_factor

    while estimated_bytes() > budget_bytes and max(tile) > 1:
        axis = max(range(3), key=lambda index: tile[index] / max(1, halo[index]))
        tile[axis] = max(1, (tile[axis] + 1) // 2)
    if estimated_bytes() > budget_bytes:
        raise LaraError("LARA-RUNTIME-001", details={"reason": "diffvae_tile_budget"})
    return tile[0], tile[1], tile[2]


@dataclass(frozen=True)
class DiffusionVideoDecoderConfig:
    """Production decoder layout with configurable dimensions for parity tests."""

    in_channels: int = 128
    out_channels: int = 3
    patch_size: int = 4
    head_dim: int = 64
    stage_channels: tuple[int, ...] = DEFAULT_STAGE_CHANNELS
    stage_depths: tuple[int, ...] = DEFAULT_STAGE_DEPTHS
    stage_kernels: tuple[tuple[int, int, int], ...] = DEFAULT_STAGE_KERNELS
    upsamples: tuple[tuple[tuple[int, int, int], int], ...] = DEFAULT_UPSAMPLES
    stage5_kernel: tuple[int, int, int] = DEFAULT_STAGE5_KERNEL
    stage5_channels: int | None = None
    timestep_embedding_dim: int = DEFAULT_TIMESTEP_EMBEDDING_DIM
    default_num_inference_steps: int = DEFAULT_INFERENCE_STEPS
    timestep_scale_multiplier: float = DEFAULT_TIMESTEP_SCALE_MULTIPLIER
    model_output_type: Literal["v", "x0"] = MODEL_OUTPUT_X0
    mask_element_budget: int = DEFAULT_MASK_ELEMENT_BUDGET


@dataclass(frozen=True)
class Stage5TilingConfig:
    """Core tile size on the patchified stage-5 time/height/width grid."""

    tile_shape: tuple[int, int, int]


@dataclass(frozen=True)
class Stage4TilingConfig:
    """Core tile size on the stage-4 input time/height/width grid."""

    tile_shape: tuple[int, int, int]


@dataclass(frozen=True)
class AxisPad:
    """Padding recorded on one latent axis."""

    before: int = 0
    after: int = 0


@dataclass(frozen=True)
class RecommendedDecoderTiling:
    """Stage-4 and stage-5 tile sizes selected from an activation budget."""

    stage4: Stage4TilingConfig
    stage5: Stage5TilingConfig
    resident_stage3_bytes: int


class DiffusionVideoDecoder(nn.Module):
    """Stages 1-4 context decoding followed by stage-5 diffusion denoising."""

    def __init__(self, config: DiffusionVideoDecoderConfig | None = None) -> None:
        super().__init__()
        config = config or DiffusionVideoDecoderConfig()
        self._validate_config(config)
        self.config = config
        self.patch_size = config.patch_size
        self.out_channels = config.out_channels
        self.stage_channels = config.stage_channels
        self.stage_depths = config.stage_depths
        self.base_channels = config.stage_channels[-1]
        self.context_channels = config.stage_channels[-1]
        self.stage5_kernel = config.stage5_kernel
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.model_output_type = config.model_output_type
        self.trailing_ghost_latent_frames = (config.stage_kernels[0][0] // 2) * 2

        self.per_channel_statistics = PerChannelStatistics(config.in_channels)
        self.conv_in = nn.Linear(config.in_channels, config.stage_channels[0], bias=True)
        self.det_stages: list[list[NABlock]] = []
        self.upsamples: list[LinearPixelShuffleUpsample] = []
        for stage_index in range(DETERMINISTIC_STAGE_COUNT):
            channels = config.stage_channels[stage_index]
            self.det_stages.append(
                [
                    NABlock(
                        NABlockConfig(
                            dim=channels,
                            kernel_size=config.stage_kernels[stage_index],
                            head_dim=config.head_dim,
                            mask_element_budget=config.mask_element_budget,
                        )
                    )
                    for _ in range(config.stage_depths[stage_index])
                ]
            )
            stride, reduction = config.upsamples[stage_index]
            self.upsamples.append(
                LinearPixelShuffleUpsample(
                    LinearPixelShuffleUpsampleConfig(
                        in_channels=channels,
                        stride=stride,
                        out_channels_reduction_factor=reduction,
                    )
                )
            )

        stage5_channels = config.stage5_channels or self.context_channels
        patched_pixel_channels = config.out_channels * config.patch_size**2
        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(config.timestep_embedding_dim)
        self.conv_in_x_t = nn.Linear(patched_pixel_channels, stage5_channels, bias=True)
        self.shared_adaln = AdaLNZero(stage5_channels, config.timestep_embedding_dim)
        self.diff_blocks = [
            CombinedDiffusionNABlock(
                CombinedDiffusionNABlockConfig(
                    dim=stage5_channels,
                    context_channels=self.context_channels,
                    kernel_size=config.stage5_kernel,
                    head_dim=config.head_dim,
                    mask_element_budget=config.mask_element_budget,
                )
            )
            for _ in range(config.stage_depths[-1])
        ]
        self.norm_out = RMSNorm(stage5_channels)
        self.conv_out = nn.Linear(stage5_channels, patched_pixel_channels, bias=True)

    @property
    def minimum_latent_shape(self) -> tuple[int, int, int]:
        cumulative = [1, 1, 1]
        minimum = [1, 1, 1]
        for stage_index in range(DETERMINISTIC_STAGE_COUNT):
            for axis in range(3):
                kernel = self.config.stage_kernels[stage_index][axis]
                minimum[axis] = max(minimum[axis], -(-kernel // cumulative[axis]))
            stride, _ = self.config.upsamples[stage_index]
            cumulative = [value * factor for value, factor in zip(cumulative, stride, strict=True)]
        for axis in range(3):
            minimum[axis] = max(minimum[axis], -(-self.stage5_kernel[axis] // cumulative[axis]))
        return minimum[0], minimum[1], minimum[2]

    def output_shape_for_latent(self, latent_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        time, height, width = latent_shape
        for stride, _ in self.config.upsamples:
            time *= stride[0]
            height *= stride[1]
            width *= stride[2]
            if stride[0] == 2:
                time -= 1
        return time, height * self.patch_size, width * self.patch_size

    def recommend_tiling(
        self,
        latent_shape: tuple[int, int, int],
        *,
        activation_budget_bytes: int,
        batch_size: int = 1,
    ) -> RecommendedDecoderTiling:
        """Select conservative tile cores without quantizing or swapping weights."""

        if activation_budget_bytes <= 0 or batch_size <= 0 or any(value <= 0 for value in latent_shape):
            raise LaraError("LARA-RUNTIME-001", details={"reason": "invalid_diffvae_budget"})
        minimum = self.minimum_latent_shape
        work = tuple(max(value, floor) for value, floor in zip(latent_shape, minimum, strict=True))
        time = work[0] + self.trailing_ghost_latent_frames
        height, width = work[1], work[2]
        for stride, _ in self.config.upsamples[:3]:
            time *= stride[0]
            height *= stride[1]
            width *= stride[2]
            if stride[0] == 2:
                time -= 1
        stage4_dimensions = (time, height, width)
        resident_bytes = batch_size * math.prod(stage4_dimensions) * self.config.stage_channels[3] * BF16_BYTES
        remaining = activation_budget_bytes - resident_bytes
        if remaining <= 0:
            raise LaraError("LARA-RUNTIME-001", details={"reason": "stage3_resident_budget"})
        per_stage_budget = remaining // TILED_STAGE_COUNT
        stage4_depth = self.config.stage_depths[3]
        stage4_halo = tuple((kernel // 2) * stage4_depth for kernel in self.config.stage_kernels[3])
        stage4_shape = _fit_tile_shape(
            stage4_dimensions,
            stage4_halo,
            channels=self.config.stage_channels[3],
            budget_bytes=per_stage_budget,
            working_set_factor=DETERMINISTIC_WORKING_SET_FACTOR,
        )
        last_stride = self.config.upsamples[3][0]
        stage5_dimensions = tuple(
            dimension * stride - (1 if axis == 0 and stride == 2 else 0)
            for axis, (dimension, stride) in enumerate(zip(stage4_dimensions, last_stride, strict=True))
        )
        stage5_halo = tuple((kernel // 2) * self.config.stage_depths[-1] for kernel in self.stage5_kernel)
        stage5_channels = self.config.stage5_channels or self.context_channels
        stage5_shape = _fit_tile_shape(
            stage5_dimensions,
            stage5_halo,
            channels=stage5_channels,
            budget_bytes=per_stage_budget,
            working_set_factor=DIFFUSION_WORKING_SET_FACTOR,
        )
        return RecommendedDecoderTiling(
            stage4=Stage4TilingConfig(stage4_shape),
            stage5=Stage5TilingConfig(stage5_shape),
            resident_stage3_bytes=resident_bytes,
        )

    @staticmethod
    def _resize_axis(x: mx.array, axis: int, size: int, *, symmetric: bool) -> tuple[mx.array, AxisPad]:
        length = x.shape[axis]
        if length == size:
            return x, AxisPad()
        if length > size:
            before = (length - size) // 2 if symmetric else 0
            slices = [slice(None)] * x.ndim
            slices[axis] = slice(before, before + size)
            return x[tuple(slices)], AxisPad(before, length - size - before)
        needed = size - length
        before = needed // 2 if symmetric else 0
        after = needed - before
        pieces: list[mx.array] = []
        if before:
            first_shape = list(x.shape)
            first_shape[axis] = before
            pieces.append(mx.broadcast_to(mx.take(x, mx.array([0]), axis=axis), first_shape))
        pieces.append(x)
        if after:
            last_shape = list(x.shape)
            last_shape[axis] = after
            pieces.append(mx.broadcast_to(mx.take(x, mx.array([length - 1]), axis=axis), last_shape))
        return mx.concatenate(pieces, axis=axis), AxisPad(before, after)

    def prepare_latent(self, latent: mx.array) -> tuple[mx.array, tuple[AxisPad, AxisPad, AxisPad]]:
        minimum = self.minimum_latent_shape
        x, time_pad = self._resize_axis(latent, 2, max(latent.shape[2], minimum[0]), symmetric=False)
        x, height_pad = self._resize_axis(x, 3, max(x.shape[3], minimum[1]), symmetric=True)
        x, width_pad = self._resize_axis(x, 4, max(x.shape[4], minimum[2]), symmetric=True)
        return x, (time_pad, height_pad, width_pad)

    def _production_context(
        self,
        latent: mx.array,
        stage4_tiling: Stage4TilingConfig | None,
    ) -> mx.array:
        ghosted = latent
        if self.trailing_ghost_latent_frames:
            ghosted, _ = self._resize_axis(
                latent,
                2,
                latent.shape[2] + self.trailing_ghost_latent_frames,
                symmetric=False,
            )
        stage3 = self.forward_stages_1_to_3(ghosted)
        context = (
            self.forward_stage_4(stage3) if stage4_tiling is None else self.forward_stage_4_tiled(stage3, stage4_tiling)
        )
        time_scale = 1
        for stride, _ in self.config.upsamples:
            time_scale *= stride[0]
        ghost_frames = self.trailing_ghost_latent_frames * time_scale
        content_frames = max(context.shape[1] - ghost_frames, 1)
        keep_frames = min(context.shape[1], max(content_frames, self.stage5_kernel[0]))
        return context[:, :keep_frames]

    def _diffusion_loop(
        self,
        context: mx.array,
        initial_noise: mx.array,
        timestep_schedule: mx.array,
        stage5_tiling: Stage5TilingConfig | None,
    ) -> mx.array:
        x_t = initial_noise
        for step_index in range(timestep_schedule.shape[1]):
            timestep_now = timestep_schedule[:, step_index]
            context_value, x = self.context_and_x_for_diff_step(context, x_t)
            model_output = (
                self.forward_diff_step(context_value, x, timestep_now)
                if stage5_tiling is None
                else self.forward_diff_step_tiled(context_value, x, timestep_now, stage5_tiling)
            )
            if self.model_output_type == MODEL_OUTPUT_X0 and step_index == timestep_schedule.shape[1] - 1:
                return model_output
            timestep_next = (
                timestep_schedule[:, step_index + 1]
                if step_index + 1 < timestep_schedule.shape[1]
                else mx.zeros_like(timestep_now)
            )
            x_t = self.euler_step(x_t, model_output, timestep_now, timestep_next)
        return x_t

    @staticmethod
    def _validate_config(config: DiffusionVideoDecoderConfig) -> None:
        stage_lengths_match = (
            len(config.stage_channels) == len(config.stage_depths) == len(config.stage_kernels) == STAGE_COUNT
        )
        dimensions = (*config.stage_channels, config.stage5_channels or config.stage_channels[-1])
        valid_reductions = len(config.upsamples) == DETERMINISTIC_STAGE_COUNT and all(
            reduction > 0 for _, reduction in config.upsamples
        )
        reductions_match = (
            all(
                config.stage_channels[index] // config.upsamples[index][1] == config.stage_channels[index + 1]
                for index in range(len(config.upsamples))
            )
            if valid_reductions
            else False
        )
        if (
            not stage_lengths_match
            or not reductions_match
            or config.head_dim <= 0
            or any(value <= 0 or value % config.head_dim for value in dimensions)
            or any(depth <= 0 for depth in config.stage_depths)
            or config.patch_size <= 0
            or config.default_num_inference_steps <= 0
            or config.model_output_type not in (MODEL_OUTPUT_V, MODEL_OUTPUT_X0)
            or any(
                len(kernel) != 3 or any(value <= 0 or value % 2 == 0 for value in kernel)
                for kernel in config.stage_kernels
            )
            or len(config.stage5_kernel) != 3
            or any(value <= 0 or value % 2 == 0 for value in config.stage5_kernel)
        ):
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_diffusion_decoder_config"})

    def _run_det_stage(self, x: mx.array, stage_index: int, *, drop_leading_frame: bool) -> mx.array:
        for block in self.det_stages[stage_index]:
            x = block(x)
        return self.upsamples[stage_index](x, drop_leading_frame=drop_leading_frame)

    def forward_stages_1_to_3(self, latent: mx.array, *, drop_leading_frame: bool = True) -> mx.array:
        x = self.per_channel_statistics.un_normalize(latent)
        x = self.conv_in(mx.transpose(x, (0, 2, 3, 4, 1)))
        for stage_index in range(DETERMINISTIC_STAGE_COUNT - 1):
            x = self._run_det_stage(x, stage_index, drop_leading_frame=drop_leading_frame)
        return x

    def forward_stage_4(self, x: mx.array, *, drop_leading_frame: bool = True) -> mx.array:
        return self._run_det_stage(x, DETERMINISTIC_STAGE_COUNT - 1, drop_leading_frame=drop_leading_frame)

    def _stage4_features(self, x: mx.array) -> mx.array:
        for block in self.det_stages[DETERMINISTIC_STAGE_COUNT - 1]:
            x = block(x)
        return x

    def forward_stage_4_tiled(
        self,
        x: mx.array,
        tiling: Stage4TilingConfig,
        *,
        drop_leading_frame: bool = True,
    ) -> mx.array:
        """Run stage 4 in halo-expanded tiles before per-core upsampling."""

        if len(tiling.tile_shape) != 3 or any(value <= 0 for value in tiling.tile_shape):
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_stage4_tile_shape"})
        stage_index = DETERMINISTIC_STAGE_COUNT - 1
        depth = len(self.det_stages[stage_index])
        halo = tuple((kernel // 2) * depth for kernel in self.config.stage_kernels[stage_index])
        dimensions = x.shape[1:4]
        time_parts: list[mx.array] = []
        for time_start in range(0, dimensions[0], tiling.tile_shape[0]):
            time_end = min(time_start + tiling.tile_shape[0], dimensions[0])
            height_parts: list[mx.array] = []
            for height_start in range(0, dimensions[1], tiling.tile_shape[1]):
                height_end = min(height_start + tiling.tile_shape[1], dimensions[1])
                width_parts: list[mx.array] = []
                for width_start in range(0, dimensions[2], tiling.tile_shape[2]):
                    width_end = min(width_start + tiling.tile_shape[2], dimensions[2])
                    intervals = tuple(
                        _bounded_halo_interval(
                            start,
                            end,
                            halo=axis_halo,
                            dimension=dimension,
                            minimum_size=kernel,
                        )
                        for start, end, axis_halo, dimension, kernel in zip(
                            (time_start, height_start, width_start),
                            (time_end, height_end, width_end),
                            halo,
                            dimensions,
                            self.config.stage_kernels[stage_index],
                            strict=True,
                        )
                    )
                    outer_starts = tuple(interval[0] for interval in intervals)
                    outer_ends = tuple(interval[1] for interval in intervals)
                    slices = (
                        slice(None),
                        slice(outer_starts[0], outer_ends[0]),
                        slice(outer_starts[1], outer_ends[1]),
                        slice(outer_starts[2], outer_ends[2]),
                        slice(None),
                    )
                    tile = self._stage4_features(x[slices])
                    core_slices = (
                        slice(None),
                        slice(time_start - outer_starts[0], time_end - outer_starts[0]),
                        slice(height_start - outer_starts[1], height_end - outer_starts[1]),
                        slice(width_start - outer_starts[2], width_end - outer_starts[2]),
                        slice(None),
                    )
                    core = tile[core_slices]
                    output = self.upsamples[stage_index](
                        core,
                        drop_leading_frame=drop_leading_frame and time_start == 0,
                    )
                    mx.eval(output)
                    width_parts.append(output)
                height_parts.append(mx.concatenate(width_parts, axis=3))
            time_parts.append(mx.concatenate(height_parts, axis=2))
        return mx.concatenate(time_parts, axis=1)

    def context_and_x_for_diff_step(self, context: mx.array, x_t: mx.array) -> tuple[mx.array, mx.array]:
        patched = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(mx.transpose(patched, (0, 2, 3, 4, 1)))
        if x.shape[:-1] != context.shape[:-1]:
            raise LaraError(
                "LARA-TENSOR-016",
                details={"reason": "context_noise_shape_mismatch", "context": context.shape, "noise": x.shape},
            )
        return context, x

    def _diffusion_features(self, context: mx.array, x: mx.array, timestep: mx.array) -> mx.array:
        embedding = self.t_embedder(self.timestep_scale_multiplier * timestep, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(embedding)
        for block in self.diff_blocks:
            x = block(mx.concatenate([context, x], axis=-1), modulation)
        return x

    def _project_diffusion_output(self, x: mx.array) -> mx.array:
        x = self.conv_out(self.norm_out(x))
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def forward_diff_step(self, context: mx.array, x: mx.array, timestep: mx.array) -> mx.array:
        return self._project_diffusion_output(self._diffusion_features(context, x, timestep))

    def forward_diff_step_tiled(
        self,
        context: mx.array,
        x: mx.array,
        timestep: mx.array,
        tiling: Stage5TilingConfig,
    ) -> mx.array:
        """Run stage 5 in exact halo-expanded tiles and stitch disjoint cores."""

        if len(tiling.tile_shape) != 3 or any(value <= 0 for value in tiling.tile_shape):
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_stage5_tile_shape"})
        halo = tuple((kernel // 2) * len(self.diff_blocks) for kernel in self.stage5_kernel)
        dimensions = context.shape[1:4]
        time_parts: list[mx.array] = []
        for time_start in range(0, dimensions[0], tiling.tile_shape[0]):
            time_end = min(time_start + tiling.tile_shape[0], dimensions[0])
            height_parts: list[mx.array] = []
            for height_start in range(0, dimensions[1], tiling.tile_shape[1]):
                height_end = min(height_start + tiling.tile_shape[1], dimensions[1])
                width_parts: list[mx.array] = []
                for width_start in range(0, dimensions[2], tiling.tile_shape[2]):
                    width_end = min(width_start + tiling.tile_shape[2], dimensions[2])
                    intervals = tuple(
                        _bounded_halo_interval(
                            start,
                            end,
                            halo=axis_halo,
                            dimension=dimension,
                            minimum_size=kernel,
                        )
                        for start, end, axis_halo, dimension, kernel in zip(
                            (time_start, height_start, width_start),
                            (time_end, height_end, width_end),
                            halo,
                            dimensions,
                            self.stage5_kernel,
                            strict=True,
                        )
                    )
                    outer_starts = tuple(interval[0] for interval in intervals)
                    outer_ends = tuple(interval[1] for interval in intervals)
                    slices = (
                        slice(None),
                        slice(outer_starts[0], outer_ends[0]),
                        slice(outer_starts[1], outer_ends[1]),
                        slice(outer_starts[2], outer_ends[2]),
                        slice(None),
                    )
                    tile = self._diffusion_features(context[slices], x[slices], timestep)
                    core_slices = (
                        slice(None),
                        slice(time_start - outer_starts[0], time_end - outer_starts[0]),
                        slice(height_start - outer_starts[1], height_end - outer_starts[1]),
                        slice(width_start - outer_starts[2], width_end - outer_starts[2]),
                        slice(None),
                    )
                    core = tile[core_slices]
                    mx.eval(core)
                    width_parts.append(core)
                height_parts.append(mx.concatenate(width_parts, axis=3))
            time_parts.append(mx.concatenate(height_parts, axis=2))
        features = mx.concatenate(time_parts, axis=1)
        return self._project_diffusion_output(features)

    def euler_step(
        self,
        x_t: mx.array,
        model_output: mx.array,
        timestep_now: mx.array,
        timestep_next: mx.array,
    ) -> mx.array:
        broadcast_shape = (timestep_now.shape[0],) + (1,) * (x_t.ndim - 1)
        delta = (timestep_now - timestep_next).reshape(broadcast_shape).astype(mx.float32)
        x_t_float = x_t.astype(mx.float32)
        if self.model_output_type == MODEL_OUTPUT_V:
            velocity = model_output.astype(mx.float32)
        else:
            sigma = timestep_now.reshape(broadcast_shape).astype(mx.float32)
            velocity = (x_t_float - model_output.astype(mx.float32)) / sigma
        result = x_t_float - delta * velocity
        return result.astype(x_t.dtype)

    def default_timesteps(self, batch_size: int) -> mx.array:
        steps = self.config.default_num_inference_steps
        sequence = mx.linspace(1.0, 1.0 / steps, steps, dtype=mx.float32)
        return mx.broadcast_to(sequence[None], (batch_size, steps))

    def decode_untiled(
        self,
        latent: mx.array,
        initial_noise: mx.array,
        *,
        timesteps: mx.array | None = None,
    ) -> mx.array:
        """Decode one full volume from explicit noise for reproducible serving."""

        timestep_schedule = self.default_timesteps(latent.shape[0]) if timesteps is None else timesteps
        if timestep_schedule.ndim != 2 or timestep_schedule.shape[0] != latent.shape[0]:
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_timestep_schedule"})
        context = self.forward_stage_4(self.forward_stages_1_to_3(latent))
        x_t = initial_noise
        for step_index in range(timestep_schedule.shape[1]):
            timestep_now = timestep_schedule[:, step_index]
            context_value, x = self.context_and_x_for_diff_step(context, x_t)
            model_output = self.forward_diff_step(context_value, x, timestep_now)
            if self.model_output_type == MODEL_OUTPUT_X0 and step_index == timestep_schedule.shape[1] - 1:
                return model_output
            timestep_next = (
                timestep_schedule[:, step_index + 1]
                if step_index + 1 < timestep_schedule.shape[1]
                else mx.zeros_like(timestep_now)
            )
            x_t = self.euler_step(x_t, model_output, timestep_now, timestep_next)
        return x_t

    def decode_stage5_tiled(
        self,
        latent: mx.array,
        initial_noise: mx.array,
        tiling: Stage5TilingConfig,
        *,
        timesteps: mx.array | None = None,
    ) -> mx.array:
        """Decode with full deterministic context and memory-bounded stage 5."""

        timestep_schedule = self.default_timesteps(latent.shape[0]) if timesteps is None else timesteps
        if timestep_schedule.ndim != 2 or timestep_schedule.shape[0] != latent.shape[0]:
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_timestep_schedule"})
        context = self.forward_stage_4(self.forward_stages_1_to_3(latent))
        x_t = initial_noise
        for step_index in range(timestep_schedule.shape[1]):
            timestep_now = timestep_schedule[:, step_index]
            context_value, x = self.context_and_x_for_diff_step(context, x_t)
            model_output = self.forward_diff_step_tiled(context_value, x, timestep_now, tiling)
            if self.model_output_type == MODEL_OUTPUT_X0 and step_index == timestep_schedule.shape[1] - 1:
                return model_output
            timestep_next = (
                timestep_schedule[:, step_index + 1]
                if step_index + 1 < timestep_schedule.shape[1]
                else mx.zeros_like(timestep_now)
            )
            x_t = self.euler_step(x_t, model_output, timestep_now, timestep_next)
        return x_t

    def decode_tiled(
        self,
        latent: mx.array,
        initial_noise: mx.array,
        stage4_tiling: Stage4TilingConfig,
        stage5_tiling: Stage5TilingConfig,
        *,
        timesteps: mx.array | None = None,
    ) -> mx.array:
        """Decode with halo tiling across both high-resolution decoder stages."""

        timestep_schedule = self.default_timesteps(latent.shape[0]) if timesteps is None else timesteps
        if timestep_schedule.ndim != 2 or timestep_schedule.shape[0] != latent.shape[0]:
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_timestep_schedule"})
        stage3 = self.forward_stages_1_to_3(latent)
        context = self.forward_stage_4_tiled(stage3, stage4_tiling)
        x_t = initial_noise
        for step_index in range(timestep_schedule.shape[1]):
            timestep_now = timestep_schedule[:, step_index]
            context_value, x = self.context_and_x_for_diff_step(context, x_t)
            model_output = self.forward_diff_step_tiled(context_value, x, timestep_now, stage5_tiling)
            if self.model_output_type == MODEL_OUTPUT_X0 and step_index == timestep_schedule.shape[1] - 1:
                return model_output
            timestep_next = (
                timestep_schedule[:, step_index + 1]
                if step_index + 1 < timestep_schedule.shape[1]
                else mx.zeros_like(timestep_now)
            )
            x_t = self.euler_step(x_t, model_output, timestep_now, timestep_next)
        return x_t

    def decode(
        self,
        latent: mx.array,
        initial_noise: mx.array,
        *,
        timesteps: mx.array | None = None,
        stage4_tiling: Stage4TilingConfig | None = None,
        stage5_tiling: Stage5TilingConfig | None = None,
        activation_budget_bytes: int | None = None,
    ) -> mx.array:
        """Production decode with size-floor padding, trailing ghosting, and crop."""

        if latent.ndim != 5 or initial_noise.ndim != 5 or latent.shape[0] != initial_noise.shape[0]:
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_decode_input"})
        content_shape = self.output_shape_for_latent(tuple(latent.shape[2:5]))
        expected_noise_shape = (latent.shape[0], self.out_channels, *content_shape)
        if initial_noise.shape != expected_noise_shape:
            raise LaraError(
                "LARA-TENSOR-016",
                details={"reason": "invalid_initial_noise_shape", "expected": expected_noise_shape},
            )
        if activation_budget_bytes is not None:
            if stage4_tiling is not None or stage5_tiling is not None:
                raise LaraError("LARA-TENSOR-016", details={"reason": "conflicting_tiling_configuration"})
            recommended = self.recommend_tiling(
                tuple(latent.shape[2:5]),
                activation_budget_bytes=activation_budget_bytes,
                batch_size=latent.shape[0],
            )
            stage4_tiling = recommended.stage4
            stage5_tiling = recommended.stage5
        padded_latent, (_, height_pad, width_pad) = self.prepare_latent(latent)
        work_shape = self.output_shape_for_latent(tuple(padded_latent.shape[2:5]))
        noise, _ = self._resize_axis(initial_noise, 2, work_shape[0], symmetric=False)
        noise, _ = self._resize_axis(noise, 3, work_shape[1], symmetric=True)
        noise, _ = self._resize_axis(noise, 4, work_shape[2], symmetric=True)
        timestep_schedule = self.default_timesteps(latent.shape[0]) if timesteps is None else timesteps
        if timestep_schedule.ndim != 2 or timestep_schedule.shape[0] != latent.shape[0]:
            raise LaraError("LARA-TENSOR-016", details={"reason": "invalid_timestep_schedule"})
        context = self._production_context(padded_latent, stage4_tiling)
        pixels = self._diffusion_loop(context, noise, timestep_schedule, stage5_tiling)
        spatial_scale_height = work_shape[1] // padded_latent.shape[3]
        spatial_scale_width = work_shape[2] // padded_latent.shape[4]
        height_start = height_pad.before * spatial_scale_height
        width_start = width_pad.before * spatial_scale_width
        return pixels[
            :,
            :,
            : content_shape[0],
            height_start : height_start + content_shape[1],
            width_start : width_start + content_shape[2],
        ]

    def __call__(self, latent: mx.array, initial_noise: mx.array) -> mx.array:
        return self.decode(latent, initial_noise)


def load_diffusion_video_decoder(
    *,
    checkpoint: Path,
    mapping: dict[str, object],
    config: DiffusionVideoDecoderConfig | None = None,
) -> DiffusionVideoDecoder:
    """Strict-load all reviewed decoder/statistic targets from one official shard."""

    decoder = DiffusionVideoDecoder(config)
    target_shapes = {name: tuple(value.shape) for name, value in tree_flatten(decoder.parameters())}
    ignored = mapping.get("ignored_sources")
    if mapping.get("component") != DIFFUSION_DECODER_COMPONENT or ignored != list(DIFFUSION_DECODER_IGNORED_SOURCES):
        raise LaraError("LARA-MODEL-037", details={"key": "decoder_manifest_metadata"})
    rules = validate_mapping(mapping, expected_target_keys=target_shapes)
    if any(rule.get("dtype") != BF16_DTYPE for rule in rules):
        raise LaraError("LARA-MODEL-037", details={"key": "decoder_source_dtype"})
    loaded_count = 0
    for batch in iter_component_weight_batches(
        (checkpoint,),
        rules,
        expected_target_shapes=target_shapes,
    ):
        decoder.load_weights(batch)
        mx.eval(*[value for _, value in batch])
        loaded_count += len(batch)
    if loaded_count != len(target_shapes):
        raise LaraError("LARA-MODEL-037", details={"key": "decoder_target_count"})
    return decoder
