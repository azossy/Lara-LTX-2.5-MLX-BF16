"""Bounded-workspace 3D neighborhood attention for the DiffVAE decoder."""

from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx

from lara_ltx.errors import LaraError

SPATIAL_DIMENSION_COUNT = 3
EXPECTED_TENSOR_RANK = 6


def _invalid(reason: str, shape: object) -> LaraError:
    return LaraError("LARA-TENSOR-007", details={"reason": reason, "shape": shape})


def _window_bounds(length: int, kernel: int, causal: bool) -> tuple[list[int], list[int]]:
    """Return upstream-compatible inclusive-start/exclusive-end windows."""

    starts: list[int] = []
    ends: list[int] = []
    if causal:
        for index in range(length):
            starts.append(max(0, index - kernel + 1))
            ends.append(index + 1)
    else:
        kernel = min(kernel, length)
        last_start = length - kernel
        half = kernel // 2
        for index in range(length):
            start = min(max(index - half, 0), last_start)
            starts.append(start)
            ends.append(start + kernel)
    return starts, ends


def _pick_tiles(
    dimensions: tuple[int, int, int],
    kernels: tuple[int, int, int],
    mask_element_budget: int,
) -> tuple[int, int, int]:
    """Choose query tile lengths whose largest local mask fits the budget."""

    tiles = list(dimensions)

    def cost(candidate: list[int]) -> int:
        query_elements = math.prod(candidate)
        key_elements = math.prod(
            min(dimension, tile + kernel - 1)
            for tile, kernel, dimension in zip(candidate, kernels, dimensions, strict=True)
        )
        return query_elements * key_elements

    while cost(tiles) > mask_element_budget and max(tiles) > 1:
        axis = max(range(SPATIAL_DIMENSION_COUNT), key=lambda item: tiles[item] / kernels[item])
        if tiles[axis] <= 1:
            break
        tiles[axis] = max(1, (tiles[axis] + 1) // 2)

    if cost(tiles) > mask_element_budget:
        minimum_cost = math.prod(min(dimension, kernel) for dimension, kernel in zip(dimensions, kernels, strict=True))
        raise _invalid(
            f"mask_element_budget={mask_element_budget} cannot hold the smallest local window; minimum={minimum_cost}",
            dimensions,
        )
    return tiles[0], tiles[1], tiles[2]


def _attention_mask(
    relative_bounds: tuple[tuple[Sequence[int], Sequence[int]], ...],
    key_dimensions: tuple[int, int, int],
    dtype: mx.Dtype,
) -> mx.array:
    """Build one additive ``[1, 1, Nq, Nk]`` tile mask on the Metal device."""

    axes: list[mx.array] = []
    for (starts, ends), key_length in zip(relative_bounds, key_dimensions, strict=True):
        start_array = mx.array(starts)[:, None]
        end_array = mx.array(ends)[:, None]
        key_indices = mx.arange(key_length)[None, :]
        axes.append((key_indices >= start_array) & (key_indices < end_array))

    visible = (
        axes[0][:, None, None, :, None, None]
        & axes[1][None, :, None, None, :, None]
        & axes[2][None, None, :, None, None, :]
    )
    query_elements = math.prod(len(bounds[0]) for bounds in relative_bounds)
    key_elements = math.prod(key_dimensions)
    visible = visible.reshape(query_elements, key_elements)
    additive = mx.where(visible, mx.array(0.0, dtype=dtype), mx.array(-math.inf, dtype=dtype))
    return additive[None, None, :, :]


def _flatten_attention_input(tensor: mx.array) -> mx.array:
    batch, time, height, width, heads, head_dim = tensor.shape
    return mx.transpose(tensor, (0, 4, 1, 2, 3, 5)).reshape(
        batch,
        heads,
        time * height * width,
        head_dim,
    )


def neighborhood_attention_3d(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    kernel_size: Sequence[int],
    mask_element_budget: int,
    is_causal: Sequence[bool] | None = None,
    scale: float | None = None,
) -> mx.array:
    """Apply NATTEN-compatible attention to ``(B, T, H, W, NH, HD)`` tensors.

    Query tiles are evaluated eagerly at the SDPA boundary. This is intentional:
    MLX lazy evaluation must not retain every local mask and K/V slice until the
    complete DiffVAE output is requested.
    """

    if query.ndim != EXPECTED_TENSOR_RANK:
        raise _invalid(f"expected rank {EXPECTED_TENSOR_RANK}", query.shape)
    if query.shape != key.shape or query.shape != value.shape:
        raise _invalid("query, key, and value shapes must match", (query.shape, key.shape, value.shape))
    if len(kernel_size) != SPATIAL_DIMENSION_COUNT or any(int(item) <= 0 for item in kernel_size):
        raise _invalid("kernel_size must contain three positive integers", tuple(kernel_size))
    if mask_element_budget <= 0:
        raise _invalid("mask_element_budget must be positive", mask_element_budget)

    causal = (False, False, False) if is_causal is None else tuple(bool(item) for item in is_causal)
    if len(causal) != SPATIAL_DIMENSION_COUNT:
        raise _invalid("is_causal must contain three booleans", causal)

    batch, time, height, width, heads, head_dim = query.shape
    dimensions = (time, height, width)
    kernels = tuple(
        int(kernel) if causal_axis else min(int(kernel), dimension)
        for kernel, causal_axis, dimension in zip(kernel_size, causal, dimensions, strict=True)
    )
    bounded_kernels = tuple(min(kernel, dimension) for kernel, dimension in zip(kernels, dimensions, strict=True))
    tile_time, tile_height, tile_width = _pick_tiles(dimensions, bounded_kernels, mask_element_budget)
    bounds = tuple(
        _window_bounds(dimension, kernel, causal_axis)
        for dimension, kernel, causal_axis in zip(dimensions, kernels, causal, strict=True)
    )
    attention_scale = head_dim**-0.5 if scale is None else scale

    time_blocks: list[mx.array] = []
    for time_start in range(0, time, tile_time):
        time_end = min(time_start + tile_time, time)
        key_time_start = bounds[0][0][time_start]
        key_time_end = bounds[0][1][time_end - 1]
        relative_time = (
            tuple(item - key_time_start for item in bounds[0][0][time_start:time_end]),
            tuple(item - key_time_start for item in bounds[0][1][time_start:time_end]),
        )
        height_blocks: list[mx.array] = []
        for height_start in range(0, height, tile_height):
            height_end = min(height_start + tile_height, height)
            key_height_start = bounds[1][0][height_start]
            key_height_end = bounds[1][1][height_end - 1]
            relative_height = (
                tuple(item - key_height_start for item in bounds[1][0][height_start:height_end]),
                tuple(item - key_height_start for item in bounds[1][1][height_start:height_end]),
            )
            width_blocks: list[mx.array] = []
            for width_start in range(0, width, tile_width):
                width_end = min(width_start + tile_width, width)
                key_width_start = bounds[2][0][width_start]
                key_width_end = bounds[2][1][width_end - 1]
                relative_width = (
                    tuple(item - key_width_start for item in bounds[2][0][width_start:width_end]),
                    tuple(item - key_width_start for item in bounds[2][1][width_start:width_end]),
                )

                query_tile = query[:, time_start:time_end, height_start:height_end, width_start:width_end]
                key_tile = key[
                    :,
                    key_time_start:key_time_end,
                    key_height_start:key_height_end,
                    key_width_start:key_width_end,
                ]
                value_tile = value[
                    :,
                    key_time_start:key_time_end,
                    key_height_start:key_height_end,
                    key_width_start:key_width_end,
                ]
                mask = _attention_mask(
                    (relative_time, relative_height, relative_width),
                    (key_time_end - key_time_start, key_height_end - key_height_start, key_width_end - key_width_start),
                    query.dtype,
                )
                flat_query = _flatten_attention_input(query_tile)
                flat_key = _flatten_attention_input(key_tile)
                flat_value = _flatten_attention_input(value_tile)
                output = mx.fast.scaled_dot_product_attention(
                    flat_query,
                    flat_key,
                    flat_value,
                    scale=attention_scale,
                    mask=mask,
                )
                output = mx.transpose(
                    output.reshape(
                        batch,
                        heads,
                        time_end - time_start,
                        height_end - height_start,
                        width_end - width_start,
                        head_dim,
                    ),
                    (0, 2, 3, 4, 1, 5),
                )
                mx.eval(output)
                width_blocks.append(output)

            height_block = mx.concatenate(width_blocks, axis=3)
            mx.eval(height_block)
            height_blocks.append(height_block)

        time_block = mx.concatenate(height_blocks, axis=2)
        mx.eval(time_block)
        time_blocks.append(time_block)

    result = mx.concatenate(time_blocks, axis=1)
    mx.eval(result)
    return result
