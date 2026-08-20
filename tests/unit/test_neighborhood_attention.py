from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.neighborhood_attention import _window_bounds, neighborhood_attention_3d


def _reference_na3d(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    kernel_size: tuple[int, int, int],
    is_causal: tuple[bool, bool, bool],
    scale: float,
) -> np.ndarray:
    output = np.empty_like(value)
    _, time, height, width, heads, _ = query.shape
    dimensions = (time, height, width)
    kernels = tuple(
        kernel if causal else min(kernel, dimension)
        for kernel, causal, dimension in zip(kernel_size, is_causal, dimensions, strict=True)
    )
    bounds = tuple(
        _window_bounds(dimension, kernel, causal)
        for dimension, kernel, causal in zip(dimensions, kernels, is_causal, strict=True)
    )

    for time_index in range(time):
        time_slice = slice(bounds[0][0][time_index], bounds[0][1][time_index])
        for height_index in range(height):
            height_slice = slice(bounds[1][0][height_index], bounds[1][1][height_index])
            for width_index in range(width):
                width_slice = slice(bounds[2][0][width_index], bounds[2][1][width_index])
                local_key = key[:, time_slice, height_slice, width_slice].reshape(1, -1, heads, key.shape[-1])
                local_value = value[:, time_slice, height_slice, width_slice].reshape(1, -1, heads, value.shape[-1])
                local_query = query[:, time_index, height_index, width_index]
                logits = np.einsum("bhd,bkhd->bhk", local_query, local_key) * scale
                logits -= logits.max(axis=-1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                output[:, time_index, height_index, width_index] = np.einsum(
                    "bhk,bkhd->bhd", probabilities, local_value
                )
    return output


@pytest.mark.parametrize(
    ("shape", "kernel_size", "is_causal", "mask_budget"),
    [
        ((1, 3, 4, 5, 2, 4), (3, 3, 3), (False, False, False), 180),
        ((1, 4, 2, 3, 1, 3), (3, 2, 3), (True, False, False), 48),
    ],
)
def test_neighborhood_attention_matches_numpy(
    shape: tuple[int, ...],
    kernel_size: tuple[int, int, int],
    is_causal: tuple[bool, bool, bool],
    mask_budget: int,
) -> None:
    generator = np.random.default_rng(20260819)
    query = generator.normal(size=shape).astype(np.float32)
    key = generator.normal(size=shape).astype(np.float32)
    value = generator.normal(size=shape).astype(np.float32)
    scale = 1.0 / math.sqrt(shape[-1])

    expected = _reference_na3d(query, key, value, kernel_size, is_causal, scale)
    actual = neighborhood_attention_3d(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        kernel_size=kernel_size,
        is_causal=is_causal,
        mask_element_budget=mask_budget,
        scale=scale,
    )

    # MLX fused SDPA changes the reduction order versus the scalar NumPy
    # reference. CUDA golden parity retains its independent release tolerance.
    np.testing.assert_allclose(np.asarray(actual), expected, atol=3e-3, rtol=3e-3)


def test_neighborhood_attention_is_stable_across_tile_budgets() -> None:
    generator = np.random.default_rng(20260819)
    shape = (1, 3, 4, 5, 2, 4)
    query = mx.array(generator.normal(size=shape).astype(np.float32))
    key = mx.array(generator.normal(size=shape).astype(np.float32))
    value = mx.array(generator.normal(size=shape).astype(np.float32))

    smaller_tiles = neighborhood_attention_3d(
        query,
        key,
        value,
        kernel_size=(3, 3, 3),
        mask_element_budget=180,
    )
    larger_tiles = neighborhood_attention_3d(
        query,
        key,
        value,
        kernel_size=(3, 3, 3),
        mask_element_budget=2_000,
    )

    np.testing.assert_allclose(np.asarray(smaller_tiles), np.asarray(larger_tiles), atol=3e-3, rtol=3e-3)


def test_neighborhood_attention_rejects_invalid_shape() -> None:
    tensor = mx.zeros((1, 2, 3, 4, 5))

    with pytest.raises(LaraError, match="LARA-TENSOR-007"):
        neighborhood_attention_3d(
            tensor,
            tensor,
            tensor,
            kernel_size=(3, 3, 3),
            mask_element_budget=128,
        )


def test_neighborhood_attention_rejects_impossible_budget() -> None:
    tensor = mx.zeros((1, 2, 3, 4, 1, 2))

    with pytest.raises(LaraError, match="LARA-TENSOR-007"):
        neighborhood_attention_3d(
            tensor,
            tensor,
            tensor,
            kernel_size=(2, 3, 3),
            mask_element_budget=1,
        )
