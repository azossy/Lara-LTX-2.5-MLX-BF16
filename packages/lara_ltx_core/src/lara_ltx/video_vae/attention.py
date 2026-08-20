"""Checkpoint-compatible per-frame spatial attention for the convolutional VAE."""

from __future__ import annotations

import math
from typing import Final

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.transformer.attention import scaled_dot_product_attention

CHANNEL_AXIS: Final = 1
NORMALIZATION_EPSILON: Final = 1e-12


class RMSNorm2D(nn.Module):
    """Upstream channel-first L2 normalization with a learnable channel gain."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = math.sqrt(channels)
        self.gamma = mx.ones((channels, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        denominator = mx.sqrt(mx.sum(mx.square(x), axis=CHANNEL_AXIS, keepdims=True))
        gain = self.scale * self.gamma[None, :, None, :, :]
        return x * gain / mx.maximum(denominator, NORMALIZATION_EPSILON)


class PointwiseConv2d(nn.Module):
    """A 1x1 2D convolution applied independently to every video frame."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = mx.zeros((out_channels, in_channels, 1, 1))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        batch, channels, frames, height, width = x.shape
        tokens = mx.transpose(x, (0, 2, 3, 4, 1)).reshape(batch * frames, height * width, channels)
        matrix = self.weight[:, :, 0, 0]
        output = tokens @ mx.transpose(matrix) + self.bias
        return mx.transpose(output.reshape(batch, frames, height, width, -1), (0, 4, 1, 2, 3))


class AttnBlock3D(nn.Module):
    """Single-head self-attention over HxW positions independently per frame."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm2D(in_channels)
        self.to_qkv = PointwiseConv2d(in_channels, in_channels * 3)
        self.proj = PointwiseConv2d(in_channels, in_channels)

    def __call__(self, hidden_states: mx.array, *, causal: bool = True) -> mx.array:
        del causal
        batch, channels, frames, height, width = hidden_states.shape
        normalized = self.norm(hidden_states)
        qkv = self.to_qkv(normalized)
        qkv = mx.transpose(qkv, (0, 2, 3, 4, 1)).reshape(batch * frames, height * width, channels * 3)
        query, key, value = mx.split(qkv, 3, axis=-1)
        attended = scaled_dot_product_attention(query, key, value, heads=1)
        attended = mx.transpose(attended.reshape(batch, frames, height, width, channels), (0, 4, 1, 2, 3))
        return hidden_states + self.proj(attended)
