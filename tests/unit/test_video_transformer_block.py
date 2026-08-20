from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.transformer.blocks import VideoTransformerBlock, VideoTransformerConfig

VIDEO_DIMENSION = 8
CONTEXT_DIMENSION = 6
ATTENTION_HEADS = 2
ATTENTION_HEAD_DIMENSION = 4
VIDEO_TOKEN_COUNT = 3
CONTEXT_TOKEN_COUNT = 4
ADALN_PARAMETER_COUNT = 6


def test_video_transformer_block_runs_with_checkpoint_compatible_layout() -> None:
    block = VideoTransformerBlock(
        VideoTransformerConfig(
            dim=VIDEO_DIMENSION,
            heads=ATTENTION_HEADS,
            head_dim=ATTENTION_HEAD_DIMENSION,
            context_dim=CONTEXT_DIMENSION,
        )
    )
    x = mx.arange(VIDEO_TOKEN_COUNT * VIDEO_DIMENSION, dtype=mx.float32).reshape(
        1,
        VIDEO_TOKEN_COUNT,
        VIDEO_DIMENSION,
    )
    context = mx.arange(CONTEXT_TOKEN_COUNT * CONTEXT_DIMENSION, dtype=mx.float32).reshape(
        1,
        CONTEXT_TOKEN_COUNT,
        CONTEXT_DIMENSION,
    )
    timesteps = mx.zeros((1, VIDEO_TOKEN_COUNT, ADALN_PARAMETER_COUNT * VIDEO_DIMENSION))

    output = block(x, context=context, timesteps=timesteps)
    mx.eval(output)

    assert output.shape == x.shape
    assert np.isfinite(np.asarray(output)).all()
    parameters = block.parameters()
    assert parameters["attn1"]["to_q"]["weight"].shape == (VIDEO_DIMENSION, VIDEO_DIMENSION)
    assert parameters["attn2"]["to_k"]["weight"].shape == (VIDEO_DIMENSION, CONTEXT_DIMENSION)
    assert parameters["ff"]["net"][0]["proj"]["weight"].shape == (VIDEO_DIMENSION * 4, VIDEO_DIMENSION)
    assert parameters["scale_shift_table"].shape == (ADALN_PARAMETER_COUNT, VIDEO_DIMENSION)
