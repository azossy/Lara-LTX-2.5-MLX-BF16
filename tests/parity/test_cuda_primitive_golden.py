from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.transformer.blocks import (
    AVTransformerBlock,
    TransformerStream,
    VideoTransformerBlock,
    VideoTransformerConfig,
)
from lara_ltx.transformer.layers import gelu_approx, rms_norm
from lara_ltx.transformer.rope import apply_split_rotary_emb
from lara_ltx.video_vae.neighborhood_attention import neighborhood_attention_3d

GOLDEN_PATH = Path("golden/cuda/primitive_bf16_cuda.npz")
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
NEIGHBORHOOD_MASK_ELEMENT_BUDGET = 180
BLOCK_WEIGHT_PREFIX = "block_weight__"
AV_BLOCK_WEIGHT_PREFIX = "av_block_weight__"
CROSS_ADALN_WEIGHT_PREFIX = "cross_adaln_weight__"
CROSS_ADALN_AV_BLOCK_WEIGHT_PREFIX = "cross_adaln_av_block_weight__"


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA primitive golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        return {key: archive[key] for key in archive.files}


def test_split_rope_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    result = apply_split_rotary_emb(
        mx.array(golden["rope_input"], dtype=mx.bfloat16),
        mx.array(golden["rope_cos"], dtype=mx.bfloat16),
        mx.array(golden["rope_sin"], dtype=mx.bfloat16),
    )
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["rope_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_rms_norm_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    result = rms_norm(
        mx.array(golden["norm_input"], dtype=mx.bfloat16),
        mx.array(golden["norm_weight"], dtype=mx.bfloat16),
    )
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["norm_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_gelu_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    result = gelu_approx(mx.array(golden["norm_input"], dtype=mx.bfloat16))
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["gelu_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_neighborhood_attention_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    result = neighborhood_attention_3d(
        mx.array(golden["na_query"], dtype=mx.bfloat16),
        mx.array(golden["na_key"], dtype=mx.bfloat16),
        mx.array(golden["na_value"], dtype=mx.bfloat16),
        kernel_size=tuple(int(item) for item in golden["na_kernel_size"]),
        mask_element_budget=NEIGHBORHOOD_MASK_ELEMENT_BUDGET,
    )
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["na_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_video_transformer_block_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    block = VideoTransformerBlock(
        VideoTransformerConfig(
            dim=golden["block_input"].shape[-1],
            heads=int(golden["block_heads"]),
            head_dim=int(golden["block_head_dim"]),
            context_dim=golden["block_context"].shape[-1],
        )
    )
    weights = [
        (name.removeprefix(BLOCK_WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
        for name, value in golden.items()
        if name.startswith(BLOCK_WEIGHT_PREFIX)
    ]
    block.load_weights(weights)
    result = block(
        mx.array(golden["block_input"], dtype=mx.bfloat16),
        context=mx.array(golden["block_context"], dtype=mx.bfloat16),
        timesteps=mx.array(golden["block_timesteps"], dtype=mx.bfloat16),
    )
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["block_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_av_transformer_block_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    config = VideoTransformerConfig(
        dim=golden["av_video_input"].shape[-1],
        heads=int(golden["av_heads"]),
        head_dim=int(golden["av_head_dim"]),
        context_dim=golden["av_context"].shape[-1],
    )
    block = AVTransformerBlock(video=config, audio=config)
    weights = [
        (name.removeprefix(AV_BLOCK_WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
        for name, value in golden.items()
        if name.startswith(AV_BLOCK_WEIGHT_PREFIX)
    ]
    block.load_weights(weights)
    video_output, audio_output = block(
        TransformerStream(
            x=mx.array(golden["av_video_input"], dtype=mx.bfloat16),
            context=mx.array(golden["av_context"], dtype=mx.bfloat16),
            timesteps=mx.array(golden["av_video_timesteps"], dtype=mx.bfloat16),
            cross_scale_shift_timestep=mx.array(golden["av_video_cross_timestep"], dtype=mx.bfloat16),
            cross_gate_timestep=mx.array(golden["av_video_cross_gate_timestep"], dtype=mx.bfloat16),
        ),
        TransformerStream(
            x=mx.array(golden["av_audio_input"], dtype=mx.bfloat16),
            context=mx.array(golden["av_context"], dtype=mx.bfloat16),
            timesteps=mx.array(golden["av_audio_timesteps"], dtype=mx.bfloat16),
            cross_scale_shift_timestep=mx.array(golden["av_audio_cross_timestep"], dtype=mx.bfloat16),
            cross_gate_timestep=mx.array(golden["av_audio_cross_gate_timestep"], dtype=mx.bfloat16),
        ),
    )
    assert video_output is not None and audio_output is not None
    np.testing.assert_allclose(
        np.asarray(video_output.astype(mx.float32)),
        golden["av_video_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(audio_output.astype(mx.float32)),
        golden["av_audio_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_video_cross_adaln_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    block = VideoTransformerBlock(
        VideoTransformerConfig(
            dim=golden["cross_adaln_input"].shape[-1],
            heads=int(golden["cross_adaln_heads"]),
            head_dim=int(golden["cross_adaln_head_dim"]),
            context_dim=golden["cross_adaln_context"].shape[-1],
            cross_attention_adaln=True,
        )
    )
    weights = [
        (name.removeprefix(CROSS_ADALN_WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
        for name, value in golden.items()
        if name.startswith(CROSS_ADALN_WEIGHT_PREFIX)
    ]
    block.load_weights(weights)
    result = block(
        mx.array(golden["cross_adaln_input"], dtype=mx.bfloat16),
        context=mx.array(golden["cross_adaln_context"], dtype=mx.bfloat16),
        timesteps=mx.array(golden["cross_adaln_timesteps"], dtype=mx.bfloat16),
        prompt_timestep=mx.array(golden["cross_adaln_prompt_timestep"], dtype=mx.bfloat16),
    )
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["cross_adaln_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )


def test_av_cross_adaln_matches_cuda_bf16(golden: dict[str, np.ndarray]) -> None:
    config = VideoTransformerConfig(
        dim=golden["cross_adaln_av_video_input"].shape[-1],
        heads=int(golden["cross_adaln_av_heads"]),
        head_dim=int(golden["cross_adaln_av_head_dim"]),
        context_dim=golden["cross_adaln_av_context"].shape[-1],
        cross_attention_adaln=True,
    )
    block = AVTransformerBlock(video=config, audio=config)
    weights = [
        (name.removeprefix(CROSS_ADALN_AV_BLOCK_WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
        for name, value in golden.items()
        if name.startswith(CROSS_ADALN_AV_BLOCK_WEIGHT_PREFIX)
    ]
    block.load_weights(weights)
    video_output, audio_output = block(
        TransformerStream(
            x=mx.array(golden["cross_adaln_av_video_input"], dtype=mx.bfloat16),
            context=mx.array(golden["cross_adaln_av_context"], dtype=mx.bfloat16),
            timesteps=mx.array(golden["cross_adaln_av_video_timesteps"], dtype=mx.bfloat16),
            prompt_timestep=mx.array(golden["cross_adaln_av_video_prompt_timestep"], dtype=mx.bfloat16),
            cross_scale_shift_timestep=mx.array(golden["cross_adaln_av_video_cross_timestep"], dtype=mx.bfloat16),
            cross_gate_timestep=mx.array(golden["cross_adaln_av_video_cross_gate_timestep"], dtype=mx.bfloat16),
        ),
        TransformerStream(
            x=mx.array(golden["cross_adaln_av_audio_input"], dtype=mx.bfloat16),
            context=mx.array(golden["cross_adaln_av_context"], dtype=mx.bfloat16),
            timesteps=mx.array(golden["cross_adaln_av_audio_timesteps"], dtype=mx.bfloat16),
            prompt_timestep=mx.array(golden["cross_adaln_av_audio_prompt_timestep"], dtype=mx.bfloat16),
            cross_scale_shift_timestep=mx.array(golden["cross_adaln_av_audio_cross_timestep"], dtype=mx.bfloat16),
            cross_gate_timestep=mx.array(golden["cross_adaln_av_audio_cross_gate_timestep"], dtype=mx.bfloat16),
        ),
    )
    assert video_output is not None and audio_output is not None
    np.testing.assert_allclose(
        np.asarray(video_output.astype(mx.float32)),
        golden["cross_adaln_av_video_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(audio_output.astype(mx.float32)),
        golden["cross_adaln_av_audio_output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
