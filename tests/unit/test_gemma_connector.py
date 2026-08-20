from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models.gemma_connector import PromptConnectorConfig, prompt_connector_target_shapes
from lara_ltx.text_encoder.connector import PromptConnectorProcessor


def _tiny_config() -> PromptConnectorConfig:
    return PromptConnectorConfig(
        video_dimensions=4,
        audio_dimensions=4,
        video_attention_heads=2,
        audio_attention_heads=2,
        video_attention_head_dim=2,
        audio_attention_head_dim=2,
        layer_count=1,
        register_count=2,
        positional_theta=10_000.0,
        positional_maximum=16,
        apply_gated_attention=True,
        double_precision_frequencies=True,
    )


def test_prompt_connector_shapes_cover_both_modalities() -> None:
    shapes = prompt_connector_target_shapes(_tiny_config())

    assert len(shapes) == 34
    assert shapes["video_connector.learnable_registers"] == (2, 4)
    assert shapes["audio_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight"] == (2, 4)
    assert shapes["video_connector.transformer_1d_blocks.0.ff.net.0.proj.weight"] == (16, 4)


def test_prompt_connector_reorders_left_padding_and_preserves_register_outputs() -> None:
    processor = PromptConnectorProcessor(_tiny_config())
    video = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4).astype(mx.bfloat16)
    audio = (video + 1).astype(mx.bfloat16)
    left_mask = mx.array([[0, 0, 1, 1]], dtype=mx.int32)

    connected_video, connected_audio, right_mask = processor(video, audio, left_mask)
    mx.eval(connected_video, connected_audio, right_mask)

    np.testing.assert_array_equal(np.asarray(right_mask), [[1, 1, 0, 0]])
    assert np.isfinite(np.asarray(connected_video.astype(mx.float32))).all()
    assert np.isfinite(np.asarray(connected_audio.astype(mx.float32))).all()
    assert np.linalg.norm(np.asarray(connected_video[:, 2:].astype(mx.float32))) > 0


def test_prompt_connector_rejects_incompatible_sequence_length() -> None:
    processor = PromptConnectorProcessor(_tiny_config())
    features = mx.zeros((1, 3, 4), dtype=mx.bfloat16)
    mask = mx.ones((1, 3), dtype=mx.int32)

    with pytest.raises(LaraError) as raised:
        processor(features, features, mask)

    assert raised.value.code == "LARA-TENSOR-031"
