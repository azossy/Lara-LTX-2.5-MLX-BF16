import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.transformer import AdaLayerNormSingle, TransformerInputConfig, TransformerInputPreprocessor


def test_adaln_single_returns_projected_and_embedded_timesteps() -> None:
    layer = AdaLayerNormSingle(hidden_dimension=8, coefficient=3)
    timestep = mx.array([0.25, 0.5], dtype=mx.bfloat16)

    projected, embedded = layer(timestep, hidden_dtype=mx.bfloat16)
    mx.eval(projected, embedded)

    assert projected.shape == (2, 24)
    assert embedded.shape == (2, 8)


def test_transformer_input_prepares_checkpoint_layout_and_masks() -> None:
    processor = TransformerInputPreprocessor(
        TransformerInputConfig(
            input_channels=4,
            hidden_dimension=8,
            adaln_coefficient=6,
            prompt_adaln_coefficient=2,
            attention_heads=2,
            max_positions=(20, 64, 64),
            use_keyframes_absolute_embedding=True,
        )
    )
    processor.keyframes_abs_pos_embedding = mx.ones((1, 8), dtype=mx.bfloat16)
    prepared = processor.prepare(
        latent=mx.ones((1, 2, 4), dtype=mx.bfloat16),
        timesteps=mx.array([[0.1, 0.2]], dtype=mx.bfloat16),
        sigma=mx.array([0.3], dtype=mx.bfloat16),
        positions=mx.array([[[[0, 1], [1, 2]], [[0, 2], [2, 4]], [[0, 2], [2, 4]]]], dtype=mx.float32),
        context=mx.ones((1, 3, 8), dtype=mx.bfloat16),
        context_mask=mx.array([[1, 1, 0]], dtype=mx.int32),
        attention_mask=mx.array([[[1.0, 0.5], [0.0, 1.0]]], dtype=mx.float32),
        keyframes_mask=mx.array([[[1], [0]]], dtype=mx.int32),
    )
    mx.eval(prepared.stream.x, prepared.stream.timesteps, prepared.embedded_timestep)

    assert prepared.stream.x.shape == (1, 2, 8)
    assert prepared.stream.timesteps.shape == (1, 2, 48)
    assert prepared.embedded_timestep.shape == (1, 2, 8)
    assert prepared.stream.prompt_timestep is not None
    assert prepared.stream.prompt_timestep.shape == (1, 1, 16)
    assert prepared.stream.context.shape == (1, 3, 8)
    assert prepared.stream.context_mask is not None
    assert prepared.stream.context_mask.shape == (1, 1, 1, 3)
    assert prepared.stream.self_attention_mask is not None
    assert prepared.stream.self_attention_mask.shape == (1, 1, 2, 2)
    assert np.asarray(prepared.stream.self_attention_mask.astype(mx.float32))[0, 0, 0, 1] == pytest.approx(
        np.log(0.5),
        abs=1e-2,
    )


def test_transformer_input_rejects_non_patchified_latent() -> None:
    processor = TransformerInputPreprocessor(
        TransformerInputConfig(
            input_channels=4,
            hidden_dimension=8,
            adaln_coefficient=6,
            attention_heads=2,
            max_positions=(20, 64, 64),
        )
    )

    with pytest.raises(LaraError) as raised:
        processor.prepare(
            latent=mx.ones((1, 2, 5)),
            timesteps=mx.ones((1, 2)),
            positions=mx.ones((1, 3, 2, 2)),
            context=mx.ones((1, 3, 8)),
        )

    assert raised.value.code == "LARA-TENSOR-021"
