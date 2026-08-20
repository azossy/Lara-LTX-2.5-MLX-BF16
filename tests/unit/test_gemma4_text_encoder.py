import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.text_encoder.gemma4 import Gemma4HiddenStateEncoder, ModelArgs


def _args() -> ModelArgs:
    return ModelArgs(
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        head_dim=4,
        global_head_dim=4,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=4,
        max_position_embeddings=32,
        attention_k_eq_v=True,
        use_double_wide_mlp=False,
        enable_moe_block=False,
        layer_types=["sliding_attention", "full_attention"],
    )


def test_gemma4_hidden_state_encoder_collects_embedding_layers_without_logits() -> None:
    encoder = Gemma4HiddenStateEncoder(_args())
    hidden_states = encoder(
        mx.array([[0, 0, 2, 7]], dtype=mx.int32),
        mx.array([[0, 0, 1, 1]], dtype=mx.int32),
    )
    mx.eval(hidden_states)

    assert hidden_states.shape == (1, 4, 8, 3)
    assert np.all(np.isfinite(np.asarray(hidden_states)))
    assert not hasattr(encoder, "lm_head")


def test_gemma4_hidden_state_encoder_rejects_right_padding() -> None:
    encoder = Gemma4HiddenStateEncoder(_args())

    with pytest.raises(LaraError) as raised:
        encoder(
            mx.array([[2, 7, 0, 0]], dtype=mx.int32),
            mx.array([[1, 1, 0, 0]], dtype=mx.int32),
        )

    assert raised.value.code == "LARA-TENSOR-022"
