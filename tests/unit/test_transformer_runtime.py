from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.sampling import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from lara_ltx.transformer import (
    AVTransformerBlock,
    TransformerStream,
    VideoTransformerConfig,
    run_transformer_block_sequence,
)


def _small_block() -> AVTransformerBlock:
    config = VideoTransformerConfig(
        dim=8,
        heads=2,
        head_dim=4,
        context_dim=8,
        apply_gated_attention=False,
        cross_attention_adaln=False,
        ff_bias=True,
    )
    return AVTransformerBlock(video=config)


def _stream() -> TransformerStream:
    return TransformerStream(
        x=mx.ones((1, 3, 8), dtype=mx.float32),
        context=mx.ones((1, 2, 8), dtype=mx.float32),
        timesteps=mx.zeros((1, 3, 48), dtype=mx.float32),
    )


def test_sequence_materializes_each_block_and_preserves_stream_metadata() -> None:
    source = _stream()

    result = run_transformer_block_sequence(source, None, ((0, _small_block()), (1, _small_block())))

    assert result.completed_block_count == 2
    assert result.video is not None
    assert result.video.context is source.context
    assert result.video.x.shape == source.x.shape
    assert np.isfinite(np.asarray(result.video.x)).all()


def test_sequence_rejects_a_block_with_no_active_modality() -> None:
    with pytest.raises(LaraError) as error:
        run_transformer_block_sequence(None, None, ((7, _small_block()),))

    assert error.value.code == "LARA-TENSOR-028"


def test_sequence_attaches_configured_perturbations_before_the_target_block() -> None:
    perturbations = BatchedPerturbationConfig(
        (
            PerturbationConfig(
                (Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=(0,)),),
            ),
        ),
        num_blocks=1,
        dtype=mx.float32,
    )

    result = run_transformer_block_sequence(
        _stream(),
        None,
        ((0, _small_block()),),
        perturbations=perturbations,
    )

    assert result.video is not None
    assert result.video.self_attn_all_perturbed is True
