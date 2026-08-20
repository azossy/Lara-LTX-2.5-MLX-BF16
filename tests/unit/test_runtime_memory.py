import pytest
from lara_ltx.errors import LaraError
from lara_ltx.runtime.memory import (
    AttentionShape,
    bf16_attention_io_bytes,
    bf16_transformer_block_probe_bytes,
    stage_video_tokens,
)


def test_canonical_hq_token_counts() -> None:
    common = {"frames": 121, "time_scale": 8, "spatial_scale": 32}

    assert stage_video_tokens(height=544, width=960, **common) == 8_160
    assert stage_video_tokens(height=1_088, width=1_920, **common) == 32_640


def test_bf16_attention_io_estimate_has_no_score_matrix() -> None:
    shape = AttentionShape(batch_size=1, heads=32, tokens=32_640, head_dim=128)

    assert bf16_attention_io_bytes(shape) == 1_069_547_520


def test_non_aligned_video_grid_is_rejected() -> None:
    with pytest.raises(LaraError) as raised:
        stage_video_tokens(frames=120, height=1_088, width=1_920, time_scale=8, spatial_scale=32)

    assert raised.value.code == "LARA-CONFIG-003"


def test_transformer_block_probe_estimate_is_bounded_and_score_free() -> None:
    estimated = bf16_transformer_block_probe_bytes(
        tokens=32_640,
        hidden_size=4_096,
        feed_forward_multiplier=4,
    )

    assert estimated == 3_343_908_864
