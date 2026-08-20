from __future__ import annotations

import pytest
from lara_ltx.duration_head_contract import DurationHeadConfig, duration_head_target_keys, duration_head_target_shapes
from lara_ltx.errors import LaraError


def test_duration_head_contract_matches_official_parameter_count() -> None:
    keys = duration_head_target_keys()

    assert len(keys) == 19
    assert "attention_pooler.cross_attn.to_q.weight" in keys
    assert "attention_pooler.cross_attn.in_proj_weight" not in keys


def test_duration_head_contract_accepts_official_dimensions() -> None:
    config = DurationHeadConfig()

    assert config.pooler_hidden_dim == 256
    assert duration_head_target_shapes(config)["mlp_hidden.weight"] == (256, 256)
    assert duration_head_target_shapes(config)["attention_pooler.query_tokens"] == (1, 256)


def test_duration_head_contract_raises_localized_error_for_invalid_dimensions() -> None:
    with pytest.raises(LaraError) as raised:
        DurationHeadConfig(pooler_hidden_dim=255, num_pooler_heads=4)

    assert raised.value.code == "LARA-TENSOR-017"

    with pytest.raises(LaraError) as zero_dimension:
        DurationHeadConfig(video_cross_attention_dim=0)

    assert zero_dimension.value.code == "LARA-TENSOR-017"
