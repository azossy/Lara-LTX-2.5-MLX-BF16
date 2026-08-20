from __future__ import annotations

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import (
    transformer_block_target_dtypes,
    transformer_block_target_shapes,
    validate_transformer_block_mapping,
)

SOURCE_PREFIX = "model.diffusion_model.transformer_blocks."


def _mapping(target_shapes: dict[str, tuple[int, ...]]) -> dict[str, object]:
    target_dtypes = transformer_block_target_dtypes()
    return {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "transformer.safetensors",
                "source_key": f"{SOURCE_PREFIX}0.{name}",
                "target_key": name,
                "transform": "identity",
                "dtype": target_dtypes[name],
                "shape": list(shape),
            }
            for name, shape in target_shapes.items()
        ],
    }


def test_transformer_block_mapping_requires_all_84_production_targets() -> None:
    target_shapes = transformer_block_target_shapes()
    mapping = _mapping(target_shapes)

    assert len(mapping["rules"]) == 84
    assert len(validate_transformer_block_mapping(mapping)) == 84


def test_transformer_block_mapping_rejects_missing_or_wrong_weight_layout() -> None:
    target_shapes = transformer_block_target_shapes()
    broken = dict(target_shapes)
    broken.pop("audio_prompt_scale_shift_table")
    broken["attn1.to_q.weight"] = (1, 1)

    with pytest.raises(LaraError) as raised:
        validate_transformer_block_mapping(_mapping(broken))

    assert raised.value.code == "LARA-MODEL-017"

    malformed = dict(target_shapes)
    malformed["attn1.to_q.weight"] = (1, 1)
    with pytest.raises(LaraError) as wrong_shape:
        validate_transformer_block_mapping(_mapping(malformed))

    assert wrong_shape.value.code == "LARA-MODEL-028"
