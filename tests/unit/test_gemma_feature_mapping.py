from __future__ import annotations

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import (
    gemma_feature_target_shapes,
    validate_gemma_feature_mapping,
)

SOURCE_PREFIX = "text_embedding_projection."
TARGET_PREFIX = "feature_extractor."
BF16_DTYPE = "BF16"


def _mapping(target_shapes: dict[str, tuple[int, ...]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "gemma.safetensors",
                "source_key": f"{SOURCE_PREFIX}{target.removeprefix(TARGET_PREFIX)}",
                "target_key": target,
                "transform": "identity",
                "dtype": BF16_DTYPE,
                "shape": list(shape),
            }
            for target, shape in target_shapes.items()
        ],
    }


def test_gemma_feature_mapping_selects_only_four_bf16_projection_tensors() -> None:
    target_shapes = gemma_feature_target_shapes()

    assert len(target_shapes) == 4
    assert len(validate_gemma_feature_mapping(_mapping(target_shapes))) == 4


def test_gemma_feature_mapping_rejects_missing_or_incompatible_projection() -> None:
    target_shapes = gemma_feature_target_shapes()
    missing = dict(target_shapes)
    missing.pop("feature_extractor.audio_aggregate_embed.bias")
    with pytest.raises(LaraError) as missing_error:
        validate_gemma_feature_mapping(_mapping(missing))

    assert missing_error.value.code == "LARA-MODEL-017"

    incompatible = dict(target_shapes)
    incompatible["feature_extractor.video_aggregate_embed.weight"] = (1, 1)
    with pytest.raises(LaraError) as incompatible_error:
        validate_gemma_feature_mapping(_mapping(incompatible))

    assert incompatible_error.value.code == "LARA-MODEL-029"
