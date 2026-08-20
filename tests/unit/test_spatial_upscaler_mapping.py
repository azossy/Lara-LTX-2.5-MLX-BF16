from __future__ import annotations

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import spatial_upscaler_target_shapes, validate_spatial_upscaler_mapping

BF16_DTYPE = "BF16"


def _mapping(target_shapes: dict[str, tuple[int, ...]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "upscaler.safetensors",
                "source_key": target,
                "target_key": target,
                "transform": "identity",
                "dtype": BF16_DTYPE,
                "shape": list(shape),
            }
            for target, shape in target_shapes.items()
        ],
    }


def test_spatial_upscaler_mapping_has_full_x2_source_layout() -> None:
    target_shapes = spatial_upscaler_target_shapes()

    assert len(target_shapes) == 72
    assert len(validate_spatial_upscaler_mapping(_mapping(target_shapes))) == 72


def test_spatial_upscaler_mapping_rejects_missing_or_wrong_layout() -> None:
    target_shapes = spatial_upscaler_target_shapes()
    missing = dict(target_shapes)
    missing.pop("upsampler.0.weight")
    with pytest.raises(LaraError) as missing_error:
        validate_spatial_upscaler_mapping(_mapping(missing))

    assert missing_error.value.code == "LARA-MODEL-017"

    incompatible = dict(target_shapes)
    incompatible["initial_conv.weight"] = (1, 1, 1, 1, 1)
    with pytest.raises(LaraError) as incompatible_error:
        validate_spatial_upscaler_mapping(_mapping(incompatible))

    assert incompatible_error.value.code == "LARA-MODEL-030"
