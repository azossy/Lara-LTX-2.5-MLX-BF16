from __future__ import annotations

from lara_ltx.models.transformer_output import (
    OUTPUT_SOURCE_TO_TARGET,
    transformer_output_target_dtypes,
    transformer_output_target_shapes,
    validate_transformer_output_mapping,
)


def test_transformer_output_mapping_contract_covers_both_modalities() -> None:
    shapes = transformer_output_target_shapes()
    dtypes = transformer_output_target_dtypes()
    rules = [
        {
            "source_file": "transformer.safetensors",
            "source_key": f"model.diffusion_model.{source}",
            "target_key": target,
            "transform": "identity",
            "dtype": dtypes[target],
            "shape": list(shapes[target]),
        }
        for source, target in OUTPUT_SOURCE_TO_TARGET.items()
    ]
    mapping = {
        "schema_version": 2,
        "component": "transformer_output",
        "shards": [{"file": "transformer.safetensors", "tensor_count": len(rules)}],
        "rules": rules,
    }

    assert len(validate_transformer_output_mapping(mapping)) == 6
    assert shapes["video.proj_out.weight"] == (128, 4096)
    assert shapes["audio.proj_out.weight"] == (128, 2048)
    assert dtypes["video.scale_shift_table"] == "F32"
    assert dtypes["audio.proj_out.weight"] == "BF16"
