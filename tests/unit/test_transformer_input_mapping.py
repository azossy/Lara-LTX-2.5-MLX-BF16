from lara_ltx.models.transformer_input import (
    KEYFRAME_SOURCE,
    KEYFRAME_TARGET,
    SOURCE_MODULE_TO_TARGET,
    transformer_input_target_shapes,
    validate_transformer_input_mapping,
)


def _source_key(target: str) -> str:
    if target == KEYFRAME_TARGET:
        return f"model.diffusion_model.{KEYFRAME_SOURCE}"
    for source_module, target_module in SOURCE_MODULE_TO_TARGET.items():
        prefix = f"{target_module}."
        if target.startswith(prefix):
            return f"model.diffusion_model.{source_module}.{target.removeprefix(prefix)}"
    raise AssertionError(target)


def test_transformer_input_mapping_contract_has_exact_production_shapes() -> None:
    shapes = transformer_input_target_shapes()
    rules = [
        {
            "source_file": "transformer.safetensors",
            "source_key": _source_key(target),
            "target_key": target,
            "transform": "identity",
            "dtype": "BF16",
            "shape": list(shape),
        }
        for target, shape in shapes.items()
    ]
    mapping = {
        "schema_version": 2,
        "component": "transformer_input",
        "shards": [{"file": "transformer.safetensors", "tensor_count": len(rules)}],
        "rules": rules,
    }

    assert len(validate_transformer_input_mapping(mapping)) == 53
    assert shapes["video.patchify_proj.weight"] == (4096, 128)
    assert shapes["audio.adaln_single.linear.weight"] == (9 * 2048, 2048)
    assert shapes["video_cross_gate.linear.bias"] == (4096,)
