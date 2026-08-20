from lara_ltx.models.gemma_text import gemma_text_target_shapes, validate_gemma_text_mapping


def _packed_config() -> dict[str, object]:
    return {
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "intermediate_size": 15360,
            "num_attention_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "vocab_size": 262144,
            "enable_moe_block": False,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": False,
            "attention_k_eq_v": True,
            "layer_types": ["full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(48)],
        },
    }


def test_gemma_text_mapping_contract_matches_official_666_keys() -> None:
    config = _packed_config()
    shapes = gemma_text_target_shapes(config)
    rules = [
        {
            "source_file": "gemma.safetensors",
            "source_key": target,
            "target_key": target,
            "transform": "identity",
            "dtype": "BF16",
            "shape": list(shape),
        }
        for target, shape in shapes.items()
    ]
    mapping = {
        "schema_version": 2,
        "component": "gemma4_text_core",
        "shards": [{"file": "gemma.safetensors", "tensor_count": len(rules)}],
        "rules": rules,
    }

    assert len(validate_gemma_text_mapping(mapping, config)) == 666
    assert shapes["model.layers.0.self_attn.q_proj.weight"] == (4096, 3840)
    assert shapes["model.layers.5.self_attn.q_proj.weight"] == (8192, 3840)
    assert "model.layers.5.self_attn.v_proj.weight" not in shapes
