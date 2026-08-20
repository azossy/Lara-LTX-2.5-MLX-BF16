from types import SimpleNamespace

from tools.models.verify_bf16_pack import _dtype_counts, _unexpected_non_bf16_tensors


def test_dtype_counts_sorts_and_counts_tensor_dtypes() -> None:
    tensors = [SimpleNamespace(dtype="BF16"), SimpleNamespace(dtype="F32"), SimpleNamespace(dtype="BF16")]

    assert _dtype_counts(tensors) == {"BF16": 2, "F32": 1}


def test_f32_official_modulation_tables_are_explicitly_allowed() -> None:
    tensors = [
        SimpleNamespace(dtype="BF16", name="model.diffusion_model.transformer_blocks.0.attn.to_q.weight"),
        SimpleNamespace(dtype="F32", name="model.diffusion_model.scale_shift_table"),
        SimpleNamespace(
            dtype="F32",
            name="model.diffusion_model.transformer_blocks.47.scale_shift_table_a2v_ca_video",
        ),
    ]

    assert _unexpected_non_bf16_tensors(tensors) == []


def test_unknown_f32_tensor_is_rejected() -> None:
    tensors = [SimpleNamespace(dtype="F32", name="model.diffusion_model.unreviewed_tensor")]

    assert _unexpected_non_bf16_tensors(tensors) == ["model.diffusion_model.unreviewed_tensor:F32"]


def test_official_gemma_serialized_assets_are_explicitly_allowed() -> None:
    tensors = [
        SimpleNamespace(dtype="U8", name="hf_asset__tokenizer_config.json"),
        SimpleNamespace(dtype="U8", name="tokenizer_json"),
    ]

    assert _unexpected_non_bf16_tensors(tensors) == []


def test_unreviewed_u8_asset_is_rejected() -> None:
    tensors = [SimpleNamespace(dtype="U8", name="hf_asset__unreviewed.json")]

    assert _unexpected_non_bf16_tensors(tensors) == ["hf_asset__unreviewed.json:U8"]
