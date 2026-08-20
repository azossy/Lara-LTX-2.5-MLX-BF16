from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import (
    build_diffusion_vae_decoder_mapping,
    build_duration_head_mapping,
    build_mapping_template,
    load_and_validate_mapping,
    validate_duration_head_mapping,
    validate_mapping,
    write_mapping_template,
)


def _write_safetensors(path: Path, tensor_name: str) -> None:
    header = {tensor_name: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(2))


def _write_safetensors_header(path: Path, tensors: dict[str, tuple[int, ...]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    for name, shape in tensors.items():
        size = 2
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def test_mapping_template_is_header_derived_and_deterministic(tmp_path: Path) -> None:
    second = tmp_path / "02.safetensors"
    first = tmp_path / "01.safetensors"
    _write_safetensors(first, "first.weight")
    _write_safetensors(second, "second.weight")

    template = build_mapping_template((second, first))

    assert template["schema_version"] == 2
    assert [item["file"] for item in template["shards"]] == ["01.safetensors", "02.safetensors"]
    assert [item["source_key"] for item in template["rules"]] == ["first.weight", "second.weight"]
    assert all(item["target_key"] is None for item in template["rules"])


def test_mapping_template_rejects_cross_shard_duplicate_keys(tmp_path: Path) -> None:
    first = tmp_path / "01.safetensors"
    second = tmp_path / "02.safetensors"
    _write_safetensors(first, "duplicate")
    _write_safetensors(second, "duplicate")

    with pytest.raises(LaraError) as raised:
        build_mapping_template((first, second))

    assert raised.value.code == "LARA-MODEL-012"


def test_diffusion_vae_mapping_audits_type_embedding_and_qkv_transform(tmp_path: Path) -> None:
    shard = tmp_path / "video-vae.safetensors"
    _write_safetensors_header(
        shard,
        {
            "encoder.conv_in.weight": (1,),
            "decoder.type_emb": (1,),
            "decoder.t_embedder.mlp.0.weight": (2, 3),
            "decoder.det_stages.0.0.attn.qkv.weight": (6, 2),
            "per_channel_statistics.std-of-means": (2,),
        },
    )

    mapping = build_diffusion_vae_decoder_mapping(shard)

    assert mapping["component"] == "diffusion_video_decoder"
    assert mapping["ignored_sources"] == [{"source_key": "decoder.type_emb", "reason": "upstream_load_artifact"}]
    rules = mapping["rules"]
    assert isinstance(rules, list)
    targets = {rule["target_key"] for rule in rules if isinstance(rule, dict)}
    assert targets == {
        "t_embedder.timestep_embedder.linear_1.weight",
        "det_stages.0.0.attn.qkv.to_q.weight",
        "det_stages.0.0.attn.qkv.to_k.weight",
        "det_stages.0.0.attn.qkv.to_v.weight",
        "per_channel_statistics.std_of_means",
    }
    assert len(validate_mapping(mapping, expected_target_keys=targets)) == len(rules)


def test_diffusion_vae_mapping_rejects_missing_type_embedding(tmp_path: Path) -> None:
    missing = tmp_path / "missing.safetensors"
    _write_safetensors_header(missing, {"decoder.conv_in.weight": (1,)})
    with pytest.raises(LaraError) as missing_type_embedding:
        build_diffusion_vae_decoder_mapping(missing)
    assert missing_type_embedding.value.code == "LARA-MODEL-014"


def test_duration_head_mapping_splits_pytorch_fused_qkv_and_strips_prefix(tmp_path: Path) -> None:
    shard = tmp_path / "duration-head.safetensors"
    _write_safetensors_header(
        shard,
        {
            "duration_head.attention_pooler.cross_attn.in_proj_weight": (768, 256),
            "duration_head.attention_pooler.cross_attn.in_proj_bias": (768,),
            "duration_head.attention_pooler.cross_attn.out_proj.weight": (256, 256),
            "duration_head.mlp_out.bias": (1,),
        },
    )

    mapping = build_duration_head_mapping(shard)

    rules = mapping["rules"]
    assert isinstance(rules, list)
    targets = {rule["target_key"] for rule in rules if isinstance(rule, dict)}
    assert targets == {
        "attention_pooler.cross_attn.to_q.weight",
        "attention_pooler.cross_attn.to_k.weight",
        "attention_pooler.cross_attn.to_v.weight",
        "attention_pooler.cross_attn.to_q.bias",
        "attention_pooler.cross_attn.to_k.bias",
        "attention_pooler.cross_attn.to_v.bias",
        "attention_pooler.cross_attn.out_proj.weight",
        "mlp_out.bias",
    }
    assert len(validate_mapping(mapping, expected_target_keys=targets)) == len(rules)


def test_official_duration_head_mapping_manifest_matches_mlx_contract() -> None:
    manifest_path = Path("golden/manifests/duration_head_mapping.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["component"] == "duration_head"
    assert len(manifest["rules"]) == 19
    assert sum(rule["transform"].startswith("split_qkv") for rule in manifest["rules"]) == 6
    assert len(validate_duration_head_mapping(manifest)) == len(manifest["rules"])


def test_duration_head_mapping_rejects_shape_incompatible_qkv_source(tmp_path: Path) -> None:
    shard = tmp_path / "duration-head.safetensors"
    _write_safetensors_header(
        shard,
        {
            "duration_head.video_input_proj.weight": (256, 4096),
            "duration_head.video_input_proj.bias": (256,),
            "duration_head.video_modality_emb": (256,),
            "duration_head.audio_input_proj.weight": (256, 2048),
            "duration_head.audio_input_proj.bias": (256,),
            "duration_head.audio_modality_emb": (256,),
            "duration_head.attention_pooler.query_tokens": (1, 256),
            "duration_head.attention_pooler.cross_attn.in_proj_weight": (767, 256),
            "duration_head.attention_pooler.cross_attn.in_proj_bias": (768,),
            "duration_head.attention_pooler.cross_attn.out_proj.weight": (256, 256),
            "duration_head.attention_pooler.cross_attn.out_proj.bias": (256,),
            "duration_head.mlp_hidden.weight": (256, 256),
            "duration_head.mlp_hidden.bias": (256,),
            "duration_head.mlp_out.weight": (1, 256),
            "duration_head.mlp_out.bias": (1,),
        },
    )

    with pytest.raises(LaraError) as raised:
        validate_duration_head_mapping(build_duration_head_mapping(shard))

    assert raised.value.code == "LARA-MODEL-025"


def test_official_diffusion_vae_mapping_manifest_has_exact_transformation_accounting() -> None:
    manifest_path = Path("golden/manifests/vae_bf16_decoder_mapping.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rules = manifest["rules"]

    assert manifest["component"] == "diffusion_video_decoder"
    assert manifest["ignored_sources"] == [{"reason": "upstream_load_artifact", "source_key": "decoder.type_emb"}]
    assert len(rules) == 407
    assert sum(rule["transform"].startswith("split_qkv") for rule in rules) == 144
    targets = {rule["target_key"] for rule in rules}
    assert len(targets) == len(rules)
    assert len(validate_mapping(manifest, expected_target_keys=targets)) == len(rules)


def test_mapping_template_write_is_atomic_and_reports_bad_output_path(tmp_path: Path) -> None:
    shard = tmp_path / "01.safetensors"
    _write_safetensors(shard, "weight")
    output = tmp_path / "mapping.json"

    write_mapping_template(output, (shard,))

    assert json.loads(output.read_text(encoding="utf-8"))["rules"][0]["source_key"] == "weight"
    with pytest.raises(LaraError) as raised:
        write_mapping_template(tmp_path / "missing" / "mapping.json", (shard,))

    assert raised.value.code == "LARA-MODEL-009"


def test_mapping_validation_requires_reviewed_unique_supported_targets(tmp_path: Path) -> None:
    shard = tmp_path / "01.safetensors"
    _write_safetensors(shard, "source.weight")
    template = build_mapping_template((shard,))

    with pytest.raises(LaraError) as unresolved:
        validate_mapping(template)
    assert unresolved.value.code == "LARA-MODEL-014"

    rule = template["rules"][0]
    assert isinstance(rule, dict)
    rule["target_key"] = "target.weight"
    assert validate_mapping(template, expected_target_keys=("target.weight",)) == (rule,)

    duplicate = dict(rule)
    duplicate["source_key"] = "other.weight"
    template["rules"].append(duplicate)
    with pytest.raises(LaraError) as repeated:
        validate_mapping(template)
    assert repeated.value.code == "LARA-MODEL-015"


def test_mapping_validation_reports_transform_target_and_file_errors(tmp_path: Path) -> None:
    mapping = {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "component.safetensors",
                "source_key": "source.weight",
                "target_key": "target.weight",
                "transform": "invalid",
            }
        ],
    }
    with pytest.raises(LaraError) as transform:
        validate_mapping(mapping)
    assert transform.value.code == "LARA-MODEL-016"

    mapping["rules"][0]["transform"] = "identity"
    with pytest.raises(LaraError) as targets:
        validate_mapping(mapping, expected_target_keys=("other.weight",))
    assert targets.value.code == "LARA-MODEL-017"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(LaraError) as unreadable:
        load_and_validate_mapping(malformed)
    assert unreadable.value.code == "LARA-MODEL-014"


def test_mapping_validation_rejects_duplicate_source_rule() -> None:
    mapping = {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "component.safetensors",
                "source_key": "source.weight",
                "target_key": "first.weight",
                "transform": "identity",
            },
            {
                "source_file": "component.safetensors",
                "source_key": "source.weight",
                "target_key": "second.weight",
                "transform": "identity",
            },
        ],
    }

    with pytest.raises(LaraError) as duplicate:
        validate_mapping(mapping)

    assert duplicate.value.code == "LARA-MODEL-018"


def test_mapping_validation_allows_only_complete_qkv_split_groups() -> None:
    source = {"source_file": "component.safetensors", "source_key": "attn.qkv.weight"}
    complete = {
        "schema_version": 2,
        "rules": [
            {**source, "target_key": "attn.qkv.to_q.weight", "transform": "split_qkv_q"},
            {**source, "target_key": "attn.qkv.to_k.weight", "transform": "split_qkv_k"},
            {**source, "target_key": "attn.qkv.to_v.weight", "transform": "split_qkv_v"},
        ],
    }

    assert len(validate_mapping(complete)) == 3

    incomplete = dict(complete)
    incomplete["rules"] = complete["rules"][:2]
    with pytest.raises(LaraError) as raised:
        validate_mapping(incomplete)
    assert raised.value.code == "LARA-MODEL-018"


def test_mapping_validation_requires_gate_source_only_for_gate_folding() -> None:
    mapping = {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "component.safetensors",
                "source_key": "block.attn.proj.weight",
                "target_key": "block.attn.proj.weight",
                "transform": "fold_gate",
                "gate_source_key": "block.gate_msa",
            }
        ],
    }
    assert len(validate_mapping(mapping)) == 1

    missing_gate = {**mapping, "rules": [{**mapping["rules"][0], "gate_source_key": None}]}
    with pytest.raises(LaraError) as missing:
        validate_mapping(missing_gate)
    assert missing.value.code == "LARA-MODEL-014"

    unexpected_gate = {**mapping, "rules": [{**mapping["rules"][0], "transform": "identity"}]}
    with pytest.raises(LaraError) as unexpected:
        validate_mapping(unexpected_gate)
    assert unexpected.value.code == "LARA-MODEL-014"


def test_mapping_validation_allows_reviewed_f32_and_rejects_unmappable_asset_dtype() -> None:
    reviewed_f32 = {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "component.safetensors",
                "source_key": "block.scale_shift_table",
                "target_key": "block.scale_shift_table",
                "transform": "identity",
                "dtype": "F32",
            }
        ],
    }
    assert len(validate_mapping(reviewed_f32)) == 1

    unreviewed_asset = {
        **reviewed_f32,
        "rules": [{**reviewed_f32["rules"][0], "dtype": "U8"}],
    }
    with pytest.raises(LaraError) as raised:
        validate_mapping(unreviewed_asset)
    assert raised.value.code == "LARA-MODEL-004"
