from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.models import iter_component_weight_batches, load_safetensors_shard, map_loaded_shard


def _write_shard(path: Path, tensors: dict[str, mx.array]) -> None:
    mx.save_safetensors(path, tensors, metadata={"format": "pt"})


def test_load_safetensors_shard_selects_and_preserves_bf16(tmp_path: Path) -> None:
    path = tmp_path / "component.safetensors"
    keep = mx.array([[1.0, -2.0]], dtype=mx.bfloat16)
    _write_shard(path, {"keep": keep, "discard": mx.ones((2,), dtype=mx.bfloat16)})

    shard = load_safetensors_shard(path, required_names=("keep",))

    assert shard.source.metadata == {"format": "pt"}
    assert tuple(shard.tensors) == ("keep",)
    assert shard.tensors["keep"].dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.asarray(shard.tensors["keep"].astype(mx.float32)),
        np.asarray(keep.astype(mx.float32)),
    )


def test_load_safetensors_shard_rejects_missing_required_tensor(tmp_path: Path) -> None:
    path = tmp_path / "component.safetensors"
    _write_shard(path, {"available": mx.ones((1,), dtype=mx.bfloat16)})

    with pytest.raises(LaraError) as raised:
        load_safetensors_shard(path, required_names=("missing",))

    assert raised.value.code == "LARA-MODEL-010"


def test_load_safetensors_shard_rejects_non_bf16_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "component.safetensors"
    _write_shard(path, {"float": mx.ones((1,), dtype=mx.float32)})

    with pytest.raises(LaraError) as raised:
        load_safetensors_shard(path)

    assert raised.value.code == "LARA-MODEL-007"


def test_map_loaded_shard_applies_identity_and_transpose_with_shape_gate(tmp_path: Path) -> None:
    path = tmp_path / "component.safetensors"
    source = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.bfloat16)
    _write_shard(path, {"source": source})
    shard = load_safetensors_shard(path)
    rules = (
        {"source_file": path.name, "source_key": "source", "target_key": "identity", "transform": "identity"},
        {"source_file": path.name, "source_key": "source", "target_key": "transpose", "transform": "transpose_2d"},
    )

    mapped = dict(
        map_loaded_shard(
            shard,
            rules,
            expected_target_shapes={"identity": (2, 3), "transpose": (3, 2)},
        )
    )
    np.testing.assert_array_equal(
        np.asarray(mapped["identity"].astype(mx.float32)),
        np.asarray(source.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(mapped["transpose"].astype(mx.float32)),
        np.asarray(mx.transpose(source).astype(mx.float32)),
    )

    with pytest.raises(LaraError) as shape:
        map_loaded_shard(shard, rules, expected_target_shapes={"identity": (3, 2)})
    assert shape.value.code == "LARA-MODEL-020"

    vector_path = tmp_path / "vector.safetensors"
    _write_shard(vector_path, {"vector": mx.ones((2,), dtype=mx.bfloat16)})
    vector_shard = load_safetensors_shard(vector_path)
    with pytest.raises(LaraError) as rank:
        map_loaded_shard(
            vector_shard,
            (
                {
                    "source_file": vector_path.name,
                    "source_key": "vector",
                    "target_key": "target",
                    "transform": "transpose_2d",
                },
            ),
        )
    assert rank.value.code == "LARA-MODEL-019"


def test_map_loaded_shard_splits_fused_qkv_weights_and_biases(tmp_path: Path) -> None:
    path = tmp_path / "component.safetensors"
    weight = mx.arange(0, 18, dtype=mx.float32).reshape(6, 3).astype(mx.bfloat16)
    bias = mx.arange(0, 6, dtype=mx.float32).astype(mx.bfloat16)
    _write_shard(path, {"attn.qkv.weight": weight, "attn.qkv.bias": bias})
    shard = load_safetensors_shard(path)
    rules = tuple(
        {
            "source_file": path.name,
            "source_key": source_key,
            "target_key": f"{source_key}.{part}",
            "transform": f"split_qkv_{part}",
        }
        for source_key in ("attn.qkv.weight", "attn.qkv.bias")
        for part in ("q", "k", "v")
    )

    mapped = dict(map_loaded_shard(shard, rules))
    for index, part in enumerate(("q", "k", "v")):
        np.testing.assert_array_equal(
            np.asarray(mapped[f"attn.qkv.weight.{part}"].astype(mx.float32)),
            np.asarray(weight[index * 2 : (index + 1) * 2].astype(mx.float32)),
        )
        np.testing.assert_array_equal(
            np.asarray(mapped[f"attn.qkv.bias.{part}"].astype(mx.float32)),
            np.asarray(bias[index * 2 : (index + 1) * 2].astype(mx.float32)),
        )

    malformed_path = tmp_path / "malformed.safetensors"
    _write_shard(malformed_path, {"attn.qkv.weight": mx.ones((5, 3), dtype=mx.bfloat16)})
    malformed = load_safetensors_shard(malformed_path)
    with pytest.raises(LaraError) as raised:
        map_loaded_shard(
            malformed,
            (
                {
                    "source_file": malformed_path.name,
                    "source_key": "attn.qkv.weight",
                    "target_key": "attn.qkv.to_q.weight",
                    "transform": "split_qkv_q",
                },
            ),
        )
    assert raised.value.code == "LARA-MODEL-023"


def test_map_loaded_shard_folds_static_gate_in_fp32_before_bf16_cast(tmp_path: Path) -> None:
    path = tmp_path / "component.safetensors"
    weight = mx.array([[1.0, -2.0], [3.0, -4.0]], dtype=mx.bfloat16)
    bias = mx.array([5.0, -6.0], dtype=mx.bfloat16)
    gate = mx.array([0.5, -0.25], dtype=mx.bfloat16)
    _write_shard(path, {"weight": weight, "bias": bias, "gate": gate})
    shard = load_safetensors_shard(path)
    rules = (
        {
            "source_file": path.name,
            "source_key": "weight",
            "target_key": "target.weight",
            "transform": "fold_gate",
            "gate_source_key": "gate",
        },
        {
            "source_file": path.name,
            "source_key": "bias",
            "target_key": "target.bias",
            "transform": "fold_gate",
            "gate_source_key": "gate",
        },
    )

    mapped = dict(map_loaded_shard(shard, rules))
    np.testing.assert_array_equal(
        np.asarray(mapped["target.weight"].astype(mx.float32)),
        np.asarray(mx.array([[0.5, -1.0], [-0.75, 1.0]], dtype=mx.bfloat16).astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(mapped["target.bias"].astype(mx.float32)),
        np.asarray(mx.array([2.5, 1.5], dtype=mx.bfloat16).astype(mx.float32)),
    )

    bad_gate = mx.array([1.0], dtype=mx.bfloat16)
    bad_path = tmp_path / "bad-gate.safetensors"
    _write_shard(bad_path, {"weight": weight, "gate": bad_gate})
    bad_shard = load_safetensors_shard(bad_path)
    with pytest.raises(LaraError) as raised:
        map_loaded_shard(
            bad_shard,
            (
                {
                    "source_file": bad_path.name,
                    "source_key": "weight",
                    "target_key": "target.weight",
                    "transform": "fold_gate",
                    "gate_source_key": "gate",
                },
            ),
        )
    assert raised.value.code == "LARA-MODEL-024"


def test_component_weight_batches_are_shard_scoped_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "01.safetensors"
    second = tmp_path / "02.safetensors"
    _write_shard(first, {"first": mx.ones((1,), dtype=mx.bfloat16)})
    _write_shard(second, {"second": mx.ones((2,), dtype=mx.bfloat16)})
    rules = (
        {"source_file": second.name, "source_key": "second", "target_key": "target.second", "transform": "identity"},
        {"source_file": first.name, "source_key": "first", "target_key": "target.first", "transform": "identity"},
    )

    batches = list(iter_component_weight_batches((second, first), rules))

    assert [name for batch in batches for name, _ in batch] == ["target.first", "target.second"]
    with pytest.raises(LaraError) as missing:
        list(iter_component_weight_batches((first,), rules))
    assert missing.value.code == "LARA-MODEL-021"

    duplicate_directory = tmp_path / "duplicate"
    duplicate_directory.mkdir()
    duplicate_name = duplicate_directory / first.name
    _write_shard(duplicate_name, {"first": mx.ones((1,), dtype=mx.bfloat16)})
    with pytest.raises(LaraError) as duplicated:
        list(iter_component_weight_batches((first, duplicate_name), ()))
    assert duplicated.value.code == "LARA-MODEL-022"
