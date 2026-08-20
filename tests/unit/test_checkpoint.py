import json
import struct
from pathlib import Path

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import inspect_safetensors


def _write_safetensors(path: Path, header: dict, payload: bytes) -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_inspection_reads_metadata_and_bf16_descriptors(tmp_path: Path) -> None:
    path = tmp_path / "weights.safetensors"
    header = {
        "__metadata__": {"format": "pt"},
        "block.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
    }
    _write_safetensors(path, header, bytes(12))

    inspected = inspect_safetensors(path)

    assert inspected.tensor_count == 1
    assert inspected.total_tensor_bytes == 12
    assert inspected.metadata == {"format": "pt"}
    assert inspected.tensors[0].shape == (2, 3)
    inspected.require_dtype("BF16")


def test_inspection_rejects_tensor_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "broken.safetensors"
    header = {"bad": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 10]}}
    _write_safetensors(path, header, bytes(10))

    with pytest.raises(LaraError) as raised:
        inspect_safetensors(path)

    assert raised.value.code == "LARA-MODEL-005"


def test_inspection_rejects_overlapping_tensors(tmp_path: Path) -> None:
    path = tmp_path / "overlap.safetensors"
    header = {
        "first": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "second": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
    }
    _write_safetensors(path, header, bytes(12))

    with pytest.raises(LaraError) as raised:
        inspect_safetensors(path)

    assert raised.value.code == "LARA-MODEL-008"


def test_inspection_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.safetensors"
    encoded = (
        b'{"same":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
        b'"same":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    )
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(1))

    with pytest.raises(LaraError) as raised:
        inspect_safetensors(path)

    assert raised.value.code == "LARA-MODEL-003"
