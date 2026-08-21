from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "models" / "download_safetensors_subset.py"


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("download_safetensors_subset_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_header_rebases_selected_tensor_offsets() -> None:
    tool = _tool()
    original = {
        "__metadata__": {"config": "kept"},
        "block.0.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [10, 14]},
        "block.0.bias": {"dtype": "BF16", "shape": [1], "data_offsets": [20, 22]},
        "block.1.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [30, 34]},
    }

    selected = tool._select_tensors(original, ("block.0.",))
    encoded, ranges = tool._compact_header(original, selected)
    compact = json.loads(encoded)

    assert len(encoded) % tool.HEADER_ALIGNMENT_BYTES == 0
    assert compact["__metadata__"] == {"config": "kept"}
    assert compact["block.0.weight"]["data_offsets"] == [0, 4]
    assert compact["block.0.bias"]["data_offsets"] == [4, 6]
    assert "block.1.weight" not in compact
    assert ranges == [("block.0.weight", 10, 14), ("block.0.bias", 20, 22)]


def test_read_header_accepts_trailing_payload(tmp_path: Path) -> None:
    tool = _tool()
    header = json.dumps({"tensor": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    header += b" " * ((-len(header)) % tool.HEADER_ALIGNMENT_BYTES)
    path = tmp_path / "header.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"x")

    length, decoded = tool._read_header(path)

    assert length == len(header)
    assert decoded["tensor"]["data_offsets"] == [0, 1]


def test_download_directory_rejects_concurrent_writer(tmp_path: Path) -> None:
    tool = _tool()

    with tool._exclusive_download(tmp_path):
        with pytest.raises(ValueError, match="concurrent_subset_download"):
            with tool._exclusive_download(tmp_path):
                raise AssertionError("unreachable")
