"""Strict, allocation-free safetensors header inspection."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from lara_ltx.errors import LaraError

SAFETENSORS_LENGTH_BYTES = 8
MAX_HEADER_BYTES = 256 * 1024 * 1024
BF16_DTYPE = "BF16"
DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    BF16_DTYPE: 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorDescriptor:
    """One tensor's validated location and storage contract."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    absolute_start: int
    absolute_end: int

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_count(self) -> int:
        return self.data_end - self.data_start


@dataclass(frozen=True)
class SafeTensorFile:
    """Validated safetensors header without materialized tensor payloads."""

    path: Path
    file_size: int
    header_size: int
    metadata: dict[str, str]
    tensors: tuple[TensorDescriptor, ...]

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def total_tensor_bytes(self) -> int:
        return sum(tensor.byte_count for tensor in self.tensors)

    def require_dtype(self, dtype: str) -> None:
        mismatches = [tensor.name for tensor in self.tensors if tensor.dtype != dtype]
        if mismatches:
            raise LaraError(
                "LARA-MODEL-007",
                details={"expected": dtype, "count": len(mismatches), "first": mismatches[0]},
            )


def _invalid_header(path: Path, reason: str, *, cause: Exception | None = None) -> LaraError:
    error = LaraError("LARA-MODEL-003", details={"path": str(path), "reason": reason})
    if cause is not None:
        error.__cause__ = cause
    return error


def _parse_metadata(raw: Any, path: Path) -> dict[str, str]:
    if raw is None:
        return {}
    invalid_entry = isinstance(raw, dict) and any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in raw.items()
    )
    if not isinstance(raw, dict) or invalid_entry:
        raise _invalid_header(path, "__metadata__ must be a string-to-string object")
    return dict(raw)


def _parse_descriptor(
    name: str,
    raw: Any,
    *,
    path: Path,
    data_base: int,
    file_size: int,
) -> TensorDescriptor:
    if not isinstance(raw, dict):
        raise _invalid_header(path, f"tensor {name!r} descriptor is not an object")
    dtype = raw.get("dtype")
    shape = raw.get("shape")
    offsets = raw.get("data_offsets")
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise LaraError("LARA-MODEL-004", details={"name": name, "dtype": dtype})
    if not isinstance(shape, list) or any(not isinstance(value, int) or value < 0 for value in shape):
        raise _invalid_header(path, f"tensor {name!r} has an invalid shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise _invalid_header(path, f"tensor {name!r} has invalid data offsets")

    start, end = offsets
    expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
    actual_bytes = end - start
    if actual_bytes != expected_bytes:
        raise LaraError(
            "LARA-MODEL-005",
            details={"name": name, "expected": expected_bytes, "actual": actual_bytes},
        )
    absolute_start = data_base + start
    absolute_end = data_base + end
    if absolute_end > file_size:
        raise LaraError(
            "LARA-MODEL-006",
            details={"name": name, "start": absolute_start, "end": absolute_end, "file_size": file_size},
        )
    return TensorDescriptor(
        name=name,
        dtype=dtype,
        shape=tuple(shape),
        data_start=start,
        data_end=end,
        absolute_start=absolute_start,
        absolute_end=absolute_end,
    )


def _reject_overlaps(tensors: list[TensorDescriptor]) -> None:
    nonempty = sorted((tensor for tensor in tensors if tensor.byte_count), key=lambda tensor: tensor.data_start)
    for previous, current in pairwise(nonempty):
        if current.data_start < previous.data_end:
            raise LaraError(
                "LARA-MODEL-008",
                details={"first": previous.name, "second": current.name},
            )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def inspect_safetensors(path: Path) -> SafeTensorFile:
    """Validate a safetensors file by reading only its JSON header."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            length_bytes = handle.read(SAFETENSORS_LENGTH_BYTES)
            if len(length_bytes) != SAFETENSORS_LENGTH_BYTES:
                raise _invalid_header(path, "missing 8-byte header length")
            (header_size,) = struct.unpack("<Q", length_bytes)
            if header_size == 0 or header_size > MAX_HEADER_BYTES:
                raise _invalid_header(path, f"header length {header_size} is outside the supported range")
            data_base = SAFETENSORS_LENGTH_BYTES + header_size
            if data_base > file_size:
                raise _invalid_header(path, "header extends beyond end of file")
            header_bytes = handle.read(header_size)
    except LaraError:
        raise
    except OSError as exc:
        raise _invalid_header(path, "file could not be read", cause=exc) from exc

    try:
        header = json.loads(header_bytes, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, ValueError) as exc:
        raise _invalid_header(path, "header is not valid UTF-8 JSON", cause=exc) from exc
    if not isinstance(header, dict):
        raise _invalid_header(path, "header root is not an object")

    metadata = _parse_metadata(header.get("__metadata__"), path)
    tensors = [
        _parse_descriptor(name, raw, path=path, data_base=data_base, file_size=file_size)
        for name, raw in header.items()
        if name != "__metadata__"
    ]
    _reject_overlaps(tensors)
    return SafeTensorFile(
        path=path,
        file_size=file_size,
        header_size=header_size,
        metadata=metadata,
        tensors=tuple(tensors),
    )
