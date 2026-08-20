"""Component-resident, shard-at-a-time BF16 safetensors loading for MLX."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lara_ltx.errors import LaraError

from .checkpoint import SafeTensorFile, inspect_safetensors
from .mapping import (
    FOLD_GATE_TRANSFORM,
    IDENTITY_TRANSFORM,
    SPLIT_QKV_K_TRANSFORM,
    SPLIT_QKV_Q_TRANSFORM,
    SPLIT_QKV_V_TRANSFORM,
    TRANSPOSE_2D_TRANSFORM,
)

EXPECTED_MLX_DTYPE_ATTRIBUTE_BY_SAFETENSORS_DTYPE = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}
QKV_TRANSFORM_INDEX = {
    SPLIT_QKV_Q_TRANSFORM: 0,
    SPLIT_QKV_K_TRANSFORM: 1,
    SPLIT_QKV_V_TRANSFORM: 2,
}
QKV_PART_COUNT = len(QKV_TRANSFORM_INDEX)
NUMPY_STORAGE_DTYPE_BY_SAFETENSORS_DTYPE = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
}


def _mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise LaraError("LARA-RUNTIME-003") from exc
    return mx


@dataclass(frozen=True)
class LoadedShard:
    """One verified MLX shard; callers release its tensor mapping before the next shard."""

    source: SafeTensorFile
    tensors: dict[str, Any]


def map_loaded_shard(
    shard: LoadedShard,
    rules: Iterable[dict[str, object]],
    *,
    expected_target_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Convert one verified shard into MLX ``load_weights`` entries.

    A caller validates the complete mapping once, then invokes this function
    for each source shard while its tensors are resident. The function never
    accumulates other shards, preserving component-level construction bounds.
    """

    mx = _mlx()
    mapped: list[tuple[str, Any]] = []
    for rule in rules:
        if rule.get("source_file") != shard.source.path.name:
            continue
        source_key = rule["source_key"]
        target_key = rule["target_key"]
        transform = rule["transform"]
        if not isinstance(source_key, str) or not isinstance(target_key, str):
            raise LaraError("LARA-MODEL-014", details={"source_key": "mapping_rule"})
        tensor = shard.tensors.get(source_key)
        if tensor is None:
            raise LaraError("LARA-MODEL-010", details={"name": source_key, "path": str(shard.source.path)})
        if transform == IDENTITY_TRANSFORM:
            mapped_tensor = tensor
        elif transform == TRANSPOSE_2D_TRANSFORM:
            if tensor.ndim != 2:
                raise LaraError("LARA-MODEL-019", details={"name": source_key, "shape": tuple(tensor.shape)})
            mapped_tensor = mx.transpose(tensor)
        elif transform in QKV_TRANSFORM_INDEX:
            if tensor.ndim < 1 or tensor.shape[0] % QKV_PART_COUNT:
                raise LaraError("LARA-MODEL-023", details={"name": source_key, "shape": tuple(tensor.shape)})
            part_size = tensor.shape[0] // QKV_PART_COUNT
            part_index = QKV_TRANSFORM_INDEX[transform]
            start = part_index * part_size
            mapped_tensor = tensor[start : start + part_size]
        elif transform == FOLD_GATE_TRANSFORM:
            gate_source_key = rule.get("gate_source_key")
            if not isinstance(gate_source_key, str):
                raise LaraError("LARA-MODEL-014", details={"source_key": source_key})
            gate = shard.tensors.get(gate_source_key)
            if gate is None:
                raise LaraError("LARA-MODEL-010", details={"name": gate_source_key, "path": str(shard.source.path)})
            if tensor.ndim == 2 and gate.ndim == 1 and tensor.shape[0] == gate.shape[0]:
                mapped_tensor = (tensor.astype(mx.float32) * gate.astype(mx.float32)[:, None]).astype(tensor.dtype)
            elif tensor.ndim == 1 and gate.ndim == 1 and tensor.shape == gate.shape:
                mapped_tensor = (tensor.astype(mx.float32) * gate.astype(mx.float32)).astype(tensor.dtype)
            else:
                raise LaraError(
                    "LARA-MODEL-024",
                    details={"name": source_key, "shape": tuple(tensor.shape), "gate_shape": tuple(gate.shape)},
                )
        else:
            raise LaraError("LARA-MODEL-016", details={"transform": transform})
        expected_shape = expected_target_shapes.get(target_key) if expected_target_shapes is not None else None
        if expected_shape is not None and tuple(mapped_tensor.shape) != expected_shape:
            raise LaraError(
                "LARA-MODEL-020",
                details={
                    "target_key": target_key,
                    "expected_shape": expected_shape,
                    "actual_shape": tuple(mapped_tensor.shape),
                },
            )
        mapped.append((target_key, mapped_tensor))
    return tuple(mapped)


def iter_component_weight_batches(
    shard_paths: Iterable[Path],
    rules: Iterable[dict[str, object]],
    *,
    expected_target_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> Iterable[tuple[tuple[str, Any], ...]]:
    """Yield one mapped MLX weight batch at a time for a resident component.

    Consumers must immediately pass each batch to ``load_weights`` before
    advancing the iterator. This deliberately avoids an aggregate mapping that
    would retain every split shard alongside the component under construction.
    """

    rules_by_file: dict[str, list[dict[str, object]]] = {}
    for rule in rules:
        source_file = rule.get("source_file")
        if not isinstance(source_file, str):
            raise LaraError("LARA-MODEL-014", details={"source_key": "mapping_rule"})
        rules_by_file.setdefault(source_file, []).append(rule)
    paths_by_name: dict[str, Path] = {}
    for path_value in shard_paths:
        path = Path(path_value)
        if path.name in paths_by_name:
            raise LaraError("LARA-MODEL-022", details={"file": path.name})
        paths_by_name[path.name] = path
    missing_files = sorted(set(rules_by_file) - set(paths_by_name))
    if missing_files:
        raise LaraError("LARA-MODEL-021", details={"file": missing_files[0]})
    for source_file in sorted(rules_by_file):
        file_rules = rules_by_file[source_file]
        required_names = tuple(
            dict.fromkeys(
                str(name)
                for rule in file_rules
                for name in (rule["source_key"], rule.get("gate_source_key"))
                if isinstance(name, str)
            )
        )
        allowed_dtypes = {str(rule["dtype"]) for rule in file_rules if isinstance(rule.get("dtype"), str)}
        allowed_dtypes.add("BF16")
        loaded = load_safetensors_shard(
            paths_by_name[source_file],
            required_names=required_names,
            allowed_dtypes=allowed_dtypes,
        )
        yield map_loaded_shard(loaded, file_rules, expected_target_shapes=expected_target_shapes)


def _require_names(source: SafeTensorFile, names: tuple[str, ...]) -> None:
    available = {descriptor.name for descriptor in source.tensors}
    for name in names:
        if name not in available:
            raise LaraError("LARA-MODEL-010", details={"name": name, "path": str(source.path)})


def _require_allowed_dtypes(source: SafeTensorFile, names: Iterable[str], allowed_dtypes: frozenset[str]) -> None:
    descriptors = {descriptor.name: descriptor for descriptor in source.tensors}
    mismatches = [name for name in names if descriptors[name].dtype not in allowed_dtypes]
    if mismatches:
        raise LaraError(
            "LARA-MODEL-007",
            details={
                "expected": ",".join(sorted(allowed_dtypes)),
                "count": len(mismatches),
                "first": mismatches[0],
            },
        )


def _validate_loaded_tensor(name: str, tensor: Any, source: SafeTensorFile, mx: Any) -> None:
    descriptor = next(item for item in source.tensors if item.name == name)
    expected_dtype_name = EXPECTED_MLX_DTYPE_ATTRIBUTE_BY_SAFETENSORS_DTYPE.get(descriptor.dtype)
    expected_dtype = getattr(mx, expected_dtype_name) if expected_dtype_name is not None else None
    if expected_dtype is None or tensor.dtype != expected_dtype or tuple(tensor.shape) != descriptor.shape:
        raise LaraError(
            "LARA-MODEL-011",
            details={
                "name": name,
                "expected_dtype": expected_dtype_name,
                "actual_dtype": str(tensor.dtype),
                "expected_shape": descriptor.shape,
                "actual_shape": tuple(tensor.shape),
            },
        )


def _load_selected_tensors(source: SafeTensorFile, names: tuple[str, ...], mx: Any) -> dict[str, Any]:
    """Materialize only requested entries from validated payload byte ranges.

    ``mx.load`` constructs a dictionary for every tensor in a safetensors file.
    That is safe for small test shards, but it needlessly expands the peak for
    components such as Gemma where the four LTX projection tensors share a very
    large checkpoint. The generic safetensors MLX adapter currently routes BF16
    through NumPy, which cannot represent it natively. Instead, map only each
    validated payload range as uint16 and reinterpret those bits as MLX BF16.
    """

    descriptors = {descriptor.name: descriptor for descriptor in source.tensors}
    selected: dict[str, Any] = {}
    try:
        for name in names:
            descriptor = descriptors[name]
            storage_dtype = NUMPY_STORAGE_DTYPE_BY_SAFETENSORS_DTYPE.get(descriptor.dtype)
            expected_dtype_name = EXPECTED_MLX_DTYPE_ATTRIBUTE_BY_SAFETENSORS_DTYPE.get(descriptor.dtype)
            if storage_dtype is None or expected_dtype_name is None:
                raise LaraError(
                    "LARA-MODEL-007",
                    details={"expected": "BF16,F16,F32", "count": 1, "first": name},
                )
            expected_dtype = getattr(mx, expected_dtype_name)
            if descriptor.element_count == 0:
                tensor = mx.zeros(descriptor.shape, dtype=expected_dtype)
            else:
                payload = np.memmap(
                    source.path,
                    mode="r",
                    dtype=storage_dtype,
                    offset=descriptor.absolute_start,
                    shape=descriptor.shape,
                )
                storage = mx.array(payload)
                tensor = storage.view(mx.bfloat16) if descriptor.dtype == "BF16" else storage.astype(expected_dtype)
            selected[name] = tensor
        mx.eval(*selected.values())
        return selected
    except OSError as exc:
        raise LaraError("LARA-MODEL-003", details={"path": str(source.path)}) from exc


def load_safetensors_shard(
    path: Path,
    *,
    required_names: Iterable[str] | None = None,
    required_dtype: str = "BF16",
    allowed_dtypes: Iterable[str] | None = None,
) -> LoadedShard:
    """Materialize one validated safetensors shard and return only requested tensors.

    This intentionally has no multi-shard accumulator. The component builder must
    map each returned shard into its resident component, then drop the mapping
    before requesting another shard. That bounds construction peak without using
    disk swapping during denoising.
    """

    source = inspect_safetensors(path)
    names = tuple(required_names) if required_names is not None else tuple(item.name for item in source.tensors)
    _require_names(source, names)
    permitted_dtypes = frozenset(allowed_dtypes) if allowed_dtypes is not None else frozenset({required_dtype})
    if not permitted_dtypes:
        raise LaraError(
            "LARA-MODEL-007",
            details={"expected": "at_least_one_dtype", "count": len(names), "first": names[0] if names else "<none>"},
        )
    _require_allowed_dtypes(source, names, permitted_dtypes)

    mx = _mlx()
    selected = _load_selected_tensors(source, names, mx)
    mx.eval(*selected.values())
    for name, tensor in selected.items():
        _validate_loaded_tensor(name, tensor, source, mx)
    return LoadedShard(source=source, tensors=selected)
