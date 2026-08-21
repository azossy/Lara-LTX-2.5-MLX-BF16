"""Reusable CUDA attention sub-operation capture helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TensorCapture:
    """Persist named tensors as float32 values with original dtype metadata."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, object]] = {}

    def add(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        array = tensor.detach().float().cpu().numpy()
        self.arrays[name] = array
        self.metadata[name] = {
            "original_dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stored_dtype": str(array.dtype),
            "sha256": sha256_bytes(array.tobytes()),
        }

    def add_scalar(self, name: str, value: bool | int | float) -> None:
        array = np.asarray(value)
        self.arrays[name] = array
        self.metadata[name] = {
            "original_dtype": type(value).__name__,
            "shape": [],
            "stored_dtype": str(array.dtype),
            "sha256": sha256_bytes(array.tobytes()),
        }


def deduplicate_capture(capture: TensorCapture) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return one canonical array per content hash plus alias-to-canonical names."""

    unique: dict[str, np.ndarray] = {}
    aliases: dict[str, str] = {}
    canonical_by_identity: dict[tuple[str, tuple[int, ...], str], str] = {}
    for name, array in capture.arrays.items():
        metadata = capture.metadata[name]
        identity = (str(metadata["sha256"]), tuple(array.shape), str(array.dtype))
        canonical = canonical_by_identity.get(identity)
        if canonical is None:
            canonical_by_identity[identity] = name
            unique[name] = array
        else:
            aliases[name] = canonical
    return unique, aliases


def write_artifact_shards(path: Path, arrays: dict[str, np.ndarray], *, maximum_bytes: int) -> list[Path]:
    """Write size-bounded compressed NPZ parts and remove stale parts for the base path."""

    if maximum_bytes <= 0:
        raise ValueError("invalid_maximum_artifact_shard_bytes")
    groups: list[dict[str, np.ndarray]] = []
    current: dict[str, np.ndarray] = {}
    current_bytes = 0
    for name, array in arrays.items():
        if array.nbytes > maximum_bytes:
            raise ValueError(f"attention_tensor_exceeds_shard_limit:{name}")
        if current and current_bytes + array.nbytes > maximum_bytes:
            groups.append(current)
            current = {}
            current_bytes = 0
        current[name] = array
        current_bytes += array.nbytes
    if current:
        groups.append(current)

    shards: list[Path] = []
    for index, group in enumerate(groups):
        shard = path.with_name(f"{path.stem}.part-{index:02d}{path.suffix}")
        temporary = shard.with_suffix(f"{shard.suffix}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **group)
        temporary.replace(shard)
        shards.append(shard)

    retained = set(shards)
    for stale_shard in path.parent.glob(f"{path.stem}.part-*{path.suffix}"):
        if stale_shard not in retained:
            stale_shard.unlink()
    return shards


class AttentionInternalsRecorder:
    """Capture projection, normalization, RoPE, SDPA, gate and output boundaries."""

    _CHILD_BOUNDARIES: ClassVar[dict[str, str]] = {
        "to_q": "query_projection",
        "to_k": "key_projection",
        "to_v": "value_projection",
        "q_norm": "query_normalized",
        "k_norm": "key_normalized",
        "to_gate_logits": "gate_logits",
    }

    def __init__(
        self,
        capture: TensorCapture,
        stage_name: str,
        block_index: int,
        module_name: str,
        module: Any,
    ) -> None:
        self._capture = capture
        self._stage_name = stage_name
        self._block_index = block_index
        self._module_name = module_name
        self._module = module
        self._call_index = 0
        self._handles: list[Any] = []
        self._original_callables: dict[str, Any] = {}

    def _prefix(self) -> str:
        return (
            f"{self._stage_name}_deep_block_{self._block_index:02d}_pass_{self._call_index:02d}"
            f"_internal_{self._module_name}"
        )

    def _add(self, boundary: str, value: Any) -> None:
        self._capture.add(
            f"{self._prefix()}_{boundary}",
            value if isinstance(value, torch.Tensor) else None,
        )

    def _child_hook(self, boundary: str, *, finish_call: bool = False) -> Callable[..., None]:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            self._add(f"{boundary}_input", args[0] if args else kwargs.get("input"))
            self._add(boundary, output)
            if finish_call:
                self._call_index += 1

        return hook

    def _wrap_preattention(self, original: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            query, key = original(*args, **kwargs)
            self._add("query_ready", query)
            self._add("key_ready", key)
            positional = args[4] if len(args) > 4 else kwargs.get("pe")
            key_positional = args[5] if len(args) > 5 else kwargs.get("k_pe")
            if positional is not None:
                self._add("query_rope_cos", positional[0])
                self._add("query_rope_sin", positional[1])
                effective_key = positional if key_positional is None else key_positional
                self._add("key_rope_cos", effective_key[0])
                self._add("key_rope_sin", effective_key[1])
            return query, key

        return wrapped

    def _wrap_attention(self, original: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._add("sdpa_query", args[0] if args else kwargs.get("q"))
            self._add("sdpa_key", args[1] if len(args) > 1 else kwargs.get("k"))
            self._add("sdpa_value", args[2] if len(args) > 2 else kwargs.get("v"))
            mask = args[4] if len(args) > 4 else kwargs.get("mask")
            self._add("sdpa_mask", mask)
            output = original(*args, **kwargs)
            self._add("sdpa_output", output)
            return output

        return wrapped

    def _wrap_gated_attention(self, original: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._add("gated_x", args[0] if args else kwargs.get("x"))
            self._add("gated_attention_input", args[1] if len(args) > 1 else kwargs.get("attn_out"))
            output = original(*args, **kwargs)
            self._add("gated_output", output)
            return output

        return wrapped

    def install(self) -> None:
        for child_name, boundary in self._CHILD_BOUNDARIES.items():
            child = getattr(self._module, child_name, None)
            if child is not None:
                self._handles.append(child.register_forward_hook(self._child_hook(boundary), with_kwargs=True))

        output_projection = self._module.to_out[0]
        self._handles.append(
            output_projection.register_forward_pre_hook(
                lambda _module, args, _kwargs: self._add("output_projection_input", args[0] if args else None),
                with_kwargs=True,
            )
        )
        self._handles.append(
            output_projection.register_forward_hook(
                self._child_hook("output_projection", finish_call=True),
                with_kwargs=True,
            )
        )

        wrappers = {
            "preattention_function": self._wrap_preattention,
            "attention_function": self._wrap_attention,
            "masked_attention_function": self._wrap_attention,
            "gated_attention_function": self._wrap_gated_attention,
        }
        for attribute, wrapper in wrappers.items():
            original = getattr(self._module, attribute)
            self._original_callables[attribute] = original
            setattr(self._module, attribute, wrapper(original))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for attribute, original in self._original_callables.items():
            setattr(self._module, attribute, original)
        self._original_callables.clear()
