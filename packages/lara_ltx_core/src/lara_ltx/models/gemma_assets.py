"""Read the tokenizer and sidecar assets embedded in the packed Gemma shard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lara_ltx.errors import LaraError

from .checkpoint import inspect_safetensors
from .gemma_text import GEMMA_CONFIG_METADATA_KEY

TOKENIZER_JSON_KEY = "tokenizer_json"
HF_ASSET_PREFIX = "hf_asset__"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
PROCESSOR_CONFIG_NAME = "processor_config.json"
REQUIRED_SIDECARS = (TOKENIZER_CONFIG_NAME, PROCESSOR_CONFIG_NAME)
TOKENIZER_MAX_LENGTH = 1_024
TOKENIZER_CONFIG_SKIP = frozenset(
    {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
)


@dataclass(frozen=True)
class PackedGemmaAssets:
    config: dict[str, Any]
    tokenizer_json: bytes
    sidecars: dict[str, bytes]

    def sidecar_json(self, name: str) -> dict[str, Any]:
        value = self.sidecars.get(name)
        if value is None:
            raise LaraError("LARA-MODEL-036", details={"asset": name})
        try:
            decoded = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LaraError("LARA-MODEL-036", details={"asset": name}) from exc
        if not isinstance(decoded, dict):
            raise LaraError("LARA-MODEL-036", details={"asset": name})
        return decoded


def _read_bytes(checkpoint: Path, *, start: int, count: int, asset: str) -> bytes:
    try:
        with checkpoint.open("rb") as handle:
            handle.seek(start)
            value = handle.read(count)
    except OSError as exc:
        raise LaraError("LARA-MODEL-036", details={"asset": asset}) from exc
    if len(value) != count:
        raise LaraError("LARA-MODEL-036", details={"asset": asset})
    return value


def load_packed_gemma_assets(checkpoint: Path) -> PackedGemmaAssets:
    source = inspect_safetensors(checkpoint)
    raw_config = source.metadata.get(GEMMA_CONFIG_METADATA_KEY)
    try:
        config = json.loads(raw_config) if raw_config is not None else None
    except json.JSONDecodeError as exc:
        raise LaraError("LARA-MODEL-036", details={"asset": GEMMA_CONFIG_METADATA_KEY}) from exc
    if not isinstance(config, dict):
        raise LaraError("LARA-MODEL-036", details={"asset": GEMMA_CONFIG_METADATA_KEY})

    byte_tensors = {descriptor.name: descriptor for descriptor in source.tensors if descriptor.dtype == "U8"}
    tokenizer_descriptor = byte_tensors.get(TOKENIZER_JSON_KEY)
    if tokenizer_descriptor is None or tokenizer_descriptor.shape != (tokenizer_descriptor.byte_count,):
        raise LaraError("LARA-MODEL-036", details={"asset": TOKENIZER_JSON_KEY})
    tokenizer_json = _read_bytes(
        checkpoint,
        start=tokenizer_descriptor.absolute_start,
        count=tokenizer_descriptor.byte_count,
        asset=TOKENIZER_JSON_KEY,
    )
    sidecars: dict[str, bytes] = {}
    for key, descriptor in byte_tensors.items():
        if not key.startswith(HF_ASSET_PREFIX):
            continue
        name = key.removeprefix(HF_ASSET_PREFIX)
        sidecars[name] = _read_bytes(
            checkpoint,
            start=descriptor.absolute_start,
            count=descriptor.byte_count,
            asset=name,
        )
    missing = [name for name in REQUIRED_SIDECARS if name not in sidecars]
    if missing:
        raise LaraError("LARA-MODEL-036", details={"asset": missing[0]})
    return PackedGemmaAssets(config=config, tokenizer_json=tokenizer_json, sidecars=sidecars)


def build_packed_gemma_tokenizer(assets: PackedGemmaAssets, *, max_length: int = TOKENIZER_MAX_LENGTH) -> Any:
    """Build the exact left-padding tokenizer without creating temporary files."""

    try:
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise LaraError("LARA-RUNTIME-007") from exc
    tokenizer_config = assets.sidecar_json(TOKENIZER_CONFIG_NAME)
    kwargs = {key: value for key, value in tokenizer_config.items() if key not in TOKENIZER_CONFIG_SKIP}
    chat_template = assets.sidecars.get("chat_template.jinja")
    if chat_template is not None:
        kwargs.setdefault("chat_template", chat_template.decode("utf-8"))
    try:
        backend = Tokenizer.from_buffer(assets.tokenizer_json)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            model_max_length=max_length,
            **kwargs,
        )
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise LaraError("LARA-MODEL-036", details={"asset": TOKENIZER_JSON_KEY}) from exc
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        raise LaraError("LARA-MODEL-036", details={"asset": "bos_token"})
    return tokenizer


def tokenize_prompts(
    tokenizer: Any,
    prompts: list[str],
    *,
    max_length: int = TOKENIZER_MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise LaraError("LARA-TENSOR-022", details={"reason": "invalid_prompts"})
    token_rows: list[list[int]] = []
    bos_token_id = tokenizer.bos_token_id
    for prompt in prompts:
        encoded = tokenizer(
            prompt.strip(),
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if not encoded or encoded[0] != bos_token_id:
            encoded = [bos_token_id, *encoded][:max_length]
        token_rows.append(encoded)
    padded = tokenizer.pad(
        {"input_ids": token_rows},
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
        return_attention_mask=True,
    )
    return np.asarray(padded["input_ids"], dtype=np.int32), np.asarray(padded["attention_mask"], dtype=np.int32)
