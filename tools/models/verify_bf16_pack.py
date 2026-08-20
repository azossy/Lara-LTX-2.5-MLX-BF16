#!/usr/bin/env python3
"""Verify the downloaded pinned model pack without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from lara_ltx.models import inspect_safetensors

MODEL_ERROR_CODE = "LARA-MODEL-002"
EXPECTED_DTYPE = "BF16"
OFFICIAL_AUXILIARY_DTYPE = "F32"
OFFICIAL_SERIALIZED_ASSET_DTYPE = "U8"
OFFICIAL_F32_AUXILIARY_KEY = re.compile(
    r"model\.diffusion_model\.(?:"
    r"(?:audio_)?scale_shift_table|"
    r"transformer_blocks\.\d+\.(?:"
    r"audio_prompt_scale_shift_table|audio_scale_shift_table|prompt_scale_shift_table|"
    r"scale_shift_table|scale_shift_table_a2v_ca_audio|scale_shift_table_a2v_ca_video)"
    r")"
)
OFFICIAL_GEMMA_SERIALIZED_ASSET_KEYS = frozenset(
    {
        "hf_asset__chat_template.jinja",
        "hf_asset__generation_config.json",
        "hf_asset__processor_config.json",
        "hf_asset__tokenizer_config.json",
        "tokenizer_json",
    }
)
HASH_BLOCK_BYTES = 1024 * 1024
REPORT_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _dtype_counts(tensors: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(tensor.dtype) for tensor in tensors).items()))


def _unexpected_non_bf16_tensors(tensors: Iterable[Any]) -> list[str]:
    """Allow only upstream FP32 AdaLN modulation tables alongside BF16 payloads."""
    unexpected: list[str] = []
    for tensor in tensors:
        if tensor.dtype == EXPECTED_DTYPE:
            continue
        if tensor.dtype == OFFICIAL_AUXILIARY_DTYPE and OFFICIAL_F32_AUXILIARY_KEY.fullmatch(tensor.name):
            continue
        if tensor.dtype == OFFICIAL_SERIALIZED_ASSET_DTYPE and tensor.name in OFFICIAL_GEMMA_SERIALIZED_ASSET_KEYS:
            continue
        unexpected.append(f"{tensor.name}:{tensor.dtype}")
    return unexpected


def _fail(cause: str, action: str) -> int:
    print(json.dumps({"ok": False, "error_code": MODEL_ERROR_CODE, "cause": cause, "action": action}))
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    manifest: dict[str, Any] = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    components: list[dict[str, Any]] = []
    for entry in manifest["model"]["files"]:
        relative_path = str(entry["path"])
        checkpoint_path = arguments.model_dir / relative_path
        expected_size = int(entry["size"])
        expected_hash = entry.get("sha256")
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size != expected_size:
            return _fail(f"size_mismatch:{relative_path}", "download_affected_file_again")
        if not isinstance(expected_hash, str) or not expected_hash:
            return _fail(f"missing_manifest_hash:{relative_path}", "run_verified_downloader_again")
        actual_hash = _sha256(checkpoint_path)
        if actual_hash != expected_hash:
            return _fail(f"sha256_mismatch:{relative_path}", "download_affected_file_again")
        try:
            inventory = inspect_safetensors(checkpoint_path)
        except (OSError, ValueError) as error:
            return _fail(
                f"invalid_safetensors_header:{relative_path}:{type(error).__name__}", "download_affected_file_again"
            )
        dtype_counts = _dtype_counts(inventory.tensors)
        unexpected_tensors = _unexpected_non_bf16_tensors(inventory.tensors)
        if unexpected_tensors:
            return _fail(
                f"dtype_mismatch:{relative_path}:{unexpected_tensors[:3]}",
                "use_the_pinned_bf16_component",
            )
        components.append(
            {
                "path": relative_path,
                "sha256": actual_hash,
                "size_bytes": checkpoint_path.stat().st_size,
                "tensor_count": inventory.tensor_count,
                "dtype_counts": dtype_counts,
                "official_f32_auxiliary_tensor_count": dtype_counts.get(OFFICIAL_AUXILIARY_DTYPE, 0),
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "ok": True,
        "primary_dtype": EXPECTED_DTYPE,
        "official_auxiliary_dtype": OFFICIAL_AUXILIARY_DTYPE,
        "official_serialized_asset_dtype": OFFICIAL_SERIALIZED_ASSET_DTYPE,
        "components": components,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    print(json.dumps({"ok": True, "components": len(components)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
