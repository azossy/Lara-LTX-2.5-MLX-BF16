#!/usr/bin/env python3
"""Compare one block's exact-input MLX attention sub-operations with CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from lara_ltx.errors import LaraError
from lara_ltx.parity import compare_tensors
from lara_ltx.transformer import CheckpointTransformerBlocks, apply_rotary_emb, scaled_dot_product_attention

CONFIG_SCHEMA_VERSION = 1
CHILD_OPERATIONS = {
    "query_projection": "to_q",
    "key_projection": "to_k",
    "value_projection": "to_v",
    "query_normalized": "q_norm",
    "key_normalized": "k_norm",
    "gate_logits": "to_gate_logits",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-checkpoint", required=True, type=Path)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--cuda-reference", required=True, action="append", type=Path)
    parser.add_argument("--cuda-report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"unreadable_json:{path}"}) from error
    if not isinstance(value, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_json_object:{path}"})
    return value


def _required(value: dict[str, Any], key: str, expected: type | tuple[type, ...]) -> Any:
    result = value.get(key)
    if not isinstance(result, expected) or (isinstance(result, bool) and expected in (int, float)):
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"invalid_attention_config:{key}"})
    return result


def _prefix(stage: str, block_index: int, pass_index: int, module_name: str) -> str:
    return f"{stage}_deep_block_{block_index:02d}_pass_{pass_index:02d}_internal_{module_name}"


def _tensor(reference: dict[str, np.ndarray], key: str, *, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    value = reference.get(key)
    if value is None:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_attention_tensor:{key}"})
    return mx.array(value, dtype=dtype)


def _optional_tensor(
    reference: dict[str, np.ndarray],
    metadata: dict[str, Any],
    key: str,
) -> mx.array | None:
    value = reference.get(key)
    if value is None:
        return None
    tensor_metadata = metadata.get(key, {})
    original_dtype = tensor_metadata.get("original_dtype") if isinstance(tensor_metadata, dict) else None
    dtype = {
        "torch.bool": mx.bool_,
        "torch.float32": mx.float32,
        "torch.float64": mx.float32,
    }.get(original_dtype, mx.bfloat16)
    return mx.array(value, dtype=dtype)


def _child_candidate(module: Any, operation: str, reference: dict[str, np.ndarray], prefix: str) -> mx.array | None:
    attribute = CHILD_OPERATIONS[operation]
    child = getattr(module, attribute, None)
    output_key = f"{prefix}_{operation}"
    input_key = f"{output_key}_input"
    if output_key not in reference or input_key not in reference:
        return None
    if child is None:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_attention_module:{attribute}"})
    return child(_tensor(reference, input_key))


def _rope_candidate(module: Any, boundary: str, reference: dict[str, np.ndarray], prefix: str) -> mx.array | None:
    output_key = f"{prefix}_{boundary}_ready"
    normalized_key = f"{prefix}_{boundary}_normalized"
    if output_key not in reference or normalized_key not in reference:
        return None
    cos_key = f"{prefix}_{boundary}_rope_cos"
    sin_key = f"{prefix}_{boundary}_rope_sin"
    normalized = _tensor(reference, normalized_key)
    if cos_key not in reference and sin_key not in reference:
        return normalized
    if cos_key not in reference or sin_key not in reference:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"incomplete_attention_rope:{prefix}_{boundary}"})
    return apply_rotary_emb(
        normalized,
        (_tensor(reference, cos_key), _tensor(reference, sin_key)),
        module.rope_type,
    )


def _sdpa_candidate(
    module: Any,
    reference: dict[str, np.ndarray],
    metadata: dict[str, Any],
    prefix: str,
) -> mx.array | None:
    output_key = f"{prefix}_sdpa_output"
    if output_key not in reference:
        return None
    query = _tensor(reference, f"{prefix}_sdpa_query")
    key = _tensor(reference, f"{prefix}_sdpa_key")
    value = _tensor(reference, f"{prefix}_sdpa_value")
    mask = _optional_tensor(reference, metadata, f"{prefix}_sdpa_mask")
    return scaled_dot_product_attention(query, key, value, module.heads, mask)


def _gated_candidate(module: Any, reference: dict[str, np.ndarray], prefix: str) -> mx.array | None:
    output_key = f"{prefix}_gated_output"
    logits_key = f"{prefix}_gate_logits"
    input_key = f"{prefix}_gated_attention_input"
    if output_key not in reference:
        return None
    if logits_key not in reference or input_key not in reference:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"incomplete_attention_gate:{prefix}"})
    attention = _tensor(reference, input_key)
    logits = _tensor(reference, logits_key)
    batch, tokens = attention.shape[:2]
    gates = 2.0 * mx.sigmoid(logits)
    gated = attention.reshape(batch, tokens, module.heads, module.dim_head) * mx.expand_dims(gates, -1)
    return gated.reshape(batch, tokens, module.heads * module.dim_head)


def _output_projection_candidate(module: Any, reference: dict[str, np.ndarray], prefix: str) -> mx.array | None:
    output_key = f"{prefix}_output_projection"
    input_key = f"{prefix}_output_projection_input"
    if output_key not in reference or input_key not in reference:
        return None
    return module.to_out[0](_tensor(reference, input_key))


def _linear_fp32(layer: Any, value: mx.array) -> mx.array:
    result = mx.matmul(value.astype(mx.float32), mx.swapaxes(layer.weight.astype(mx.float32), -1, -2))
    bias = getattr(layer, "bias", None)
    if bias is not None:
        result = result + bias.astype(mx.float32)
    return result.astype(mx.bfloat16)


def _fp32_variant(
    module: Any,
    operation: str,
    reference: dict[str, np.ndarray],
    prefix: str,
) -> mx.array | None:
    if operation in CHILD_OPERATIONS:
        attribute = CHILD_OPERATIONS[operation]
        child = getattr(module, attribute, None)
        input_key = f"{prefix}_{operation}_input"
        if child is None or input_key not in reference:
            return None
        value = _tensor(reference, input_key)
        if operation in {"query_normalized", "key_normalized"}:
            value_fp32 = value.astype(mx.float32)
            normalized = value_fp32 * mx.rsqrt(mx.mean(mx.square(value_fp32), axis=-1, keepdims=True) + child.eps)
            return (normalized * child.weight.astype(mx.float32)).astype(mx.bfloat16)
        return _linear_fp32(child, value)
    if operation in {"query_ready", "key_ready"}:
        boundary = "query" if operation == "query_ready" else "key"
        normalized_key = f"{prefix}_{boundary}_normalized"
        cos_key = f"{prefix}_{boundary}_rope_cos"
        sin_key = f"{prefix}_{boundary}_rope_sin"
        if normalized_key not in reference or cos_key not in reference or sin_key not in reference:
            return None
        return apply_rotary_emb(
            _tensor(reference, normalized_key, dtype=mx.float32),
            (_tensor(reference, cos_key, dtype=mx.float32), _tensor(reference, sin_key, dtype=mx.float32)),
            module.rope_type,
        ).astype(mx.bfloat16)
    if operation == "gated_output":
        logits_key = f"{prefix}_gate_logits"
        input_key = f"{prefix}_gated_attention_input"
        if logits_key not in reference or input_key not in reference:
            return None
        attention = _tensor(reference, input_key, dtype=mx.float32)
        logits = _tensor(reference, logits_key, dtype=mx.float32)
        batch, tokens = attention.shape[:2]
        gates = 2.0 * mx.sigmoid(logits)
        gated = attention.reshape(batch, tokens, module.heads, module.dim_head) * mx.expand_dims(gates, -1)
        return gated.reshape(batch, tokens, module.heads * module.dim_head).astype(mx.bfloat16)
    if operation == "output_projection":
        input_key = f"{prefix}_output_projection_input"
        if input_key not in reference:
            return None
        return _linear_fp32(module.to_out[0], _tensor(reference, input_key))
    return None


def _candidate_operations(
    module: Any,
    reference: dict[str, np.ndarray],
    metadata: dict[str, Any],
    prefix: str,
) -> tuple[tuple[str, Callable[[], mx.array | None]], ...]:
    return (
        *(
            (operation, lambda operation=operation: _child_candidate(module, operation, reference, prefix))
            for operation in CHILD_OPERATIONS
        ),
        ("query_ready", lambda: _rope_candidate(module, "query", reference, prefix)),
        ("key_ready", lambda: _rope_candidate(module, "key", reference, prefix)),
        ("sdpa_output", lambda: _sdpa_candidate(module, reference, metadata, prefix)),
        ("gated_output", lambda: _gated_candidate(module, reference, prefix)),
        ("output_projection", lambda: _output_projection_candidate(module, reference, prefix)),
    )


def main() -> int:
    arguments = parse_arguments()
    config = _read_json(arguments.config)
    cuda_report = _read_json(arguments.cuda_report)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_attention_config_schema"})
    expected_cuda_schema = int(_required(config, "cuda_artifact_schema_version", int))
    if cuda_report.get("schema_version") != expected_cuda_schema:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unsupported_attention_cuda_schema"})
    metadata = _required(cuda_report, "tensor_boundaries", dict)
    stage = str(_required(config, "stage", str))
    block_index = int(_required(config, "block_index", int))
    passes = _required(config, "passes", list)
    module_names = _required(config, "modules", list)
    lora_strength = float(_required(config, "lora_strength", (int, float)))
    diagnose_fp32_variants = bool(config.get("diagnose_fp32_variants", False))
    reference: dict[str, np.ndarray] = {}
    try:
        for path in arguments.cuda_reference:
            with np.load(path) as archive:
                overlap = reference.keys() & archive.files
                if overlap:
                    raise ValueError(f"duplicate_attention_tensor:{sorted(overlap)[0]}")
                reference.update({name: archive[name] for name in archive.files})
    except (OSError, ValueError) as error:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "unreadable_attention_reference"}) from error
    aliases = cuda_report.get("tensor_aliases", {})
    if not isinstance(aliases, dict):
        raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_attention_aliases"})
    for alias, canonical in aliases.items():
        if not isinstance(alias, str) or not isinstance(canonical, str) or canonical not in reference:
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_attention_alias"})
        reference[alias] = reference[canonical]

    provider = CheckpointTransformerBlocks(
        checkpoint=arguments.transformer_checkpoint,
        block_count=block_index + 1,
        lora_checkpoint=arguments.lora_checkpoint,
        lora_strength=lora_strength,
    )
    block = None
    for current_index, current_block in provider:
        if current_index == block_index:
            block = current_block
            break
    if block is None:
        raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_attention_block:{block_index}"})

    comparisons: list[dict[str, Any]] = []
    precision_diagnostics: list[dict[str, Any]] = []
    for pass_index in passes:
        if not isinstance(pass_index, int) or isinstance(pass_index, bool):
            raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_attention_pass_index"})
        for module_name in module_names:
            if not isinstance(module_name, str):
                raise LaraError("LARA-RUNTIME-008", details={"reason": "invalid_attention_module_name"})
            module = getattr(block, module_name, None)
            if module is None:
                raise LaraError("LARA-RUNTIME-008", details={"reason": f"missing_attention_module:{module_name}"})
            prefix = _prefix(stage, block_index, pass_index, module_name)
            for operation, candidate_factory in _candidate_operations(module, reference, metadata, prefix):
                candidate = candidate_factory()
                output_key = f"{prefix}_{operation}"
                if candidate is None:
                    continue
                mx.eval(candidate)
                metrics = compare_tensors(
                    output_key,
                    reference[output_key].astype(np.float32),
                    np.asarray(candidate.astype(mx.float32)),
                )
                comparisons.append(
                    {
                        "pass_index": pass_index,
                        "module": module_name,
                        "operation": operation,
                        "metrics": metrics.to_dict(),
                    }
                )
                if diagnose_fp32_variants:
                    diagnostic_candidate = _fp32_variant(module, operation, reference, prefix)
                    if diagnostic_candidate is not None:
                        mx.eval(diagnostic_candidate)
                        diagnostic_metrics = compare_tensors(
                            output_key,
                            reference[output_key].astype(np.float32),
                            np.asarray(diagnostic_candidate.astype(mx.float32)),
                        )
                        precision_diagnostics.append(
                            {
                                "pass_index": pass_index,
                                "module": module_name,
                                "operation": operation,
                                "baseline_normalized_rmse": metrics.normalized_rmse,
                                "fp32_variant_metrics": diagnostic_metrics.to_dict(),
                            }
                        )

    if not comparisons:
        raise LaraError("LARA-RUNTIME-008", details={"reason": "empty_attention_comparisons"})
    acceptance = _required(config, "acceptance", dict)
    maximum_nrmse = float(_required(acceptance, "maximum_normalized_rmse", (int, float)))
    minimum_cosine = float(_required(acceptance, "minimum_cosine_similarity", (int, float)))
    for comparison in comparisons:
        metrics = comparison["metrics"]
        comparison["passed"] = (
            metrics["nan_count"] == 0
            and metrics["inf_count"] == 0
            and metrics["normalized_rmse"] <= maximum_nrmse
            and metrics["cosine_similarity"] >= minimum_cosine
        )
    first_failure = next((comparison for comparison in comparisons if not comparison["passed"]), None)
    report = {
        "schema_version": 1,
        "component": "single_block_exact_input_attention_suboperations",
        "block_index": block_index,
        "lora_strength": lora_strength,
        "acceptance": acceptance,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "precision_diagnostics": precision_diagnostics,
        "first_failure": first_failure,
        "passed": first_failure is None,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(f"{arguments.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
