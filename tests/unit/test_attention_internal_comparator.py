from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "parity" / "compare_mlx_attention_internal_trace.py"


def _tool():
    spec = importlib.util.spec_from_file_location("compare_mlx_attention_internal_trace", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prefix_is_stable_and_zero_padded() -> None:
    tool = _tool()

    assert tool._prefix("stage_1", 0, 2, "attn1") == "stage_1_deep_block_00_pass_02_internal_attn1"


def test_gated_candidate_replays_exact_cuda_logits() -> None:
    tool = _tool()

    class Module:
        heads = 2
        dim_head = 2

    prefix = "stage_1_deep_block_00_pass_00_internal_attn1"
    reference = {
        f"{prefix}_gated_output": np.ones((1, 1, 4), dtype=np.float32),
        f"{prefix}_gate_logits": np.zeros((1, 1, 2), dtype=np.float32),
        f"{prefix}_gated_attention_input": np.ones((1, 1, 4), dtype=np.float32),
    }

    candidate = tool._gated_candidate(Module(), reference, prefix)

    assert candidate is not None
    mx.eval(candidate)
    np.testing.assert_array_equal(np.asarray(candidate.astype(mx.float32)), np.ones((1, 1, 4), dtype=np.float32))


def test_fp32_linear_variant_rounds_once_to_bfloat16() -> None:
    tool = _tool()

    class Layer:
        weight = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)
        bias = mx.array([0.5, -0.5], dtype=mx.bfloat16)

    value = mx.array([[[2.0, 3.0]]], dtype=mx.bfloat16)
    candidate = tool._linear_fp32(Layer(), value)

    mx.eval(candidate)
    np.testing.assert_array_equal(np.asarray(candidate.astype(mx.float32)), np.asarray([[[8.5, 17.5]]]))
