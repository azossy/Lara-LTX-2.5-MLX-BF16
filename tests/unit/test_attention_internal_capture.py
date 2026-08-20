from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "parity" / "capture_cuda_hq_boundaries.py"


class FakeTensor:
    def __init__(self, value: float) -> None:
        self.value = np.asarray([value], dtype=np.float32)
        self.dtype = "torch.bfloat16"
        self.shape = self.value.shape

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeHandle:
    def __init__(self, callbacks: list[Any], callback: Any) -> None:
        self.callbacks = callbacks
        self.callback = callback

    def remove(self) -> None:
        self.callbacks.remove(self.callback)


class FakeChild:
    def __init__(self) -> None:
        self.forward_hooks: list[Any] = []
        self.pre_hooks: list[Any] = []

    def register_forward_hook(self, callback: Any, *, with_kwargs: bool) -> FakeHandle:
        assert with_kwargs is True
        self.forward_hooks.append(callback)
        return FakeHandle(self.forward_hooks, callback)

    def register_forward_pre_hook(self, callback: Any, *, with_kwargs: bool) -> FakeHandle:
        assert with_kwargs is True
        self.pre_hooks.append(callback)
        return FakeHandle(self.pre_hooks, callback)

    def emit(self, value: FakeTensor) -> FakeTensor:
        for callback in self.pre_hooks:
            callback(self, (value,), {})
        for callback in self.forward_hooks:
            callback(self, (value,), {}, value)
        return value


class FakeAttention:
    def __init__(self) -> None:
        for name in ("to_q", "to_k", "to_v", "q_norm", "k_norm", "to_gate_logits"):
            setattr(self, name, FakeChild())
        self.to_out = [FakeChild()]
        self.preattention_function = lambda query, key, *_args, **_kwargs: (query, key)
        self.attention_function = lambda *_args, **_kwargs: FakeTensor(7.0)
        self.masked_attention_function = lambda *_args, **_kwargs: FakeTensor(8.0)
        self.gated_attention_function = lambda *_args, **_kwargs: FakeTensor(9.0)


def _install_stub(monkeypatch: Any, name: str, **attributes: Any) -> None:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)


def _tool(monkeypatch: Any) -> Any:
    monkeypatch.syspath_prepend(str(TOOL_PATH.parent))
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.Generator = object
    torch.inference_mode = lambda: lambda function: function
    monkeypatch.setitem(sys.modules, "torch", torch)
    for package in (
        "ltx_core",
        "ltx_core.components",
        "ltx_core.model",
        "ltx_pipelines",
        "ltx_pipelines.utils",
    ):
        _install_stub(monkeypatch, package)
    _install_stub(monkeypatch, "ltx_core.components.guiders", MultiModalGuiderParams=object)
    _install_stub(
        monkeypatch,
        "ltx_core.model.video_vae",
        AUTO_TILING=object(),
        get_video_chunks_number=lambda *_args, **_kwargs: 1,
    )
    _install_stub(monkeypatch, "ltx_pipelines.ti2vid_two_stages_hq", TI2VidTwoStagesHQPipeline=object)
    _install_stub(
        monkeypatch,
        "ltx_pipelines.utils.args",
        add_generated_keyframes_arg=lambda parser: parser,
        hq_2_stage_arg_parser=lambda **_kwargs: object(),
    )
    _install_stub(monkeypatch, "ltx_pipelines.utils.constants", LTX_2_3_HQ_PARAMS=object())
    _install_stub(
        monkeypatch,
        "ltx_pipelines.utils.media_io",
        encode_video=lambda **_kwargs: None,
        resolve_hdr_color_space=lambda **_kwargs: None,
        vae_dtype_for_hdr=lambda *_args: None,
    )
    _install_stub(monkeypatch, "ltx_pipelines.utils.samplers", _get_new_noise=lambda latent, _generator: latent)

    spec = importlib.util.spec_from_file_location("capture_cuda_hq_boundaries_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attention_internals_recorder_tracks_passes_and_restores_callables(monkeypatch: Any) -> None:
    tool = _tool(monkeypatch)
    capture = tool.TensorCapture()
    attention = FakeAttention()
    originals = {
        name: getattr(attention, name)
        for name in (
            "preattention_function",
            "attention_function",
            "masked_attention_function",
            "gated_attention_function",
        )
    }
    recorder = tool.AttentionInternalsRecorder(capture, "stage_1", 0, "attn1", attention)
    recorder.install()

    for _ in range(2):
        query = attention.to_q.emit(FakeTensor(1.0))
        key = attention.to_k.emit(FakeTensor(2.0))
        value = attention.to_v.emit(FakeTensor(3.0))
        attention.q_norm.emit(query)
        attention.k_norm.emit(key)
        query, key = attention.preattention_function(query, key, attention, None, None, None)
        output = attention.attention_function(query, key, value, 2)
        output = attention.gated_attention_function(FakeTensor(4.0), output, attention)
        attention.to_gate_logits.emit(FakeTensor(5.0))
        attention.to_out[0].emit(output)

    assert "stage_1_deep_block_00_pass_00_internal_attn1_query_projection" in capture.arrays
    assert "stage_1_deep_block_00_pass_00_internal_attn1_query_projection_input" in capture.arrays
    assert "stage_1_deep_block_00_pass_00_internal_attn1_query_ready" in capture.arrays
    assert "stage_1_deep_block_00_pass_00_internal_attn1_sdpa_query" in capture.arrays
    assert "stage_1_deep_block_00_pass_00_internal_attn1_sdpa_output" in capture.arrays
    assert "stage_1_deep_block_00_pass_00_internal_attn1_gated_attention_input" in capture.arrays
    assert "stage_1_deep_block_00_pass_01_internal_attn1_output_projection" in capture.arrays

    recorder.remove()

    assert all(getattr(attention, name) is original for name, original in originals.items())
    assert all(not getattr(attention, name).forward_hooks for name in recorder._CHILD_BOUNDARIES)
    assert not attention.to_out[0].forward_hooks
    assert not attention.to_out[0].pre_hooks
