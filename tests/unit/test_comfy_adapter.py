from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = PROJECT_ROOT / "integrations" / "ComfyUI-LaraLTX"


def _load_nodes():
    spec = importlib.util.spec_from_file_location("lara_comfy_nodes", ADAPTER_ROOT / "nodes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_exposes_only_thin_public_pipeline_nodes() -> None:
    nodes = _load_nodes()

    assert set(nodes.NODE_CLASS_MAPPINGS) == {
        "LaraLTXModelLoader",
        "LaraLTXTextToVideo",
        "LaraLTXSaveVideo",
    }
    assert nodes.LaraLTXSaveVideo.OUTPUT_NODE is True
    assert "torch" not in (ADAPTER_ROOT / "nodes.py").read_text(encoding="utf-8")


def test_example_api_workflow_links_all_three_nodes() -> None:
    workflow = json.loads((ADAPTER_ROOT / "examples" / "text_to_video_api.json").read_text(encoding="utf-8"))
    defaults = json.loads((ADAPTER_ROOT / "resources" / "defaults.json").read_text(encoding="utf-8"))

    assert [workflow[str(index)]["class_type"] for index in range(1, 4)] == [
        "LaraLTXModelLoader",
        "LaraLTXTextToVideo",
        "LaraLTXSaveVideo",
    ]
    assert workflow["2"]["inputs"]["pipeline"] == ["1", 0]
    assert workflow["3"]["inputs"]["video"] == ["2", 0]
    expected_grid = {"width": 512, "height": 320, "num_frames": 17}
    assert {key: defaults["generation"][key]["default"] for key in expected_grid} == expected_grid
    assert {key: workflow["2"]["inputs"][key] for key in expected_grid} == expected_grid


def test_loader_caches_public_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = _load_nodes()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: object):
            calls.append((model, kwargs))
            return cls()

    monkeypatch.setitem(sys.modules, "lara_ltx", SimpleNamespace(LTXPipeline=FakePipeline))
    nodes.LaraLTXModelLoader._cache.clear()

    first = nodes.LaraLTXModelLoader.load("local-model", True)[0]
    second = nodes.LaraLTXModelLoader.load("local-model", True)[0]

    assert first is second
    assert len(calls) == 1
    assert calls[0][1]["local_files_only"] is True


def test_generation_forwards_every_workflow_value() -> None:
    nodes = _load_nodes()
    recorded: dict[str, object] = {}

    def pipeline(**kwargs: object):
        recorded.update(kwargs)
        return "video-result"

    result = nodes.LaraLTXTextToVideo.generate(
        pipeline,
        "prompt",
        42,
        512,
        320,
        17,
        24.0,
        15,
        "negative",
    )

    assert result == ("video-result",)
    assert recorded == {
        "prompt": "prompt",
        "negative_prompt": "negative",
        "seed": 42,
        "width": 512,
        "height": 320,
        "num_frames": 17,
        "frame_rate": 24.0,
        "num_inference_steps": 15,
    }


def test_save_node_confines_output_to_comfy_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = _load_nodes()
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    class FakeVideo:
        def save(self, path: Path) -> Path:
            path.touch()
            return path

    result = nodes.LaraLTXSaveVideo.save(FakeVideo(), "nested/result")

    saved = Path(result["result"][0])
    assert saved == tmp_path / "nested" / "result.mp4"
    assert result["ui"]["lara_video"][0]["type"] == "output"

    with pytest.raises(RuntimeError, match="LARA-COMFY-001"):
        nodes.LaraLTXSaveVideo.save(FakeVideo(), "../escape")
