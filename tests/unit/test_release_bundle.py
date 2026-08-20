from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "release" / "validate_release_bundle.py"


def _tool():
    spec = importlib.util.spec_from_file_location("validate_release_bundle", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_bundle_has_one_cross_service_identity() -> None:
    report = _tool().validate_release_bundle(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "configs" / "release.toml",
        pyproject_path=PROJECT_ROOT / "pyproject.toml",
        readme_path=PROJECT_ROOT / "README.md",
        hf_readme_path=PROJECT_ROOT / "release" / "huggingface" / "README.md",
        hf_manifest_path=PROJECT_ROOT / "release" / "huggingface" / "lara_ltx_model.toml",
    )

    assert report["passed"] is True
    assert report["version"] == "0.0.1"
    assert report["github_repository"] == "azossy/Lara-LTX-2.5-MLX-BF16"
    assert report["hugging_face_repository"] == "challychoi/Lara-LTX-2.5-MLX-BF16"
