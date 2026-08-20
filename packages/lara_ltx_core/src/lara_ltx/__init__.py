"""Lara LTX core package."""

from typing import Any

from .config import ProjectConfig, load_project_config
from .errors import LaraError

__all__ = ["LTXPipeline", "LTXVideo", "LaraError", "ProjectConfig", "load_project_config"]


def __getattr__(name: str) -> Any:
    """Load Metal-backed public classes only when inference is requested."""

    if name in {"LTXPipeline", "LTXVideo"}:
        from .pipeline import LTXPipeline, LTXVideo

        return {"LTXPipeline": LTXPipeline, "LTXVideo": LTXVideo}[name]
    raise AttributeError(name)
