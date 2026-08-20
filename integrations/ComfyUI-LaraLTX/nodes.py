# ruff: noqa: N802
"""ComfyUI nodes that delegate all inference to :class:`lara_ltx.LTXPipeline`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar

RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
DEFAULTS_PATH = RESOURCE_ROOT / "defaults.json"
MESSAGES_ROOT = RESOURCE_ROOT / "messages"
LOCALE_ENVIRONMENT_VARIABLE = "LARA_LOCALE"
DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES = frozenset({"en", "ko"})
ADAPTER_ERROR_CODE = "LARA-COMFY-001"
VIDEO_SUFFIX = ".mp4"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"[{ADAPTER_ERROR_CODE}] Unable to read adapter resource: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"[{ADAPTER_ERROR_CODE}] Invalid adapter resource: {path.name}")
    return payload


DEFAULTS = _read_json(DEFAULTS_PATH)


def _locale() -> str:
    requested = os.environ.get(LOCALE_ENVIRONMENT_VARIABLE, DEFAULT_LOCALE).split("_", maxsplit=1)[0].lower()
    return requested if requested in SUPPORTED_LOCALES else DEFAULT_LOCALE


def _message(key: str, **details: object) -> str:
    messages = _read_json(MESSAGES_ROOT / f"{_locale()}.json")
    template = str(messages.get(key, key))
    return template.format(**details)


def _adapter_error(reason: str) -> RuntimeError:
    return RuntimeError(f"[{ADAPTER_ERROR_CODE}] {_message('error', reason=_message(f'reason.{reason}'))}")


def _optional_path(value: str) -> Path | None:
    stripped = value.strip()
    return Path(stripped).expanduser() if stripped else None


class LaraLTXModelLoader:
    """Resolve a local or Hugging Face model into the shared public pipeline."""

    CATEGORY = DEFAULTS["node_category"]
    FUNCTION = "load"
    RETURN_TYPES = ("LARA_LTX_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    _cache: ClassVar[dict[tuple[str, str, str, bool], Any]] = {}

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "model": ("STRING", {"default": DEFAULTS["model"]}),
                "local_files_only": ("BOOLEAN", {"default": DEFAULTS["local_files_only"]}),
            },
            "optional": {
                "profile": ("STRING", {"default": DEFAULTS["profile"]}),
                "cache_dir": ("STRING", {"default": DEFAULTS["cache_dir"]}),
            },
        }

    @classmethod
    def load(
        cls,
        model: str,
        local_files_only: bool,
        profile: str = "",
        cache_dir: str = "",
    ) -> tuple[Any]:
        normalized_model = model.strip()
        if not normalized_model:
            raise _adapter_error("empty_model")
        key = (normalized_model, profile.strip(), cache_dir.strip(), bool(local_files_only))
        if key not in cls._cache:
            try:
                from lara_ltx import LTXPipeline

                cls._cache[key] = LTXPipeline.from_pretrained(
                    normalized_model,
                    profile_path=_optional_path(profile),
                    cache_dir=_optional_path(cache_dir),
                    local_files_only=bool(local_files_only),
                )
            except Exception as error:
                raise _adapter_error("model_load_failed") from error
        return (cls._cache[key],)


class LaraLTXTextToVideo:
    """Generate synchronized decoded video/audio through the public API."""

    CATEGORY = DEFAULTS["node_category"]
    FUNCTION = "generate"
    RETURN_TYPES = ("LARA_LTX_VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        generation = DEFAULTS["generation"]
        return {
            "required": {
                "pipeline": ("LARA_LTX_PIPELINE",),
                "prompt": ("STRING", {"default": generation["prompt"], "multiline": True}),
                "seed": ("INT", generation["seed"]),
                "width": ("INT", generation["width"]),
                "height": ("INT", generation["height"]),
                "num_frames": ("INT", generation["num_frames"]),
                "frame_rate": ("FLOAT", generation["frame_rate"]),
                "steps": ("INT", generation["steps"]),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {"default": generation["negative_prompt"], "multiline": True},
                ),
            },
        }

    @staticmethod
    def generate(
        pipeline: Any,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        frame_rate: float,
        steps: int,
        negative_prompt: str = "",
    ) -> tuple[Any]:
        try:
            video = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                seed=seed,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=steps,
            )
        except Exception as error:
            raise _adapter_error("generation_failed") from error
        return (video,)


class LaraLTXSaveVideo:
    """Save the public media result inside ComfyUI's configured output root."""

    CATEGORY = DEFAULTS["node_category"]
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "video": ("LARA_LTX_VIDEO",),
                "filename_prefix": ("STRING", {"default": DEFAULTS["filename_prefix"]}),
            }
        }

    @staticmethod
    def _destination(output_root: Path, filename_prefix: str) -> Path:
        prefix = Path(filename_prefix.strip())
        if not prefix.name or prefix.is_absolute() or ".." in prefix.parts:
            raise _adapter_error("invalid_output_name")
        candidate = (output_root / prefix).with_suffix(VIDEO_SUFFIX)
        resolved = candidate.resolve()
        if output_root != resolved.parent and output_root not in resolved.parents:
            raise _adapter_error("invalid_output_name")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if not resolved.exists():
            return resolved
        counter = 1
        while True:
            numbered = resolved.with_name(f"{resolved.stem}_{counter:05d}{resolved.suffix}")
            if not numbered.exists():
                return numbered
            counter += 1

    @classmethod
    def save(cls, video: Any, filename_prefix: str) -> dict[str, object]:
        try:
            import folder_paths

            output_root = Path(folder_paths.get_output_directory()).resolve()
            destination = cls._destination(output_root, filename_prefix)
            saved = Path(video.save(destination)).resolve()
        except RuntimeError:
            raise
        except Exception as error:
            raise _adapter_error("save_failed") from error
        relative = saved.relative_to(output_root)
        payload = {"filename": relative.name, "subfolder": str(relative.parent), "type": "output"}
        return {"ui": {"lara_video": [payload]}, "result": (str(saved),)}


NODE_CLASS_MAPPINGS = {
    "LaraLTXModelLoader": LaraLTXModelLoader,
    "LaraLTXTextToVideo": LaraLTXTextToVideo,
    "LaraLTXSaveVideo": LaraLTXSaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LaraLTXModelLoader": _message("node.loader"),
    "LaraLTXTextToVideo": _message("node.generate"),
    "LaraLTXSaveVideo": _message("node.save"),
}
