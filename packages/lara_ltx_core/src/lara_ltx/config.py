"""Typed project configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LaraError


@dataclass(frozen=True)
class CanonicalHQConfig:
    """Shape and model dimensions for the release acceptance workload."""

    output_height: int
    output_width: int
    num_frames: int
    vae_time_scale: int
    vae_spatial_scale: int
    video_attention_heads: int
    video_attention_head_dim: int
    video_hidden_size: int
    feed_forward_multiplier: int


@dataclass(frozen=True)
class MemoryPolicyConfig:
    """Named memory safety limits for the primary Apple Silicon target."""

    minimum_system_reserve_bytes: int
    maximum_peak_fraction: float
    neighborhood_mask_element_budget: int


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration required to reproduce the canonical upstream state."""

    name: str
    dtype: str
    quantization: str
    source_repository_url: str
    source_commit: str
    model_repository_id: str
    model_revision: str
    network_timeout_seconds: int
    network_retry_count: int
    canonical_hq: CanonicalHQConfig
    memory: MemoryPolicyConfig


def _required(mapping: dict[str, Any], key: str, location: Path) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise LaraError("LARA-CONFIG-002", details={"key": key, "path": str(location)}) from exc


def _positive_int(mapping: dict[str, Any], key: str, location: Path) -> int:
    value = _required(mapping, key, location)
    if isinstance(value, bool):
        raise LaraError("LARA-CONFIG-003", details={"key": key, "value": value, "path": str(location)})
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LaraError("LARA-CONFIG-003", details={"key": key, "value": value, "path": str(location)}) from exc
    if parsed <= 0:
        raise LaraError("LARA-CONFIG-003", details={"key": key, "value": value, "path": str(location)})
    return parsed


def _fraction(mapping: dict[str, Any], key: str, location: Path) -> float:
    value = _required(mapping, key, location)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LaraError("LARA-CONFIG-003", details={"key": key, "value": value, "path": str(location)}) from exc
    if not 0.0 < parsed < 1.0:
        raise LaraError("LARA-CONFIG-003", details={"key": key, "value": value, "path": str(location)})
    return parsed


def load_project_config(path: Path) -> ProjectConfig:
    """Load the project TOML without embedding environment-specific paths in code."""

    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LaraError("LARA-CONFIG-001", details={"path": str(path)}) from exc

    project = _required(raw, "project", path)
    upstream = _required(raw, "upstream", path)
    source = _required(upstream, "source", path)
    model = _required(upstream, "model", path)
    runtime = _required(raw, "runtime", path)
    memory = _required(runtime, "memory", path)
    acceptance = _required(raw, "acceptance", path)
    canonical_hq = _required(acceptance, "canonical_hq", path)

    return ProjectConfig(
        name=str(_required(project, "name", path)),
        dtype=str(_required(project, "dtype", path)),
        quantization=str(_required(project, "quantization", path)),
        source_repository_url=str(_required(source, "repository_url", path)),
        source_commit=str(_required(source, "commit", path)),
        model_repository_id=str(_required(model, "repository_id", path)),
        model_revision=str(_required(model, "revision", path)),
        network_timeout_seconds=_positive_int(runtime, "network_timeout_seconds", path),
        network_retry_count=_positive_int(runtime, "network_retry_count", path),
        canonical_hq=CanonicalHQConfig(
            output_height=_positive_int(canonical_hq, "output_height", path),
            output_width=_positive_int(canonical_hq, "output_width", path),
            num_frames=_positive_int(canonical_hq, "num_frames", path),
            vae_time_scale=_positive_int(canonical_hq, "vae_time_scale", path),
            vae_spatial_scale=_positive_int(canonical_hq, "vae_spatial_scale", path),
            video_attention_heads=_positive_int(canonical_hq, "video_attention_heads", path),
            video_attention_head_dim=_positive_int(canonical_hq, "video_attention_head_dim", path),
            video_hidden_size=_positive_int(canonical_hq, "video_hidden_size", path),
            feed_forward_multiplier=_positive_int(canonical_hq, "feed_forward_multiplier", path),
        ),
        memory=MemoryPolicyConfig(
            minimum_system_reserve_bytes=_positive_int(memory, "minimum_system_reserve_bytes", path),
            maximum_peak_fraction=_fraction(memory, "maximum_peak_fraction", path),
            neighborhood_mask_element_budget=_positive_int(memory, "neighborhood_mask_element_budget", path),
        ),
    )
