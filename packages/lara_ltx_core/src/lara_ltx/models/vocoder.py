"""Reviewed mapping contract for the LTX 2.5 waveform vocoder and BWE."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

SOURCE_PREFIX = "vocoder."
MEL_BINS_PER_CHANNEL = 64
STEREO_CHANNEL_COUNT = 2
VOCODER_INPUT_CHANNELS = MEL_BINS_PER_CHANNEL * STEREO_CHANNEL_COUNT
WAVEFORM_OUTPUT_CHANNELS = STEREO_CHANNEL_COUNT
CONV_KERNEL_SIZE = 7
RESIDUAL_KERNEL_SIZES = (3, 7, 11)
RESIDUAL_DILATIONS = (1, 3, 5)
ACTIVATION_FILTER_SIZE = 12
STFT_FILTER_LENGTH = 512
STFT_FREQUENCY_BINS = STFT_FILTER_LENGTH // 2 + 1
MEL_STFT_CHANNELS = 2 * STFT_FREQUENCY_BINS
COMPONENT_NAME = "audio_vocoder_with_bwe"

PRIMARY_VOCODER_CHANNELS = 1536
PRIMARY_UPSAMPLE_RATES = (5, 2, 2, 2, 2, 2)
PRIMARY_UPSAMPLE_KERNELS = (11, 4, 4, 4, 4, 4)
BWE_CHANNELS = 512
BWE_UPSAMPLE_RATES = (6, 5, 2, 2, 2)
BWE_UPSAMPLE_KERNELS = (12, 11, 4, 4, 4)


def vocoder_target_shapes() -> dict[str, tuple[int, ...]]:
    """Return the full source-compatible MLX contract for waveform decoding."""

    shapes: dict[str, tuple[int, ...]] = {}
    _add_generator(
        shapes,
        "vocoder",
        initial_channels=PRIMARY_VOCODER_CHANNELS,
        upsample_rates=PRIMARY_UPSAMPLE_RATES,
        upsample_kernels=PRIMARY_UPSAMPLE_KERNELS,
    )
    _add_generator(
        shapes,
        "bwe_generator",
        initial_channels=BWE_CHANNELS,
        upsample_rates=BWE_UPSAMPLE_RATES,
        upsample_kernels=BWE_UPSAMPLE_KERNELS,
    )
    shapes.update(
        {
            "mel_stft.mel_basis": (MEL_BINS_PER_CHANNEL, STFT_FREQUENCY_BINS),
            "mel_stft.stft_fn.forward_basis": (MEL_STFT_CHANNELS, 1, STFT_FILTER_LENGTH),
            "mel_stft.stft_fn.inverse_basis": (MEL_STFT_CHANNELS, 1, STFT_FILTER_LENGTH),
        }
    )
    return shapes


def build_vocoder_mapping(shard_path: Path) -> dict[str, object]:
    """Map all waveform vocoder, BWE and exact checkpoint STFT tensors."""

    source = inspect_safetensors(shard_path)
    target_shapes = vocoder_target_shapes()
    descriptors = {
        descriptor.name.removeprefix(SOURCE_PREFIX): descriptor
        for descriptor in source.tensors
        if descriptor.name.startswith(SOURCE_PREFIX)
    }
    _require_exact_target_keys(set(descriptors), set(target_shapes))
    rules = [
        {
            "source_file": shard_path.name,
            "source_key": f"{SOURCE_PREFIX}{target_key}",
            "target_key": target_key,
            "transform": IDENTITY_TRANSFORM,
            "dtype": descriptors[target_key].dtype,
            "shape": list(descriptors[target_key].shape),
        }
        for target_key in sorted(target_shapes)
    ]
    mapping: dict[str, object] = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "component": COMPONENT_NAME,
        "shards": [{"file": shard_path.name, "tensor_count": len(rules)}],
        "rules": rules,
    }
    validate_vocoder_mapping(mapping)
    return mapping


def validate_vocoder_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Require exact BF16 waveform-vocoder keys, shapes and source prefix."""

    target_shapes = vocoder_target_shapes()
    rules = validate_mapping(mapping, expected_target_keys=target_shapes)
    for rule in rules:
        target_key = rule["target_key"]
        source_key = rule["source_key"]
        source_shape = rule.get("shape")
        if not isinstance(target_key, str) or not isinstance(source_key, str) or not isinstance(source_shape, list):
            raise LaraError("LARA-MODEL-014", details={"source_key": "vocoder_mapping_rule"})
        if (
            source_key != f"{SOURCE_PREFIX}{target_key}"
            or rule["transform"] != IDENTITY_TRANSFORM
            or rule.get("dtype") != BF16_DTYPE
            or tuple(source_shape) != target_shapes[target_key]
        ):
            raise LaraError("LARA-MODEL-032", details={"key": source_key})
    return rules


def write_vocoder_mapping(output: Path, shard_path: Path) -> None:
    """Atomically write the reviewed waveform-vocoder mapping."""

    mapping = build_vocoder_mapping(shard_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def _add_generator(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    *,
    initial_channels: int,
    upsample_rates: Sequence[int],
    upsample_kernels: Sequence[int],
) -> None:
    if len(upsample_rates) != len(upsample_kernels):
        raise LaraError("LARA-MODEL-032", details={"key": f"{prefix}.ups"})
    shapes[f"{prefix}.conv_pre.weight"] = (initial_channels, VOCODER_INPUT_CHANNELS, CONV_KERNEL_SIZE)
    shapes[f"{prefix}.conv_pre.bias"] = (initial_channels,)
    for stage, kernel_size in enumerate(upsample_kernels):
        input_channels = initial_channels // (2**stage)
        output_channels = initial_channels // (2 ** (stage + 1))
        shapes[f"{prefix}.ups.{stage}.weight"] = (input_channels, output_channels, kernel_size)
        shapes[f"{prefix}.ups.{stage}.bias"] = (output_channels,)
        for residual_kernel in RESIDUAL_KERNEL_SIZES:
            block_index = stage * len(RESIDUAL_KERNEL_SIZES) + RESIDUAL_KERNEL_SIZES.index(residual_kernel)
            _add_amp_residual_block(shapes, f"{prefix}.resblocks.{block_index}", output_channels, residual_kernel)
    final_channels = initial_channels // (2 ** len(upsample_rates))
    _add_activation(shapes, f"{prefix}.act_post", final_channels)
    shapes[f"{prefix}.conv_post.weight"] = (WAVEFORM_OUTPUT_CHANNELS, final_channels, CONV_KERNEL_SIZE)


def _add_amp_residual_block(shapes: dict[str, tuple[int, ...]], prefix: str, channels: int, kernel_size: int) -> None:
    for collection in ("convs1", "convs2"):
        for layer_index, dilation in enumerate(RESIDUAL_DILATIONS):
            effective_dilation = dilation if collection == "convs1" else 1
            shapes[f"{prefix}.{collection}.{layer_index}.weight"] = (channels, channels, kernel_size)
            shapes[f"{prefix}.{collection}.{layer_index}.bias"] = (channels,)
            _ = effective_dilation
    for collection in ("acts1", "acts2"):
        for layer_index in range(len(RESIDUAL_DILATIONS)):
            _add_activation(shapes, f"{prefix}.{collection}.{layer_index}", channels)


def _add_activation(shapes: dict[str, tuple[int, ...]], prefix: str, channels: int) -> None:
    shapes[f"{prefix}.act.alpha"] = (channels,)
    shapes[f"{prefix}.act.beta"] = (channels,)
    shapes[f"{prefix}.upsample.filter"] = (1, 1, ACTIVATION_FILTER_SIZE)
    shapes[f"{prefix}.downsample.lowpass.filter"] = (1, 1, ACTIVATION_FILTER_SIZE)


def _require_exact_target_keys(actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        key = missing[0] if missing else unexpected[0]
        raise LaraError("LARA-MODEL-032", details={"key": key})
