"""Reviewed mapping contract for the LTX 2.5 Audio VAE core."""

from __future__ import annotations

import json
from pathlib import Path

from lara_ltx.errors import LaraError

from .checkpoint import BF16_DTYPE, inspect_safetensors
from .mapping import IDENTITY_TRANSFORM, MAPPING_SCHEMA_VERSION, validate_mapping

AUDIO_VAE_SOURCE_PREFIX = "audio_vae."
BASE_CHANNELS = 128
LATENT_CHANNELS = 8
AUDIO_CHANNELS = 2
CHANNEL_MULTIPLIERS = (1, 2, 4)
ENCODER_RESIDUAL_BLOCK_COUNT = 2
DECODER_RESIDUAL_BLOCK_COUNT = ENCODER_RESIDUAL_BLOCK_COUNT + 1
KERNEL_SIZE = 3
POINTWISE_KERNEL_SIZE = 1
COMPONENT_NAME = "audio_vae_core"


def audio_vae_target_shapes() -> dict[str, tuple[int, ...]]:
    """Return the source-compatible MLX parameter contract for Audio VAE only.

    The official pack also carries the separate waveform vocoder and BWE
    generator. Those 1,227 tensors intentionally belong to their own future
    runtime/mapping path; this contract is only the 102 Audio VAE tensors.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    _add_convolution(shapes, "encoder.conv_in.conv", AUDIO_CHANNELS, BASE_CHANNELS)
    encoder_channels = _add_encoder_path(shapes)
    _add_residual_block(shapes, "encoder.mid.block_1", encoder_channels, encoder_channels)
    _add_residual_block(shapes, "encoder.mid.block_2", encoder_channels, encoder_channels)
    _add_convolution(shapes, "encoder.conv_out.conv", encoder_channels, 2 * LATENT_CHANNELS)

    decoder_channels = BASE_CHANNELS * CHANNEL_MULTIPLIERS[-1]
    _add_convolution(shapes, "decoder.conv_in.conv", LATENT_CHANNELS, decoder_channels)
    _add_residual_block(shapes, "decoder.mid.block_1", decoder_channels, decoder_channels)
    _add_residual_block(shapes, "decoder.mid.block_2", decoder_channels, decoder_channels)
    decoder_channels = _add_decoder_path(shapes, decoder_channels)
    _add_convolution(shapes, "decoder.conv_out.conv", decoder_channels, AUDIO_CHANNELS)

    shapes["per_channel_statistics.mean-of-means"] = (BASE_CHANNELS,)
    shapes["per_channel_statistics.std-of-means"] = (BASE_CHANNELS,)
    return shapes


def build_audio_vae_mapping(shard_path: Path) -> dict[str, object]:
    """Map only the Audio VAE core, keeping vocoder/BWE tensors separate."""

    source = inspect_safetensors(shard_path)
    target_shapes = audio_vae_target_shapes()
    descriptors = {
        descriptor.name.removeprefix(AUDIO_VAE_SOURCE_PREFIX): descriptor
        for descriptor in source.tensors
        if descriptor.name.startswith(AUDIO_VAE_SOURCE_PREFIX)
    }
    _require_exact_target_keys(set(descriptors), set(target_shapes))
    rules = [
        {
            "source_file": shard_path.name,
            "source_key": f"{AUDIO_VAE_SOURCE_PREFIX}{target_key}",
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
    validate_audio_vae_mapping(mapping)
    return mapping


def validate_audio_vae_mapping(mapping: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Require exact BF16 Audio VAE core source keys, shapes and layout."""

    target_shapes = audio_vae_target_shapes()
    rules = validate_mapping(mapping, expected_target_keys=target_shapes)
    for rule in rules:
        target_key = rule["target_key"]
        source_key = rule["source_key"]
        source_shape = rule.get("shape")
        if not isinstance(target_key, str) or not isinstance(source_key, str) or not isinstance(source_shape, list):
            raise LaraError("LARA-MODEL-014", details={"source_key": "audio_vae_mapping_rule"})
        if (
            source_key != f"{AUDIO_VAE_SOURCE_PREFIX}{target_key}"
            or rule["transform"] != IDENTITY_TRANSFORM
            or rule.get("dtype") != BF16_DTYPE
            or tuple(source_shape) != target_shapes[target_key]
        ):
            raise LaraError("LARA-MODEL-031", details={"key": source_key})
    return rules


def write_audio_vae_mapping(output: Path, shard_path: Path) -> None:
    """Atomically write the reviewed Audio VAE core mapping."""

    mapping = build_audio_vae_mapping(shard_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except OSError as exc:
        raise LaraError("LARA-MODEL-009", details={"path": str(output)}) from exc


def _add_encoder_path(shapes: dict[str, tuple[int, ...]]) -> int:
    current_channels = BASE_CHANNELS
    for level, multiplier in enumerate(CHANNEL_MULTIPLIERS):
        output_channels = BASE_CHANNELS * multiplier
        for block_index in range(ENCODER_RESIDUAL_BLOCK_COUNT):
            prefix = f"encoder.down.{level}.block.{block_index}"
            _add_residual_block(shapes, prefix, current_channels, output_channels)
            current_channels = output_channels
        if level < len(CHANNEL_MULTIPLIERS) - 1:
            _add_convolution(shapes, f"encoder.down.{level}.downsample.conv", current_channels, current_channels)
    return current_channels


def _add_decoder_path(shapes: dict[str, tuple[int, ...]], current_channels: int) -> int:
    for level in reversed(range(len(CHANNEL_MULTIPLIERS))):
        output_channels = BASE_CHANNELS * CHANNEL_MULTIPLIERS[level]
        for block_index in range(DECODER_RESIDUAL_BLOCK_COUNT):
            prefix = f"decoder.up.{level}.block.{block_index}"
            _add_residual_block(shapes, prefix, current_channels, output_channels)
            current_channels = output_channels
        if level > 0:
            _add_convolution(shapes, f"decoder.up.{level}.upsample.conv.conv", current_channels, current_channels)
    return current_channels


def _add_residual_block(
    shapes: dict[str, tuple[int, ...]], prefix: str, input_channels: int, output_channels: int
) -> None:
    _add_convolution(shapes, f"{prefix}.conv1.conv", input_channels, output_channels)
    _add_convolution(shapes, f"{prefix}.conv2.conv", output_channels, output_channels)
    if input_channels != output_channels:
        _add_convolution(
            shapes,
            f"{prefix}.nin_shortcut.conv",
            input_channels,
            output_channels,
            kernel_size=POINTWISE_KERNEL_SIZE,
        )


def _add_convolution(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    input_channels: int,
    output_channels: int,
    *,
    kernel_size: int = KERNEL_SIZE,
) -> None:
    shapes[f"{prefix}.weight"] = (output_channels, input_channels, kernel_size, kernel_size)
    shapes[f"{prefix}.bias"] = (output_channels,)


def _require_exact_target_keys(actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        key = missing[0] if missing else unexpected[0]
        raise LaraError("LARA-MODEL-031", details={"key": key})
