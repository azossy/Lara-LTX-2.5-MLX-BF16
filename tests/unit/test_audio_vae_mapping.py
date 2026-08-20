from __future__ import annotations

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import audio_vae_target_shapes, validate_audio_vae_mapping

AUDIO_VAE_SOURCE_PREFIX = "audio_vae."
BF16_DTYPE = "BF16"


def _mapping(target_shapes: dict[str, tuple[int, ...]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "audio-vae.safetensors",
                "source_key": f"{AUDIO_VAE_SOURCE_PREFIX}{target}",
                "target_key": target,
                "transform": "identity",
                "dtype": BF16_DTYPE,
                "shape": list(shape),
            }
            for target, shape in target_shapes.items()
        ],
    }


def test_audio_vae_mapping_selects_only_audio_vae_core_tensors() -> None:
    target_shapes = audio_vae_target_shapes()

    assert len(target_shapes) == 102
    assert len(validate_audio_vae_mapping(_mapping(target_shapes))) == 102


def test_audio_vae_mapping_rejects_missing_or_incompatible_core_tensor() -> None:
    target_shapes = audio_vae_target_shapes()
    missing = dict(target_shapes)
    missing.pop("decoder.conv_in.conv.weight")
    with pytest.raises(LaraError) as missing_error:
        validate_audio_vae_mapping(_mapping(missing))

    assert missing_error.value.code == "LARA-MODEL-017"

    incompatible = dict(target_shapes)
    incompatible["encoder.conv_out.conv.weight"] = (1, 1, 1, 1)
    with pytest.raises(LaraError) as incompatible_error:
        validate_audio_vae_mapping(_mapping(incompatible))

    assert incompatible_error.value.code == "LARA-MODEL-031"
