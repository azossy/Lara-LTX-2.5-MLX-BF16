from __future__ import annotations

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import validate_vocoder_mapping, vocoder_target_shapes

SOURCE_PREFIX = "vocoder."
BF16_DTYPE = "BF16"


def _mapping(target_shapes: dict[str, tuple[int, ...]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "rules": [
            {
                "source_file": "audio-vae.safetensors",
                "source_key": f"{SOURCE_PREFIX}{target}",
                "target_key": target,
                "transform": "identity",
                "dtype": BF16_DTYPE,
                "shape": list(shape),
            }
            for target, shape in target_shapes.items()
        ],
    }


def test_vocoder_mapping_has_full_vocoder_bwe_and_stft_contract() -> None:
    target_shapes = vocoder_target_shapes()

    assert len(target_shapes) == 1227
    assert len(validate_vocoder_mapping(_mapping(target_shapes))) == 1227


def test_vocoder_mapping_rejects_missing_or_incompatible_tensor() -> None:
    target_shapes = vocoder_target_shapes()
    missing = dict(target_shapes)
    missing.pop("bwe_generator.conv_post.weight")
    with pytest.raises(LaraError) as missing_error:
        validate_vocoder_mapping(_mapping(missing))

    assert missing_error.value.code == "LARA-MODEL-017"

    incompatible = dict(target_shapes)
    incompatible["mel_stft.mel_basis"] = (1, 1)
    with pytest.raises(LaraError) as incompatible_error:
        validate_vocoder_mapping(_mapping(incompatible))

    assert incompatible_error.value.code == "LARA-MODEL-032"
