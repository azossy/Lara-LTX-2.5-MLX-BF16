from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from lara_ltx.errors import LaraError
from lara_ltx.models import DistilledLoraStrengths, build_lora_manifest, build_lora_pairs


def _write_header(path: Path, tensors: dict[str, tuple[int, ...]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    for name, shape in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        byte_count = count * 2
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + byte_count]}
        offset += byte_count
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def test_lora_manifest_pairs_every_adapter_with_its_base_weight(tmp_path: Path) -> None:
    lora = tmp_path / "lora.safetensors"
    transformer = tmp_path / "transformer.safetensors"
    _write_header(
        lora,
        {
            "diffusion_model.block.linear.lora_A.weight": (2, 4),
            "diffusion_model.block.linear.lora_B.weight": (6, 2),
        },
    )
    _write_header(transformer, {"model.diffusion_model.block.linear.weight": (6, 4)})

    pairs = build_lora_pairs(lora, transformer)
    manifest = build_lora_manifest(lora, transformer)

    assert pairs[0].target_key == "model.diffusion_model.block.linear.weight"
    assert pairs[0].rank == 2
    assert manifest["pairs"] == [
        {
            "a_key": "diffusion_model.block.linear.lora_A.weight",
            "b_key": "diffusion_model.block.linear.lora_B.weight",
            "base_shape": (6, 4),
            "rank": 2,
            "target_key": "model.diffusion_model.block.linear.weight",
        }
    ]


def test_distilled_lora_strengths_default_to_upstream_hq_stages() -> None:
    strengths = DistilledLoraStrengths()

    assert strengths.stage_1 == 0.25
    assert strengths.stage_2 == 0.5

    with pytest.raises(LaraError) as raised:
        DistilledLoraStrengths(stage_1=-0.1)

    assert raised.value.code == "LARA-MODEL-027"


@pytest.mark.parametrize(
    "lora_tensors",
    (
        {"diffusion_model.block.linear.lora_A.weight": (2, 4)},
        {
            "diffusion_model.block.linear.lora_A.weight": (2, 4),
            "diffusion_model.block.linear.lora_B.weight": (6, 3),
        },
    ),
)
def test_lora_manifest_rejects_unpaired_or_incompatible_adapter(
    tmp_path: Path,
    lora_tensors: dict[str, tuple[int, ...]],
) -> None:
    lora = tmp_path / "lora.safetensors"
    transformer = tmp_path / "transformer.safetensors"
    _write_header(lora, lora_tensors)
    _write_header(transformer, {"model.diffusion_model.block.linear.weight": (6, 4)})

    with pytest.raises(LaraError) as raised:
        build_lora_pairs(lora, transformer)

    assert raised.value.code == "LARA-MODEL-026"
