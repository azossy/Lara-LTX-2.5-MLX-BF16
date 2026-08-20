from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.pipeline import TwoStageContexts, load_pipeline_profile
from lara_ltx.pipeline.api import LTXPipeline, _build_request
from lara_ltx.pipeline.configuration import load_distribution_source

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "LTX-2.5"


def _contexts() -> TwoStageContexts:
    return TwoStageContexts(
        video_positive=mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
        video_negative=mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
        audio_positive=mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
        audio_negative=mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
    )


def test_packaged_hq_profile_resolves_verified_local_pack() -> None:
    profile = load_pipeline_profile()

    paths = profile.model.resolve(MODEL_ROOT)

    assert profile.generation.height == 320
    assert profile.generation.width == 512
    assert profile.generation.num_frames == 17
    assert profile.generation.num_inference_steps == 15
    assert profile.resource_policy.enforce is True
    assert profile.resource_policy.maximum_stage_two_video_tokens == 640
    assert paths.transformer.name.endswith("transformer-bf16.safetensors")
    assert len(profile.model.allow_patterns) == 6


def test_public_request_derives_video_and_audio_layouts_from_user_dimensions() -> None:
    generation = load_pipeline_profile().generation.with_overrides(
        height=320,
        width=512,
        num_frames=17,
        frame_rate=24.0,
        num_inference_steps=2,
    )

    request = _build_request(generation, _contexts())

    assert request.stage_one_video_layout.latent_shape == (1, 128, 3, 5, 8)
    assert request.stage_two_video_layout.latent_shape == (1, 128, 3, 10, 16)
    assert request.audio_layout.latent_shape == (1, 8, 18, 16)
    assert request.stage_one_sigmas.shape == (3,)
    assert request.stage_two_sigmas.shape == (4,)


@pytest.mark.parametrize(
    ("field", "value"),
    (("height", 321), ("width", 513), ("num_frames", 18), ("num_inference_steps", 0)),
)
def test_public_generation_rejects_unsupported_grid(field: str, value: int) -> None:
    profile = load_pipeline_profile().generation

    with pytest.raises(LaraError, match="LARA-PIPELINE-003"):
        profile.with_overrides(**{field: value})


def test_public_pipeline_rejects_oom_grid_before_checkpoint_access() -> None:
    pipeline = LTXPipeline.__new__(LTXPipeline)
    pipeline.profile = load_pipeline_profile()

    with pytest.raises(LaraError) as raised:
        pipeline(prompt="A fox in a forest", height=512, width=512, num_frames=33)

    assert raised.value.code == "LARA-RUNTIME-010"
    assert raised.value.details["requested_tokens"] == 1_280


def test_from_pretrained_rejects_unverified_memory_before_model_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lara_ltx.pipeline.api.physical_memory_bytes", lambda: 64 * 1024**3)

    with pytest.raises(LaraError) as raised:
        LTXPipeline.from_pretrained("missing/repository", local_files_only=True)

    assert raised.value.code == "LARA-RUNTIME-010"
    assert raised.value.details["reason"] == "insufficient_unified_memory"


def test_release_descriptor_pins_official_gated_source() -> None:
    source = load_distribution_source(PROJECT_ROOT / "release" / "huggingface" / "lara_ltx_model.toml")

    profile = load_pipeline_profile()
    assert source.repository_id == profile.model.repository_id
    assert source.revision == profile.model.revision


def test_release_descriptor_rejects_unversioned_source(tmp_path: Path) -> None:
    manifest = tmp_path / "lara_ltx_model.toml"
    manifest.write_text(
        "[manifest]\nschema_version = 2\n[source]\nrepository_id = 'Lightricks/LTX-2.5'\nrevision = 'bad'\n",
        encoding="utf-8",
    )

    with pytest.raises(LaraError, match="LARA-PIPELINE-004"):
        load_distribution_source(manifest)


def test_release_descriptor_rejects_missing_source_with_release_error(tmp_path: Path) -> None:
    manifest = tmp_path / "lara_ltx_model.toml"
    manifest.write_text("[manifest]\nschema_version = 1\n", encoding="utf-8")

    with pytest.raises(LaraError, match="LARA-PIPELINE-004"):
        load_distribution_source(manifest)
