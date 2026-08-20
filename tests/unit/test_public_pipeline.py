from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.pipeline import TwoStageContexts, load_pipeline_profile
from lara_ltx.pipeline.api import _build_request

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

    assert profile.generation.height == 1088
    assert profile.generation.width == 1920
    assert profile.generation.num_inference_steps == 15
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
