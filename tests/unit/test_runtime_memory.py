import pytest
from lara_ltx.errors import LaraError
from lara_ltx.runtime.memory import (
    AttentionShape,
    GenerationResourcePolicy,
    bf16_attention_io_bytes,
    bf16_transformer_block_probe_bytes,
    physical_memory_bytes,
    stage_video_tokens,
    validate_generation_resources,
)

GIBIBYTE = 1024**3


def _resource_policy(*, enforce: bool = True) -> GenerationResourcePolicy:
    return GenerationResourcePolicy(
        enforce=enforce,
        minimum_unified_memory_bytes=128 * GIBIBYTE,
        maximum_stage_two_video_tokens=640,
        video_time_scale=8,
        stage_two_spatial_scale=32,
        recommended_height=320,
        recommended_width=512,
        recommended_num_frames=17,
    )


def test_canonical_hq_token_counts() -> None:
    common = {"frames": 121, "time_scale": 8, "spatial_scale": 32}

    assert stage_video_tokens(height=544, width=960, **common) == 8_160
    assert stage_video_tokens(height=1_088, width=1_920, **common) == 32_640


def test_bf16_attention_io_estimate_has_no_score_matrix() -> None:
    shape = AttentionShape(batch_size=1, heads=32, tokens=32_640, head_dim=128)

    assert bf16_attention_io_bytes(shape) == 1_069_547_520


def test_non_aligned_video_grid_is_rejected() -> None:
    with pytest.raises(LaraError) as raised:
        stage_video_tokens(frames=120, height=1_088, width=1_920, time_scale=8, spatial_scale=32)

    assert raised.value.code == "LARA-CONFIG-003"


def test_transformer_block_probe_estimate_is_bounded_and_score_free() -> None:
    estimated = bf16_transformer_block_probe_bytes(
        tokens=32_640,
        hidden_size=4_096,
        feed_forward_multiplier=4,
    )

    assert estimated == 3_343_908_864


@pytest.mark.parametrize(
    ("height", "width", "num_frames", "expected_tokens"),
    ((320, 512, 17, 480), (512, 320, 25, 640)),
)
def test_generation_resource_preflight_accepts_measured_envelope(
    height: int,
    width: int,
    num_frames: int,
    expected_tokens: int,
) -> None:
    assessment = validate_generation_resources(
        height=height,
        width=width,
        num_frames=num_frames,
        policy=_resource_policy(),
        detected_unified_memory_bytes=128 * GIBIBYTE,
    )

    assert assessment.requested_stage_two_video_tokens == expected_tokens
    assert assessment.token_limit_satisfied is True
    assert assessment.memory_requirement_satisfied is True


def test_generation_resource_preflight_rejects_measured_oom_grid() -> None:
    with pytest.raises(LaraError) as raised:
        validate_generation_resources(
            height=512,
            width=512,
            num_frames=33,
            policy=_resource_policy(),
            detected_unified_memory_bytes=128 * GIBIBYTE,
        )

    assert raised.value.code == "LARA-RUNTIME-010"
    assert raised.value.details["requested_tokens"] == 1_280
    assert raised.value.details["maximum_tokens"] == 640


def test_generation_resource_preflight_rejects_unverified_memory_tier() -> None:
    with pytest.raises(LaraError) as raised:
        validate_generation_resources(
            height=320,
            width=512,
            num_frames=17,
            policy=_resource_policy(),
            detected_unified_memory_bytes=64 * GIBIBYTE,
        )

    assert raised.value.code == "LARA-RUNTIME-010"
    assert raised.value.details["reason"] == "insufficient_unified_memory"


def test_generation_resource_preflight_can_be_disabled_only_by_custom_policy() -> None:
    assessment = validate_generation_resources(
        height=512,
        width=512,
        num_frames=33,
        policy=_resource_policy(enforce=False),
        detected_unified_memory_bytes=64 * GIBIBYTE,
    )

    assert assessment.token_limit_satisfied is False
    assert assessment.memory_requirement_satisfied is False


def test_physical_memory_uses_sysconf_without_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"SC_PAGE_SIZE": 16_384, "SC_PHYS_PAGES": 8_388_608}
    monkeypatch.setattr("lara_ltx.runtime.memory.os.sysconf", values.__getitem__)

    assert physical_memory_bytes() == 128 * GIBIBYTE
