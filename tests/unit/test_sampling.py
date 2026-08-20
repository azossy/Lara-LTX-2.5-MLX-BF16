from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.sampling import (
    BatchedPerturbationConfig,
    LTX2Scheduler,
    MultiModalGuider,
    MultiModalGuiderParams,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
    Res2sLatentState,
    Res2sSampler,
    attach_block_perturbations,
    build_guidance_batch_plan,
    calculate_guided_output,
    get_res2s_coefficients,
    phi,
)
from lara_ltx.sampling.res2s import _NoiseGenerator
from lara_ltx.transformer import TransformerStream


def test_ltx2_scheduler_matches_cuda_hq_stage_1_schedule() -> None:
    with np.load(Path("golden/cuda_hq_boundary_trace_v2.npz")) as reference:
        expected = reference["stage_1_sigmas"]

    actual = LTX2Scheduler().execute(steps=15, token_count=120)

    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-7, rtol=0.0)


def test_ltx2_scheduler_rejects_one_step_stretched_schedule() -> None:
    with pytest.raises(LaraError, match="LARA-SAMPLING-001"):
        LTX2Scheduler().execute(steps=1)


def test_multimodal_guider_combines_cfg_stg_and_isolation() -> None:
    guider = MultiModalGuider(
        MultiModalGuiderParams(
            cfg_scale=3.0,
            stg_scale=0.5,
            modality_scale=2.0,
        )
    )

    result = guider.calculate(
        mx.full((1, 2, 3), 4.0),
        uncond=mx.full((1, 2, 3), 1.0),
        perturbed=mx.full((1, 2, 3), 2.0),
        isolated=mx.full((1, 2, 3), 3.0),
    )

    np.testing.assert_allclose(np.asarray(result), 12.0)


def test_guidance_batch_plan_builds_and_splits_all_required_passes() -> None:
    video = MultiModalGuider(MultiModalGuiderParams(cfg_scale=3.0, stg_scale=0.5, stg_blocks=(1,), modality_scale=2.0))
    audio = MultiModalGuider(MultiModalGuiderParams(cfg_scale=7.0, stg_scale=0.5, stg_blocks=(1,), modality_scale=2.0))
    plan = build_guidance_batch_plan(video, audio)
    outputs = mx.stack(
        (
            mx.full((2, 1, 1), 4.0),
            mx.full((2, 1, 1), 1.0),
            mx.full((2, 1, 1), 2.0),
            mx.full((2, 1, 1), 3.0),
        )
    ).reshape(8, 1, 1)

    guided = calculate_guided_output(video, plan, outputs, original_batch_size=2)

    assert plan.pass_names == ("cond", "uncond", "ptb", "mod")
    assert len(plan.repeated_perturbations(original_batch_size=2)) == 8
    np.testing.assert_allclose(np.asarray(guided), 12.0)


def _stream() -> TransformerStream:
    return TransformerStream(
        x=mx.ones((3, 2, 4)),
        context=mx.ones((3, 2, 4)),
        timesteps=mx.ones((3, 2, 24)),
    )


def test_batched_perturbations_attach_partial_stg_and_cross_masks() -> None:
    config = BatchedPerturbationConfig(
        (
            PerturbationConfig(),
            PerturbationConfig(
                (Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=(1,)),),
            ),
            PerturbationConfig(
                (
                    Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                    Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                )
            ),
        ),
        num_blocks=2,
        dtype=mx.float32,
    )

    video, audio = attach_block_perturbations(_stream(), _stream(), config, block=1)

    assert video is not None and audio is not None
    np.testing.assert_array_equal(np.asarray(video.self_attn_perturbation_mask).reshape(-1), [1, 0, 1])
    np.testing.assert_array_equal(np.asarray(video.cross_attn_perturbation_mask).reshape(-1), [1, 1, 0])
    assert audio.self_attn_perturbation_mask is None
    np.testing.assert_array_equal(np.asarray(audio.cross_attn_perturbation_mask).reshape(-1), [1, 1, 0])


def test_res2s_coefficients_have_stable_small_step_limit() -> None:
    assert phi(1, 0.0) == pytest.approx(1.0)
    assert phi(2, 0.0) == pytest.approx(0.5)
    a21, b1, b2 = get_res2s_coefficients(0.25)
    assert all(np.isfinite((a21, b1, b2)))


def test_res2s_noise_is_normalized_over_each_spatial_plane() -> None:
    noise = _NoiseGenerator(step_seed=11, substep_seed=10_011)(
        mx.zeros((2, 3, 4, 5), dtype=mx.float32),
        "step",
    )
    mx.eval(noise)

    values = np.asarray(noise)
    np.testing.assert_allclose(values.mean(axis=(-2, -1)), 0.0, atol=2e-7)
    np.testing.assert_allclose(values.std(axis=(-2, -1), ddof=1), 1.0, atol=2e-7)


def test_res2s_sampler_runs_two_stage_and_terminal_denoise() -> None:
    calls: list[float] = []
    state = Res2sLatentState(
        latent=mx.ones((1, 2, 3), dtype=mx.bfloat16),
        denoise_mask=mx.ones((1, 2, 1), dtype=mx.float32),
        clean_latent=mx.zeros((1, 2, 3), dtype=mx.bfloat16),
    )

    def denoiser(
        video: Res2sLatentState | None,
        audio: Res2sLatentState | None,
        sigma: float,
    ) -> tuple[mx.array | None, mx.array | None]:
        calls.append(sigma)
        return (video.latent * 0.25 if video is not None else None, audio.latent * 0.25 if audio is not None else None)

    sampler = Res2sSampler(
        bongmath=False,
        noise_fn=lambda value, stream: mx.zeros(value.shape, dtype=mx.float32),
    )
    video, audio = sampler.sample(mx.array([1.0, 0.5, 0.0]), state, None, denoiser)
    assert video is not None and audio is None
    mx.eval(video.latent)

    assert len(calls) == 5
    assert np.all(np.isfinite(np.asarray(video.latent.astype(mx.float32))))


def test_res2s_sampler_notifies_step_aware_denoiser() -> None:
    class StepAwareDenoiser:
        def __init__(self) -> None:
            self.step_indices: list[int] = []

        def set_step_index(self, step_index: int) -> None:
            self.step_indices.append(step_index)

        def __call__(
            self,
            video: Res2sLatentState | None,
            audio: Res2sLatentState | None,
            sigma: float,
        ) -> tuple[mx.array | None, mx.array | None]:
            del sigma
            return (
                video.latent if video is not None else None,
                audio.latent if audio is not None else None,
            )

    state = Res2sLatentState(
        latent=mx.ones((1, 2, 3), dtype=mx.bfloat16),
        denoise_mask=mx.ones((1, 2, 1), dtype=mx.float32),
        clean_latent=mx.zeros((1, 2, 3), dtype=mx.bfloat16),
    )
    denoiser = StepAwareDenoiser()

    Res2sSampler(
        bongmath=False,
        noise_fn=lambda value, stream: mx.zeros(value.shape, dtype=mx.float32),
    ).sample(mx.array([1.0, 0.5, 0.0]), state, None, denoiser)

    assert denoiser.step_indices == [0, 0, 1, 0, 2]
