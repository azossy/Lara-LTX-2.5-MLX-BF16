"""Second-order res_2s sampler and variance-preserving SDE step for MLX."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from lara_ltx.errors import LaraError

SMALL_STEP_THRESHOLD = 1e-10
MIDPOINT_FRACTION = 0.5
TERMINAL_EPSILON_SIGMA = 0.0011
DEFAULT_ETA = 0.5
DEFAULT_BONG_MAX_ITERATIONS = 100
DEFAULT_BONG_STEP_THRESHOLD = 0.5
DEFAULT_BONG_SIGMA_THRESHOLD = 0.03
SAMPLER_HOST_DTYPE = np.float64
MLX_TRANSFER_DTYPE = np.float32


def phi(order: int, negative_step: float) -> float:
    if order <= 0:
        raise LaraError("LARA-SAMPLING-003", details={"reason": "invalid_phi_order"})
    if abs(negative_step) < SMALL_STEP_THRESHOLD:
        return 1.0 / math.factorial(order)
    remainder = sum(negative_step**power / math.factorial(power) for power in range(order))
    return (math.exp(negative_step) - remainder) / negative_step**order


def get_res2s_coefficients(step: float, *, midpoint: float = MIDPOINT_FRACTION) -> tuple[float, float, float]:
    if not math.isfinite(step) or step <= 0 or not 0.0 < midpoint < 1.0:
        raise LaraError("LARA-SAMPLING-003", details={"reason": "invalid_res2s_step"})
    a21 = midpoint * phi(1, -step * midpoint)
    b2 = phi(2, -step) / midpoint
    b1 = phi(1, -step) - b2
    return a21, b1, b2


@dataclass(frozen=True)
class Res2sLatentState:
    latent: mx.array
    denoise_mask: mx.array
    clean_latent: mx.array


def post_process_latent(denoised: mx.array, state: Res2sLatentState) -> mx.array:
    return (
        denoised.astype(mx.float32) * state.denoise_mask.astype(mx.float32)
        + state.clean_latent.astype(mx.float32) * (1.0 - state.denoise_mask.astype(mx.float32))
    ).astype(denoised.dtype)


class Res2sDiffusionStep:
    @staticmethod
    def get_sde_coefficients(sigma_next: float, sigma_up: float) -> tuple[float, float, float]:
        sigma_up = min(sigma_up, sigma_next * 0.9999)
        sigma_signal = 1.0 - sigma_next
        sigma_residual = math.sqrt(max(sigma_next**2 - sigma_up**2, 0.0))
        alpha_ratio = sigma_signal + sigma_residual
        sigma_down = sigma_residual / alpha_ratio
        return alpha_ratio, sigma_down, sigma_up

    def step(
        self,
        sample: mx.array,
        denoised_sample: mx.array,
        *,
        sigma: float,
        sigma_next: float,
        noise: mx.array,
        eta: float = DEFAULT_ETA,
    ) -> mx.array:
        if not 0.0 <= eta <= 1.0 or sigma <= sigma_next or sigma <= 0.0:
            raise LaraError("LARA-SAMPLING-003", details={"reason": "invalid_sde_step"})
        alpha_ratio, sigma_down, sigma_up = self.get_sde_coefficients(sigma_next, sigma_next * eta)
        if sigma_up == 0.0 or sigma_next == 0.0:
            return denoised_sample
        sample_f = sample.astype(mx.float32)
        denoised_f = denoised_sample.astype(mx.float32)
        epsilon_next = (sample_f - denoised_f) / (sigma - sigma_next)
        denoised_next = sample_f - sigma * epsilon_next
        noised = alpha_ratio * (denoised_next + sigma_down * epsilon_next) + sigma_up * noise.astype(mx.float32)
        return noised.astype(denoised_sample.dtype)

    def step_host(
        self,
        sample: np.ndarray,
        denoised_sample: np.ndarray,
        *,
        sigma: float,
        sigma_next: float,
        noise: np.ndarray,
        eta: float = DEFAULT_ETA,
    ) -> np.ndarray:
        """Evaluate the CUDA-reference sampler math in host float64."""

        if not 0.0 <= eta <= 1.0 or sigma <= sigma_next or sigma <= 0.0:
            raise LaraError("LARA-SAMPLING-003", details={"reason": "invalid_sde_step"})
        alpha_ratio, sigma_down, sigma_up = self.get_sde_coefficients(sigma_next, sigma_next * eta)
        if sigma_up == 0.0 or sigma_next == 0.0:
            return denoised_sample
        epsilon_next = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * epsilon_next
        return alpha_ratio * (denoised_next + sigma_down * epsilon_next) + sigma_up * noise


Denoiser = Callable[
    [Res2sLatentState | None, Res2sLatentState | None, float],
    tuple[mx.array | None, mx.array | None],
]
NoiseFunction = Callable[[mx.array, str], mx.array]


def _set_denoiser_step(denoiser: Denoiser, step_index: int) -> None:
    callback = getattr(denoiser, "set_step_index", None)
    if callable(callback):
        callback(step_index)


class _NoiseGenerator:
    def __init__(self, step_seed: int, substep_seed: int) -> None:
        self.keys = {
            "step": mx.random.key(step_seed),
            "substep": mx.random.key(substep_seed),
        }

    def __call__(self, value: mx.array, stream: str) -> mx.array:
        next_key, draw_key = mx.random.split(self.keys[stream], num=2)
        self.keys[stream] = next_key
        noise = mx.random.normal(value.shape, key=draw_key, dtype=mx.float32)
        noise = (noise - mx.mean(noise)) / mx.std(noise, ddof=1)
        return _channelwise_normalize(noise)


def _channelwise_normalize(value: mx.array) -> mx.array:
    spatial_mean = mx.mean(value, axis=(-2, -1), keepdims=True)
    spatial_std = mx.std(value, axis=(-2, -1), keepdims=True, ddof=1)
    return (value - spatial_mean) / spatial_std


class Res2sSampler:
    def __init__(
        self,
        *,
        eta: float = DEFAULT_ETA,
        bongmath: bool = True,
        bongmath_max_iterations: int = DEFAULT_BONG_MAX_ITERATIONS,
        noise_seed: int = -1,
        noise_seed_substep: int | None = None,
        noise_fn: NoiseFunction | None = None,
    ) -> None:
        if not 0.0 <= eta <= 1.0 or bongmath_max_iterations < 0:
            raise LaraError("LARA-SAMPLING-003", details={"reason": "invalid_sampler_configuration"})
        self.eta = eta
        self.bongmath = bongmath
        self.bongmath_max_iterations = bongmath_max_iterations
        substep_seed = noise_seed + 10_000 if noise_seed_substep is None else noise_seed_substep
        self.noise_fn = noise_fn or _NoiseGenerator(noise_seed, substep_seed)
        self.stepper = Res2sDiffusionStep()

    @staticmethod
    def _to_host(value: mx.array) -> np.ndarray:
        mx.eval(value)
        return np.asarray(value.astype(mx.float32), dtype=MLX_TRANSFER_DTYPE).astype(
            SAMPLER_HOST_DTYPE,
            copy=False,
        )

    @staticmethod
    def _to_model(value: np.ndarray, dtype: mx.Dtype) -> mx.array:
        return mx.array(np.asarray(value, dtype=MLX_TRANSFER_DTYPE), dtype=dtype)

    def _post_process_host(self, value: np.ndarray, state: Res2sLatentState) -> np.ndarray:
        mask = self._to_host(state.denoise_mask)
        clean = self._to_host(state.clean_latent)
        return value * mask + clean * (1.0 - mask)

    def _inject_host(
        self,
        state: Res2sLatentState,
        sample: np.ndarray,
        denoised: np.ndarray,
        *,
        sigma: float,
        sigma_next: float,
        stream: str,
        eta: float,
    ) -> np.ndarray:
        noise = self._to_host(self.noise_fn(state.latent, stream))
        updated = self.stepper.step_host(
            sample,
            denoised,
            sigma=sigma,
            sigma_next=sigma_next,
            noise=noise,
            eta=eta,
        )
        return self._post_process_host(updated, state)

    @staticmethod
    def _prepare_prediction(prediction: mx.array | None, state: Res2sLatentState | None) -> mx.array | None:
        return post_process_latent(prediction, state) if prediction is not None and state is not None else None

    def sample(
        self,
        sigmas: mx.array,
        video: Res2sLatentState | None,
        audio: Res2sLatentState | None,
        denoiser: Denoiser,
    ) -> tuple[Res2sLatentState | None, Res2sLatentState | None]:
        if video is None and audio is None:
            raise LaraError("LARA-SAMPLING-003", details={"reason": "missing_modalities"})
        schedule = np.asarray(sigmas.astype(mx.float32))
        if schedule.ndim != 1 or len(schedule) < 2 or np.any(np.diff(schedule) > 0) or schedule[0] <= 0:
            raise LaraError("LARA-SAMPLING-003", details={"reason": "invalid_sigma_schedule"})
        full_step_count = len(schedule) - 1
        terminal = schedule[-1] == 0.0
        if terminal:
            schedule = np.concatenate((schedule[:-1], np.array([TERMINAL_EPSILON_SIGMA, 0.0], dtype=np.float32)))

        for step_index in range(full_step_count):
            _set_denoiser_step(denoiser, step_index)
            sigma = float(schedule[step_index])
            sigma_next = float(schedule[step_index + 1])
            step = -math.log(sigma_next / sigma)
            a21, b1, b2 = get_res2s_coefficients(step)
            sub_sigma = math.sqrt(sigma * sigma_next)
            anchors = {
                "video": self._to_host(video.latent) if video is not None else None,
                "audio": self._to_host(audio.latent) if audio is not None else None,
            }
            first_video, first_audio = denoiser(video, audio, sigma)
            first = {
                "video": self._prepare_prediction(first_video, video),
                "audio": self._prepare_prediction(first_audio, audio),
            }
            states = {"video": video, "audio": audio}
            midpoint_values: dict[str, np.ndarray | None] = {}
            epsilon_first: dict[str, np.ndarray | None] = {}
            for name in ("video", "audio"):
                anchor = anchors[name]
                prediction = first[name]
                state = states[name]
                if anchor is None or prediction is None or state is None:
                    midpoint_values[name] = None
                    epsilon_first[name] = None
                    continue
                prediction_host = self._to_host(prediction)
                epsilon = prediction_host - anchor
                midpoint = anchor + step * a21 * epsilon
                midpoint = self._inject_host(
                    state,
                    anchor,
                    midpoint,
                    sigma=sigma,
                    sigma_next=sub_sigma,
                    stream="substep",
                    eta=DEFAULT_ETA,
                )
                if self.bongmath and step < DEFAULT_BONG_STEP_THRESHOLD and sigma > DEFAULT_BONG_SIGMA_THRESHOLD:
                    for _ in range(self.bongmath_max_iterations):
                        anchor = midpoint - step * a21 * epsilon
                        epsilon = prediction_host - anchor
                anchors[name] = anchor
                midpoint_values[name] = midpoint
                epsilon_first[name] = epsilon
            mid_video = (
                replace(video, latent=self._to_model(midpoint_values["video"], video.latent.dtype))
                if video is not None and midpoint_values["video"] is not None
                else None
            )
            mid_audio = (
                replace(audio, latent=self._to_model(midpoint_values["audio"], audio.latent.dtype))
                if audio is not None and midpoint_values["audio"] is not None
                else None
            )
            _set_denoiser_step(denoiser, 0)
            second_video, second_audio = denoiser(mid_video, mid_audio, sub_sigma)
            second = {
                "video": self._prepare_prediction(second_video, video),
                "audio": self._prepare_prediction(second_audio, audio),
            }
            updated: dict[str, Res2sLatentState | None] = {}
            for name in ("video", "audio"):
                state = states[name]
                anchor = anchors[name]
                epsilon_1 = epsilon_first[name]
                prediction_2 = second[name]
                if state is None or anchor is None or epsilon_1 is None or prediction_2 is None:
                    updated[name] = state
                    continue
                epsilon_2 = self._to_host(prediction_2) - anchor
                next_latent = anchor + step * (b1 * epsilon_1 + b2 * epsilon_2)
                next_latent = self._inject_host(
                    state,
                    anchor,
                    next_latent,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    stream="step",
                    eta=self.eta,
                )
                model_latent = self._to_model(next_latent, state.latent.dtype)
                mx.eval(model_latent)
                updated[name] = replace(state, latent=model_latent)
            video, audio = updated["video"], updated["audio"]

        if terminal:
            _set_denoiser_step(denoiser, full_step_count)
            final_video, final_audio = denoiser(video, audio, float(schedule[full_step_count]))
            if video is not None and final_video is not None:
                video = replace(video, latent=post_process_latent(final_video, video).astype(video.latent.dtype))
            if audio is not None and final_audio is not None:
                audio = replace(audio, latent=post_process_latent(final_audio, audio).astype(audio.latent.dtype))
        return video, audio
