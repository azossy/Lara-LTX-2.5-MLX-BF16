#!/usr/bin/env python3
"""Capture official CUDA HQ pipeline boundary tensors for MLX parity work."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.args import add_generated_keyframes_arg, hq_2_stage_arg_parser
from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS
from ltx_pipelines.utils.media_io import encode_video, resolve_hdr_color_space, vae_dtype_for_hdr
from ltx_pipelines.utils.samplers import _get_new_noise

ARTIFACT_SCHEMA_VERSION = 6
HASH_BLOCK_BYTES = 1024 * 1024
MAX_RECORDED_NOISER_CALLS_PER_STAGE = 2


class DiagnosticCaptureCompleteError(RuntimeError):
    """Stop the pipeline after the requested diagnostic boundary is durable."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TensorCapture:
    """Persist named tensors as float32 values with original dtype metadata."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, object]] = {}

    def add(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        array = tensor.detach().float().cpu().numpy()
        self.arrays[name] = array
        self.metadata[name] = {
            "original_dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stored_dtype": str(array.dtype),
            "sha256": _sha256_bytes(array.tobytes()),
        }

    def add_scalar(self, name: str, value: bool | int | float) -> None:
        array = np.asarray(value)
        self.arrays[name] = array
        self.metadata[name] = {
            "original_dtype": type(value).__name__,
            "shape": [],
            "stored_dtype": str(array.dtype),
            "sha256": _sha256_bytes(array.tobytes()),
        }


class CallRecorder:
    """Delegate one pipeline component while capturing its input/output boundary."""

    def __init__(self, component: Any, callback: Callable[[tuple[Any, ...], dict[str, Any], Any], Any | None]) -> None:
        self._component = component
        self._callback = callback

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._component(*args, **kwargs)
        replacement = self._callback(args, kwargs, output)
        return output if replacement is None else replacement

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component, name)


class AttentionInternalsRecorder:
    """Capture projection, normalization, RoPE, SDPA, gate and output boundaries."""

    _CHILD_BOUNDARIES: ClassVar[dict[str, str]] = {
        "to_q": "query_projection",
        "to_k": "key_projection",
        "to_v": "value_projection",
        "q_norm": "query_normalized",
        "k_norm": "key_normalized",
        "to_gate_logits": "gate_logits",
    }

    def __init__(
        self,
        capture: TensorCapture,
        stage_name: str,
        block_index: int,
        module_name: str,
        module: Any,
    ) -> None:
        self._capture = capture
        self._stage_name = stage_name
        self._block_index = block_index
        self._module_name = module_name
        self._module = module
        self._call_index = 0
        self._handles: list[Any] = []
        self._original_callables: dict[str, Any] = {}

    def _prefix(self) -> str:
        return (
            f"{self._stage_name}_deep_block_{self._block_index:02d}_pass_{self._call_index:02d}"
            f"_internal_{self._module_name}"
        )

    def _add(self, boundary: str, value: Any) -> None:
        self._capture.add(
            f"{self._prefix()}_{boundary}",
            value if isinstance(value, torch.Tensor) else None,
        )

    def _child_hook(self, boundary: str, *, finish_call: bool = False) -> Callable[..., None]:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            self._add(f"{boundary}_input", args[0] if args else kwargs.get("input"))
            self._add(boundary, output)
            if finish_call:
                self._call_index += 1

        return hook

    def _wrap_preattention(self, original: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            query, key = original(*args, **kwargs)
            self._add("query_ready", query)
            self._add("key_ready", key)
            positional = args[4] if len(args) > 4 else kwargs.get("pe")
            key_positional = args[5] if len(args) > 5 else kwargs.get("k_pe")
            if positional is not None:
                self._add("query_rope_cos", positional[0])
                self._add("query_rope_sin", positional[1])
                effective_key = positional if key_positional is None else key_positional
                self._add("key_rope_cos", effective_key[0])
                self._add("key_rope_sin", effective_key[1])
            return query, key

        return wrapped

    def _wrap_attention(self, original: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._add("sdpa_query", args[0] if args else kwargs.get("q"))
            self._add("sdpa_key", args[1] if len(args) > 1 else kwargs.get("k"))
            self._add("sdpa_value", args[2] if len(args) > 2 else kwargs.get("v"))
            mask = args[4] if len(args) > 4 else kwargs.get("mask")
            self._add("sdpa_mask", mask)
            output = original(*args, **kwargs)
            self._add("sdpa_output", output)
            return output

        return wrapped

    def _wrap_gated_attention(self, original: Any) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._add("gated_x", args[0] if args else kwargs.get("x"))
            self._add("gated_attention_input", args[1] if len(args) > 1 else kwargs.get("attn_out"))
            output = original(*args, **kwargs)
            self._add("gated_output", output)
            return output

        return wrapped

    def install(self) -> None:
        for child_name, boundary in self._CHILD_BOUNDARIES.items():
            child = getattr(self._module, child_name, None)
            if child is not None:
                self._handles.append(child.register_forward_hook(self._child_hook(boundary), with_kwargs=True))

        output_projection = self._module.to_out[0]
        self._handles.append(
            output_projection.register_forward_pre_hook(
                lambda _module, args, _kwargs: self._add("output_projection_input", args[0] if args else None),
                with_kwargs=True,
            )
        )
        self._handles.append(
            output_projection.register_forward_hook(
                self._child_hook("output_projection", finish_call=True),
                with_kwargs=True,
            )
        )

        wrappers = {
            "preattention_function": self._wrap_preattention,
            "attention_function": self._wrap_attention,
            "masked_attention_function": self._wrap_attention,
            "gated_attention_function": self._wrap_gated_attention,
        }
        for attribute, wrapper in wrappers.items():
            original = getattr(self._module, attribute)
            self._original_callables[attribute] = original
            setattr(self._module, attribute, wrapper(original))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for attribute, original in self._original_callables.items():
            setattr(self._module, attribute, original)
        self._original_callables.clear()


class NoiserRecorder:
    """Capture the initial noise transition for each video/audio diffusion stage."""

    def __init__(self, noiser: Any, capture: TensorCapture, stage_name: str) -> None:
        self._noiser = noiser
        self._capture = capture
        self._stage_name = stage_name
        self._call_count = 0

    def __call__(self, latent_state: Any, noise_scale: float = 1.0) -> Any:
        call_index = self._call_count
        self._call_count += 1
        if call_index < MAX_RECORDED_NOISER_CALLS_PER_STAGE:
            self._capture.add(f"{self._stage_name}_noiser_input_{call_index}", latent_state.latent)
        result = self._noiser(latent_state, noise_scale)
        if call_index < MAX_RECORDED_NOISER_CALLS_PER_STAGE:
            self._capture.add(f"{self._stage_name}_noiser_output_{call_index}", result.latent)
        return result


class DeepTransformerRecorder:
    """Capture selected block outputs from the first real transformer call."""

    def __init__(self, capture: TensorCapture, stage_name: str, block_indices: tuple[int, ...]) -> None:
        self._capture = capture
        self._stage_name = stage_name
        self._block_indices = block_indices
        self._handles: list[Any] = []
        self._attention_recorders: list[AttentionInternalsRecorder] = []
        self._block_call_counts = {index: 0 for index in block_indices}
        self._internal_call_counts: dict[str, int] = {}

    def _record_argument(self, prefix: str, argument: Any) -> None:
        if argument is None:
            return
        for field in (
            "x",
            "context",
            "timesteps",
            "embedded_timestep",
            "prompt_timestep",
            "cross_scale_shift_timestep",
            "cross_gate_timestep",
            "context_mask",
            "self_attention_mask",
            "self_attn_perturbation_mask",
            "cross_attn_perturbation_mask",
        ):
            self._capture.add(f"{prefix}_{field}", getattr(argument, field, None))
        for field in ("enabled", "self_attn_all_perturbed", "cross_attn_skip_all"):
            value = getattr(argument, field, None)
            if isinstance(value, (bool, int, float)):
                self._capture.add_scalar(f"{prefix}_{field}", value)
        for field in ("positional_embeddings", "cross_positional_embeddings"):
            frequencies = getattr(argument, field, None)
            if frequencies is not None:
                self._capture.add(f"{prefix}_{field}_cos", frequencies[0])
                self._capture.add(f"{prefix}_{field}_sin", frequencies[1])

    def install(self, transformer: Any) -> None:
        velocity_model = getattr(transformer, "velocity_model", None)
        blocks = getattr(velocity_model, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError("Transformer does not expose velocity_model.transformer_blocks")
        block_count = len(blocks)
        if any(index < 0 or index >= block_count for index in self._block_indices):
            raise RuntimeError(f"Deep capture block index is outside [0, {block_count})")
        first_index = min(self._block_indices)

        first_block = blocks[first_index]
        for module_name in (
            "attn1",
            "attn2",
            "ff",
            "audio_attn1",
            "audio_attn2",
            "audio_ff",
            "audio_to_video_attn",
            "video_to_audio_attn",
        ):
            module = getattr(first_block, module_name, None)
            if module is None:
                continue
            self._internal_call_counts[module_name] = 0

            def internal_hook(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
                *,
                current_name: str = module_name,
            ) -> None:
                pass_index = self._internal_call_counts[current_name]
                self._internal_call_counts[current_name] += 1
                prefix = (
                    f"{self._stage_name}_deep_block_{first_index:02d}_pass_{pass_index:02d}_internal_{current_name}"
                )
                primary_input = args[0] if args else kwargs.get("x")
                self._capture.add(f"{prefix}_input", primary_input)
                self._capture.add(f"{prefix}_output", output if isinstance(output, torch.Tensor) else None)

            self._handles.append(module.register_forward_hook(internal_hook, with_kwargs=True))
            if hasattr(module, "attention_function"):
                recorder = AttentionInternalsRecorder(
                    self._capture,
                    self._stage_name,
                    first_index,
                    module_name,
                    module,
                )
                recorder.install()
                self._attention_recorders.append(recorder)

        for block_index in self._block_indices:
            prefix = f"{self._stage_name}_deep_block_{block_index:02d}"

            def hook(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
                *,
                current_index: int = block_index,
                current_prefix: str = prefix,
            ) -> None:
                pass_index = self._block_call_counts[current_index]
                self._block_call_counts[current_index] += 1
                pass_prefix = f"{current_prefix}_pass_{pass_index:02d}"
                video_input = kwargs.get("video", args[0] if args else None)
                audio_input = kwargs.get("audio", args[1] if len(args) > 1 else None)
                if current_index == first_index:
                    self._record_argument(f"{pass_prefix}_video_input", video_input)
                    self._record_argument(f"{pass_prefix}_audio_input", audio_input)
                video_output, audio_output = output
                self._capture.add(
                    f"{pass_prefix}_video_output",
                    video_output.x if video_output is not None else None,
                )
                self._capture.add(
                    f"{pass_prefix}_audio_output",
                    audio_output.x if audio_output is not None else None,
                )

            self._handles.append(blocks[block_index].register_forward_hook(hook, with_kwargs=True))

    def remove(self) -> None:
        for recorder in self._attention_recorders:
            recorder.remove()
        self._attention_recorders.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class DenoiserRecorder:
    """Capture every Res2S denoiser input and output without changing execution."""

    def __init__(
        self,
        denoiser: Any,
        capture: TensorCapture,
        stage_name: str,
        *,
        deep_block_indices: tuple[int, ...] = (),
        stop_after_first_call: bool = False,
    ) -> None:
        self._denoiser = denoiser
        self._capture = capture
        self._stage_name = stage_name
        self._call_count = 0
        self._deep_block_indices = deep_block_indices
        self._stop_after_first_call = stop_after_first_call

    def __call__(
        self,
        transformer: Any,
        video_state: Any,
        audio_state: Any,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> Any:
        call_index = self._call_count
        self._call_count += 1
        prefix = f"{self._stage_name}_denoiser_call_{call_index:02d}"
        self._capture.add(f"{prefix}_sigma", sigmas[step_index])
        self._capture.add(f"{prefix}_video_input", video_state.latent if video_state is not None else None)
        self._capture.add(f"{prefix}_audio_input", audio_state.latent if audio_state is not None else None)
        deep_recorder = (
            DeepTransformerRecorder(self._capture, self._stage_name, self._deep_block_indices)
            if call_index == 0 and self._deep_block_indices
            else None
        )
        if deep_recorder is not None:
            deep_recorder.install(transformer)
        try:
            video_result, audio_result = self._denoiser(
                transformer,
                video_state,
                audio_state,
                sigmas,
                step_index,
            )
        finally:
            if deep_recorder is not None:
                deep_recorder.remove()
        self._capture.add(
            f"{prefix}_video_output",
            video_result.denoised if video_result is not None else None,
        )
        self._capture.add(
            f"{prefix}_audio_output",
            audio_result.denoised if audio_result is not None else None,
        )
        if call_index == 0:
            for modality, result in (("video", video_result), ("audio", audio_result)):
                if result is None:
                    continue
                for component in ("cond", "uncond", "ptb", "mod"):
                    value = getattr(result, component, None)
                    if isinstance(value, torch.Tensor):
                        self._capture.add(f"{prefix}_{modality}_{component}", value)
        if call_index == 0 and self._stop_after_first_call:
            raise DiagnosticCaptureCompleteError
        return video_result, audio_result


class SdeNoiseRecorder:
    """Capture the normalized random tensors consumed by both Res2S RNG streams."""

    _stream_order = ("substep", "step")
    _modality_order = ("video", "audio")

    def __init__(self, capture: TensorCapture, stage_name: str) -> None:
        self._capture = capture
        self._stage_name = stage_name
        self._generator_streams: dict[int, str] = {}
        self._stream_counts = {name: 0 for name in self._stream_order}

    def __call__(self, latent: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        generator_id = id(generator)
        if generator_id not in self._generator_streams:
            stream_index = len(self._generator_streams)
            if stream_index >= len(self._stream_order):
                raise RuntimeError("Unexpected additional Res2S noise generator")
            self._generator_streams[generator_id] = self._stream_order[stream_index]
        stream = self._generator_streams[generator_id]
        call_index = self._stream_counts[stream]
        self._stream_counts[stream] += 1
        modality = self._modality_order[call_index % len(self._modality_order)]
        step_index = call_index // len(self._modality_order)
        noise = _get_new_noise(latent, generator)
        self._capture.add(
            f"{self._stage_name}_sde_{stream}_{step_index:02d}_{modality}",
            noise,
        )
        return noise


class SdeTraceLoop:
    """Delegate the official loop with deterministic SDE and denoiser recorders."""

    def __init__(
        self,
        loop: Any,
        capture: TensorCapture,
        stage_name: str,
        *,
        deep_block_indices: tuple[int, ...] = (),
        stop_after_first_call: bool = False,
    ) -> None:
        self._loop = loop
        self._capture = capture
        self._stage_name = stage_name
        self._deep_block_indices = deep_block_indices
        self._stop_after_first_call = stop_after_first_call

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        recorded_kwargs = dict(kwargs)
        recorded_kwargs["denoiser"] = DenoiserRecorder(
            kwargs["denoiser"],
            self._capture,
            self._stage_name,
            deep_block_indices=self._deep_block_indices,
            stop_after_first_call=self._stop_after_first_call,
        )
        recorded_kwargs["new_noise_fn"] = SdeNoiseRecorder(self._capture, self._stage_name)
        return self._loop(*args, **recorded_kwargs)


class StageCallRecorder:
    """Delegate a diffusion stage while replacing its noiser with a recorder."""

    def __init__(
        self,
        component: Any,
        capture: TensorCapture,
        stage_name: str,
        *,
        deep_block_indices: tuple[int, ...] = (),
        stop_after_first_call: bool = False,
    ) -> None:
        self._component = component
        self._capture = capture
        self._stage_name = stage_name
        self._deep_block_indices = deep_block_indices
        self._stop_after_first_call = stop_after_first_call

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        recorded_kwargs = dict(kwargs)
        recorded_kwargs["noiser"] = NoiserRecorder(kwargs["noiser"], self._capture, self._stage_name)
        recorded_kwargs["loop"] = SdeTraceLoop(
            kwargs["loop"],
            self._capture,
            self._stage_name,
            deep_block_indices=self._deep_block_indices,
            stop_after_first_call=self._stop_after_first_call,
        )
        output = self._component(*args, **recorded_kwargs)
        _record_stage(self._capture, self._stage_name)(args, recorded_kwargs, output)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component, name)


def _record_context(capture: TensorCapture, args: tuple[Any, ...], _: dict[str, Any], output: Any) -> None:
    positive, negative = output
    capture.add("prompt_video_positive", positive.video_encoding)
    capture.add("prompt_audio_positive", positive.audio_encoding)
    capture.add("prompt_video_negative", negative.video_encoding)
    capture.add("prompt_audio_negative", negative.audio_encoding)


def _record_stage(capture: TensorCapture, stage_name: str) -> Callable[[tuple[Any, ...], dict[str, Any], Any], None]:
    def callback(_: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        capture.add(f"{stage_name}_sigmas", kwargs.get("sigmas"))
        video_state, audio_state = output
        capture.add(f"{stage_name}_video_latent", video_state.latent)
        capture.add(f"{stage_name}_audio_latent", audio_state.latent)

    return callback


def _record_upsampler(capture: TensorCapture, args: tuple[Any, ...], _: dict[str, Any], output: Any) -> None:
    capture.add("upsampler_input_video_latent", args[0])
    capture.add("upscaled_video_latent", output)


def _record_video_decoder(
    capture: TensorCapture, args: tuple[Any, ...], _: dict[str, Any], output: Iterator[torch.Tensor]
) -> None:
    capture.add("video_decoder_input_latent", args[0])
    original_output = output

    def recorded_frames() -> Iterator[torch.Tensor]:
        frame_count = 0
        for frame in original_output:
            if frame_count == 0:
                capture.add("decoded_video_chunk_0", frame)
            frame_count += 1
            yield frame
        capture.arrays["decoded_video_frame_count"] = np.asarray(frame_count, dtype=np.int64)
        capture.metadata["decoded_video_frame_count"] = {
            "original_dtype": "int64",
            "shape": [],
            "stored_dtype": "int64",
            "sha256": _sha256_bytes(np.asarray(frame_count, dtype=np.int64).tobytes()),
        }

    return recorded_frames()


def _record_audio_decoder(capture: TensorCapture, args: tuple[Any, ...], _: dict[str, Any], output: Any) -> None:
    capture.add("audio_decoder_input_latent", args[0])
    capture.add("decoded_audio", output.waveform)


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = add_generated_keyframes_arg(hq_2_stage_arg_parser(params=LTX_2_3_HQ_PARAMS, supports_auto_duration=True))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trace-report", type=Path, required=True)
    parser.add_argument("--deep-block-indices", nargs="*", type=int, default=[])
    parser.add_argument("--diagnostic-first-call-only", action="store_true")
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    capture = TensorCapture()
    pipeline = TI2VidTwoStagesHQPipeline(
        model_paths=arguments.model_paths,
        distilled_lora=arguments.distilled_lora,
        distilled_lora_strength_stage_1=arguments.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=arguments.distilled_lora_strength_stage_2,
        spatial_upsampler_path=arguments.spatial_upsampler_path,
        loras=tuple(arguments.lora) if arguments.lora else (),
        quantization=arguments.quantization,
        compilation_config=arguments.compile,
        offload_mode=arguments.offload_mode,
        prompt_enhancer_gemma_root=arguments.prompt_enhancer_gemma_root,
        diffvae_optimization=arguments.diffvae_optimization,
    )
    pipeline.prompt_encoder = CallRecorder(
        pipeline.prompt_encoder, lambda args, kwargs, output: _record_context(capture, args, kwargs, output)
    )
    deep_block_indices = tuple(sorted(set(arguments.deep_block_indices)))
    pipeline.stage_1 = StageCallRecorder(
        pipeline.stage_1,
        capture,
        "stage_1",
        deep_block_indices=deep_block_indices,
        stop_after_first_call=arguments.diagnostic_first_call_only,
    )
    pipeline.stage_2 = StageCallRecorder(pipeline.stage_2, capture, "stage_2")
    pipeline.upsampler = CallRecorder(
        pipeline.upsampler, lambda args, kwargs, output: _record_upsampler(capture, args, kwargs, output)
    )
    pipeline.video_decoder = CallRecorder(
        pipeline.video_decoder,
        lambda args, kwargs, output: _record_video_decoder(capture, args, kwargs, output),
    )
    pipeline.audio_decoder = CallRecorder(
        pipeline.audio_decoder,
        lambda args, kwargs, output: _record_audio_decoder(capture, args, kwargs, output),
    )

    hdr = resolve_hdr_color_space(images=arguments.images, hdr=arguments.hdr)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    diagnostic_complete = False
    try:
        video, audio, num_frames, tiling_config = pipeline(
            prompt=arguments.prompt,
            negative_prompt=arguments.negative_prompt,
            seed=arguments.seed,
            height=arguments.height,
            width=arguments.width,
            num_frames=arguments.num_frames,
            frame_rate=arguments.frame_rate,
            num_inference_steps=arguments.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=arguments.video_cfg_guidance_scale,
                stg_scale=arguments.video_stg_guidance_scale,
                rescale_scale=arguments.video_rescale_scale,
                modality_scale=arguments.a2v_guidance_scale,
                skip_step=arguments.video_skip_step,
                stg_blocks=arguments.video_stg_blocks,
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=arguments.audio_cfg_guidance_scale,
                stg_scale=arguments.audio_stg_guidance_scale,
                rescale_scale=arguments.audio_rescale_scale,
                modality_scale=arguments.v2a_guidance_scale,
                skip_step=arguments.audio_skip_step,
                stg_blocks=arguments.audio_stg_blocks,
            ),
            images=arguments.images,
            vae_dtype=vae_dtype,
            color_space=hdr,
            enhance_prompt=arguments.enhance_prompt,
            enhance_static_cache=arguments.enhance_static_cache,
            max_batch_size=arguments.max_batch_size,
            tiling_config=AUTO_TILING,
            generated_keyframes=arguments.num_generated_keyframes,
        )
    except DiagnosticCaptureCompleteError:
        diagnostic_complete = True
    if not diagnostic_complete:
        encode_video(
            video=video,
            fps=arguments.frame_rate,
            audio=audio,
            output_path=arguments.output_path,
            video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
            color_space=hdr,
        )
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.artifact, **capture.arrays)
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "artifact": str(arguments.artifact),
        "tensor_boundaries": capture.metadata,
        "output_path": str(arguments.output_path),
        "capture_mode": "diagnostic_first_call" if diagnostic_complete else "complete_pipeline",
        "deep_block_indices": list(deep_block_indices),
    }
    arguments.trace_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.trace_report.with_suffix(f"{arguments.trace_report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.trace_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
