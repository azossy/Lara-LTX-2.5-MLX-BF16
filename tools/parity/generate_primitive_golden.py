#!/usr/bin/env python3
"""Generate deterministic CUDA BF16 golden vectors for MLX primitive parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.model.video_vae.transformer.fallback_na.eager import na3d

DEFAULT_SEED = 25_081_900
DEFAULT_EPSILON = 1e-6
DEFAULT_DEVICE = "cuda:0"
ROPE_SHAPE = (2, 5, 4, 8)
GELU_SHAPE = (3, 7, 16)
NEIGHBORHOOD_ATTENTION_SHAPE = (1, 3, 4, 5, 2, 4)
NEIGHBORHOOD_ATTENTION_KERNEL = (3, 3, 3)
BLOCK_DIMENSION = 8
BLOCK_CONTEXT_DIMENSION = 6
BLOCK_CROSS_ADALN_CONTEXT_DIMENSION = BLOCK_DIMENSION
BLOCK_HEADS = 2
BLOCK_HEAD_DIMENSION = 4
BLOCK_VIDEO_TOKENS = 3
BLOCK_CONTEXT_TOKENS = 4
BLOCK_ADALN_PARAMETER_COUNT = 6
BLOCK_CROSS_ADALN_PARAMETER_COUNT = 9
AV_CROSS_SCALE_PARAMETER_COUNT = 4
AV_CROSS_GATE_PARAMETER_COUNT = 1
BLOCK_WEIGHT_PREFIX = "block_weight__"
AV_BLOCK_WEIGHT_PREFIX = "av_block_weight__"
CROSS_ADALN_AV_BLOCK_WEIGHT_PREFIX = "cross_adaln_av_block_weight__"
BLOCK_PARAMETER_STANDARD_DEVIATION = 0.02
AV_AUDIO_TOKENS = 2
AV_PERTURBATION_MASK_SHAPE = (1, 1, 1)
PROMPT_TIMESTEP_TOKEN_COUNT = 1


def _split_rope(x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor) -> torch.Tensor:
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat(
        (first_half * cos_freqs - second_half * sin_freqs, second_half * cos_freqs + first_half * sin_freqs),
        dim=-1,
    )


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _video_block_golden(device: torch.device, generator: torch.Generator) -> dict[str, np.ndarray]:
    """Return CUDA BF16 inputs, weights and output for the video-only block path."""

    block = BasicAVTransformerBlock(
        video=TransformerConfig(
            dim=BLOCK_DIMENSION,
            heads=BLOCK_HEADS,
            d_head=BLOCK_HEAD_DIMENSION,
            context_dim=BLOCK_CONTEXT_DIMENSION,
        )
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * BLOCK_PARAMETER_STANDARD_DEVIATION
            )
    video_input = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    context = torch.randn(
        (1, BLOCK_CONTEXT_TOKENS, BLOCK_CONTEXT_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    timesteps = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_ADALN_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    arguments = TransformerArgs(
        x=video_input,
        context=context,
        context_mask=None,
        timesteps=timesteps,
        embedded_timestep=torch.zeros((1, 1), device=device, dtype=torch.bfloat16),
        positional_embeddings=None,
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=None,
        cross_gate_timestep=None,
        enabled=True,
    )
    output, _ = block(arguments, None)
    assert output is not None
    result = {
        "block_input": _as_numpy(video_input),
        "block_context": _as_numpy(context),
        "block_timesteps": _as_numpy(timesteps),
        "block_output": _as_numpy(output.x),
        "block_heads": np.asarray(BLOCK_HEADS, dtype=np.int64),
        "block_head_dim": np.asarray(BLOCK_HEAD_DIMENSION, dtype=np.int64),
    }
    result.update({f"{BLOCK_WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()})
    return result


def _av_block_golden(device: torch.device, generator: torch.Generator) -> dict[str, np.ndarray]:
    """Return CUDA BF16 vectors for the simultaneous audio/video block path."""

    config = TransformerConfig(
        dim=BLOCK_DIMENSION,
        heads=BLOCK_HEADS,
        d_head=BLOCK_HEAD_DIMENSION,
        context_dim=BLOCK_CONTEXT_DIMENSION,
    )
    block = BasicAVTransformerBlock(video=config, audio=config).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * BLOCK_PARAMETER_STANDARD_DEVIATION
            )
    video_input = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_DIMENSION), generator=generator, device=device, dtype=torch.bfloat16
    )
    audio_input = torch.randn(
        (1, AV_AUDIO_TOKENS, BLOCK_DIMENSION), generator=generator, device=device, dtype=torch.bfloat16
    )
    context = torch.randn(
        (1, BLOCK_CONTEXT_TOKENS, BLOCK_CONTEXT_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    video_timesteps = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_ADALN_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_timesteps = torch.randn(
        (1, AV_AUDIO_TOKENS, BLOCK_ADALN_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    video_cross_timestep = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, AV_CROSS_SCALE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_cross_timestep = torch.randn(
        (1, AV_AUDIO_TOKENS, AV_CROSS_SCALE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    perturbation_mask = torch.ones(AV_PERTURBATION_MASK_SHAPE, device=device, dtype=torch.bfloat16)
    video_cross_gate_timestep = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, AV_CROSS_GATE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_cross_gate_timestep = torch.randn(
        (1, AV_AUDIO_TOKENS, AV_CROSS_GATE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    shared = {
        "context": context,
        "context_mask": None,
        "embedded_timestep": torch.zeros((1, 1), device=device, dtype=torch.bfloat16),
        "positional_embeddings": None,
        "cross_positional_embeddings": None,
        "enabled": True,
        "cross_attn_perturbation_mask": perturbation_mask,
    }
    video = TransformerArgs(
        x=video_input,
        timesteps=video_timesteps,
        cross_scale_shift_timestep=video_cross_timestep,
        cross_gate_timestep=video_cross_gate_timestep,
        **shared,
    )
    audio = TransformerArgs(
        x=audio_input,
        timesteps=audio_timesteps,
        cross_scale_shift_timestep=audio_cross_timestep,
        cross_gate_timestep=audio_cross_gate_timestep,
        **shared,
    )
    video_output, audio_output = block(video, audio)
    assert video_output is not None and audio_output is not None
    result = {
        "av_video_input": _as_numpy(video_input),
        "av_audio_input": _as_numpy(audio_input),
        "av_context": _as_numpy(context),
        "av_video_timesteps": _as_numpy(video_timesteps),
        "av_audio_timesteps": _as_numpy(audio_timesteps),
        "av_video_cross_timestep": _as_numpy(video_cross_timestep),
        "av_audio_cross_timestep": _as_numpy(audio_cross_timestep),
        "av_video_cross_gate_timestep": _as_numpy(video_cross_gate_timestep),
        "av_audio_cross_gate_timestep": _as_numpy(audio_cross_gate_timestep),
        "av_video_output": _as_numpy(video_output.x),
        "av_audio_output": _as_numpy(audio_output.x),
        "av_heads": np.asarray(BLOCK_HEADS, dtype=np.int64),
        "av_head_dim": np.asarray(BLOCK_HEAD_DIMENSION, dtype=np.int64),
    }
    result.update({f"{AV_BLOCK_WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()})
    return result


def _cross_adaln_block_golden(device: torch.device, generator: torch.Generator) -> dict[str, np.ndarray]:
    """Return CUDA BF16 vectors for text cross-attention AdaLN modulation."""

    config = TransformerConfig(
        dim=BLOCK_DIMENSION,
        heads=BLOCK_HEADS,
        d_head=BLOCK_HEAD_DIMENSION,
        context_dim=BLOCK_CROSS_ADALN_CONTEXT_DIMENSION,
        cross_attention_adaln=True,
    )
    block = BasicAVTransformerBlock(video=config).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * BLOCK_PARAMETER_STANDARD_DEVIATION
            )
    video_input = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_DIMENSION), generator=generator, device=device, dtype=torch.bfloat16
    )
    context = torch.randn(
        (1, BLOCK_CONTEXT_TOKENS, BLOCK_CROSS_ADALN_CONTEXT_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    timesteps = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_CROSS_ADALN_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    prompt_timestep = torch.randn(
        (1, PROMPT_TIMESTEP_TOKEN_COUNT, 2 * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    arguments = TransformerArgs(
        x=video_input,
        context=context,
        context_mask=None,
        timesteps=timesteps,
        embedded_timestep=torch.zeros((1, 1), device=device, dtype=torch.bfloat16),
        positional_embeddings=None,
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=None,
        cross_gate_timestep=None,
        enabled=True,
        prompt_timestep=prompt_timestep,
    )
    output, _ = block(arguments, None)
    assert output is not None
    result = {
        "cross_adaln_input": _as_numpy(video_input),
        "cross_adaln_context": _as_numpy(context),
        "cross_adaln_timesteps": _as_numpy(timesteps),
        "cross_adaln_prompt_timestep": _as_numpy(prompt_timestep),
        "cross_adaln_output": _as_numpy(output.x),
        "cross_adaln_heads": np.asarray(BLOCK_HEADS, dtype=np.int64),
        "cross_adaln_head_dim": np.asarray(BLOCK_HEAD_DIMENSION, dtype=np.int64),
    }
    result.update({f"cross_adaln_weight__{name}": _as_numpy(value) for name, value in block.state_dict().items()})
    return result


def _cross_adaln_av_block_golden(device: torch.device, generator: torch.Generator) -> dict[str, np.ndarray]:
    """Return CUDA BF16 vectors for the full AV prompt-side AdaLN path."""

    config = TransformerConfig(
        dim=BLOCK_DIMENSION,
        heads=BLOCK_HEADS,
        d_head=BLOCK_HEAD_DIMENSION,
        context_dim=BLOCK_CROSS_ADALN_CONTEXT_DIMENSION,
        cross_attention_adaln=True,
    )
    block = BasicAVTransformerBlock(video=config, audio=config).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * BLOCK_PARAMETER_STANDARD_DEVIATION
            )
    video_input = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_DIMENSION), generator=generator, device=device, dtype=torch.bfloat16
    )
    audio_input = torch.randn(
        (1, AV_AUDIO_TOKENS, BLOCK_DIMENSION), generator=generator, device=device, dtype=torch.bfloat16
    )
    context = torch.randn(
        (1, BLOCK_CONTEXT_TOKENS, BLOCK_CROSS_ADALN_CONTEXT_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    video_timesteps = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, BLOCK_CROSS_ADALN_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_timesteps = torch.randn(
        (1, AV_AUDIO_TOKENS, BLOCK_CROSS_ADALN_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    video_prompt_timestep = torch.randn(
        (1, PROMPT_TIMESTEP_TOKEN_COUNT, 2 * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_prompt_timestep = torch.randn(
        (1, PROMPT_TIMESTEP_TOKEN_COUNT, 2 * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    video_cross_timestep = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, AV_CROSS_SCALE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_cross_timestep = torch.randn(
        (1, AV_AUDIO_TOKENS, AV_CROSS_SCALE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    video_cross_gate_timestep = torch.randn(
        (1, BLOCK_VIDEO_TOKENS, AV_CROSS_GATE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_cross_gate_timestep = torch.randn(
        (1, AV_AUDIO_TOKENS, AV_CROSS_GATE_PARAMETER_COUNT * BLOCK_DIMENSION),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    shared = {
        "context": context,
        "context_mask": None,
        "embedded_timestep": torch.zeros((1, 1), device=device, dtype=torch.bfloat16),
        "positional_embeddings": None,
        "cross_positional_embeddings": None,
        "enabled": True,
        "cross_attn_perturbation_mask": torch.ones(AV_PERTURBATION_MASK_SHAPE, device=device, dtype=torch.bfloat16),
    }
    video = TransformerArgs(
        x=video_input,
        timesteps=video_timesteps,
        prompt_timestep=video_prompt_timestep,
        cross_scale_shift_timestep=video_cross_timestep,
        cross_gate_timestep=video_cross_gate_timestep,
        **shared,
    )
    audio = TransformerArgs(
        x=audio_input,
        timesteps=audio_timesteps,
        prompt_timestep=audio_prompt_timestep,
        cross_scale_shift_timestep=audio_cross_timestep,
        cross_gate_timestep=audio_cross_gate_timestep,
        **shared,
    )
    video_output, audio_output = block(video, audio)
    assert video_output is not None and audio_output is not None
    result = {
        "cross_adaln_av_video_input": _as_numpy(video_input),
        "cross_adaln_av_audio_input": _as_numpy(audio_input),
        "cross_adaln_av_context": _as_numpy(context),
        "cross_adaln_av_video_timesteps": _as_numpy(video_timesteps),
        "cross_adaln_av_audio_timesteps": _as_numpy(audio_timesteps),
        "cross_adaln_av_video_prompt_timestep": _as_numpy(video_prompt_timestep),
        "cross_adaln_av_audio_prompt_timestep": _as_numpy(audio_prompt_timestep),
        "cross_adaln_av_video_cross_timestep": _as_numpy(video_cross_timestep),
        "cross_adaln_av_audio_cross_timestep": _as_numpy(audio_cross_timestep),
        "cross_adaln_av_video_cross_gate_timestep": _as_numpy(video_cross_gate_timestep),
        "cross_adaln_av_audio_cross_gate_timestep": _as_numpy(audio_cross_gate_timestep),
        "cross_adaln_av_video_output": _as_numpy(video_output.x),
        "cross_adaln_av_audio_output": _as_numpy(audio_output.x),
        "cross_adaln_av_heads": np.asarray(BLOCK_HEADS, dtype=np.int64),
        "cross_adaln_av_head_dim": np.asarray(BLOCK_HEAD_DIMENSION, dtype=np.int64),
    }
    result.update(
        {f"{CROSS_ADALN_AV_BLOCK_WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in block.state_dict().items()}
    )
    return result


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")

    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16

    rope_input = torch.randn(ROPE_SHAPE, generator=generator, device=device, dtype=dtype)
    angles = torch.randn((*ROPE_SHAPE[:-1], ROPE_SHAPE[-1] // 2), generator=generator, device=device).to(dtype)
    rope_cos = torch.cos(angles)
    rope_sin = torch.sin(angles)
    rope_output = _split_rope(rope_input, rope_cos, rope_sin)

    norm_input = torch.randn(GELU_SHAPE, generator=generator, device=device, dtype=dtype)
    norm_weight = torch.randn((GELU_SHAPE[-1],), generator=generator, device=device, dtype=dtype)
    norm_output = torch.nn.functional.rms_norm(
        norm_input,
        (GELU_SHAPE[-1],),
        weight=norm_weight,
        eps=DEFAULT_EPSILON,
    )
    gelu_output = torch.nn.functional.gelu(norm_input, approximate="tanh")

    na_query = torch.randn(NEIGHBORHOOD_ATTENTION_SHAPE, generator=generator, device=device, dtype=dtype)
    na_key = torch.randn(NEIGHBORHOOD_ATTENTION_SHAPE, generator=generator, device=device, dtype=dtype)
    na_value = torch.randn(NEIGHBORHOOD_ATTENTION_SHAPE, generator=generator, device=device, dtype=dtype)
    na_output = na3d(
        na_query,
        na_key,
        na_value,
        kernel_size=NEIGHBORHOOD_ATTENTION_KERNEL,
    )
    block_golden = _video_block_golden(device, generator)
    av_block_golden = _av_block_golden(device, generator)
    cross_adaln_block_golden = _cross_adaln_block_golden(device, generator)
    cross_adaln_av_block_golden = _cross_adaln_av_block_golden(device, generator)

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "primitive_bf16_cuda.npz",
        rope_input=_as_numpy(rope_input),
        rope_cos=_as_numpy(rope_cos),
        rope_sin=_as_numpy(rope_sin),
        rope_output=_as_numpy(rope_output),
        norm_input=_as_numpy(norm_input),
        norm_weight=_as_numpy(norm_weight),
        norm_output=_as_numpy(norm_output),
        gelu_output=_as_numpy(gelu_output),
        na_query=_as_numpy(na_query),
        na_key=_as_numpy(na_key),
        na_value=_as_numpy(na_value),
        na_output=_as_numpy(na_output),
        na_kernel_size=np.asarray(NEIGHBORHOOD_ATTENTION_KERNEL, dtype=np.int64),
        **block_golden,
        **av_block_golden,
        **cross_adaln_block_golden,
        **cross_adaln_av_block_golden,
    )
    metadata = {
        "artifact": "primitive_bf16_cuda.npz",
        "device": str(device),
        "dtype": str(dtype),
        "epsilon": DEFAULT_EPSILON,
        "gpu_name": torch.cuda.get_device_name(device),
        "neighborhood_attention_kernel": list(NEIGHBORHOOD_ATTENTION_KERNEL),
        "neighborhood_attention_shape": list(NEIGHBORHOOD_ATTENTION_SHAPE),
        "video_block": {
            "context_dim": BLOCK_CONTEXT_DIMENSION,
            "head_dim": BLOCK_HEAD_DIMENSION,
            "heads": BLOCK_HEADS,
            "hidden_dim": BLOCK_DIMENSION,
            "video_tokens": BLOCK_VIDEO_TOKENS,
        },
        "av_block": {
            "audio_tokens": AV_AUDIO_TOKENS,
            "context_dim": BLOCK_CONTEXT_DIMENSION,
            "head_dim": BLOCK_HEAD_DIMENSION,
            "heads": BLOCK_HEADS,
            "hidden_dim": BLOCK_DIMENSION,
            "video_tokens": BLOCK_VIDEO_TOKENS,
        },
        "cross_adaln_block": {
            "context_dim": BLOCK_CROSS_ADALN_CONTEXT_DIMENSION,
            "head_dim": BLOCK_HEAD_DIMENSION,
            "heads": BLOCK_HEADS,
            "hidden_dim": BLOCK_DIMENSION,
            "video_tokens": BLOCK_VIDEO_TOKENS,
        },
        "cross_adaln_av_block": {
            "audio_tokens": AV_AUDIO_TOKENS,
            "context_dim": BLOCK_CROSS_ADALN_CONTEXT_DIMENSION,
            "head_dim": BLOCK_HEAD_DIMENSION,
            "heads": BLOCK_HEADS,
            "hidden_dim": BLOCK_DIMENSION,
            "video_tokens": BLOCK_VIDEO_TOKENS,
        },
        "seed": seed,
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    (output_directory / "primitive_bf16_cuda.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    generate(arguments.output_directory, arguments.device, arguments.seed)


if __name__ == "__main__":
    main()
