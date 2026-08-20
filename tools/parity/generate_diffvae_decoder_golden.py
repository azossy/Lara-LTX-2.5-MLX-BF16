#!/usr/bin/env python3
"""Generate CUDA BF16 vectors for the complete untiled Diffusion VAE path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from ltx_core.model.video_vae.transformer import EagerSdpaAttention

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_913
IN_CHANNELS = 4
OUT_CHANNELS = 3
PATCH_SIZE = 2
HEAD_DIM = 16
STAGE_CHANNELS = (16, 16, 16, 16, 16)
STAGE_DEPTHS = (1, 1, 1, 1, 2)
STAGE_KERNELS = ((3, 3, 3),) * 5
UPSAMPLES = (((1, 2, 2), 1), ((1, 1, 1), 1), ((1, 1, 1), 1), ((1, 1, 1), 1))
STAGE5_KERNEL = (3, 3, 3)
TIMESTEP_EMBEDDING_DIM = 16
INFERENCE_STEPS = 2
LATENT_SHAPE = (1, IN_CHANNELS, 3, 3, 3)
NOISE_SHAPE = (1, OUT_CHANNELS, 3, 12, 12)
PARAMETER_STANDARD_DEVIATION = 0.02
WEIGHT_PREFIX = "diffvae_decoder_weight__"
OUTPUT_STEM = "diffvae_decoder_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _use_eager_attention(decoder: DiffusionVideoDecoder) -> None:
    for module in decoder.modules():
        if hasattr(module, "attention_function"):
            module.attention_function = EagerSdpaAttention()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    decoder = DiffusionVideoDecoder(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        patch_size=PATCH_SIZE,
        head_dim=HEAD_DIM,
        stage_channels=STAGE_CHANNELS,
        stage_depths=STAGE_DEPTHS,
        stage_kernels=STAGE_KERNELS,
        upsamples=UPSAMPLES,
        stage5_kernel=STAGE5_KERNEL,
        t_emb_dim=TIMESTEP_EMBEDDING_DIM,
        default_num_inference_steps=INFERENCE_STEPS,
        model_output_type="v",
    ).to(device=device, dtype=torch.bfloat16)
    _use_eager_attention(decoder)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16)
                * PARAMETER_STANDARD_DEVIATION
            )
        latent = torch.randn(LATENT_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
        initial_noise = torch.randn(NOISE_SHAPE, generator=generator, device=device, dtype=torch.bfloat16)
        timesteps = decoder.default_inference_timesteps[None].to(device).expand(latent.shape[0], -1)
        stage3 = decoder.forward_stages_1_to_3(latent)
        context = decoder.forward_stage_4(stage3, pad_trailing=False)
        x_t = initial_noise
        model_outputs: list[torch.Tensor] = []
        for step_index in range(timesteps.shape[1]):
            timestep_now = timesteps[:, step_index]
            context_and_x = decoder._context_and_x_for_diff_step(context, x_t)
            model_output = decoder.forward_diff_step(context_and_x, timestep_now)
            model_outputs.append(model_output)
            timestep_next = (
                timesteps[:, step_index + 1] if step_index + 1 < timesteps.shape[1] else torch.zeros_like(timestep_now)
            )
            x_t = decoder._euler_step(x_t, model_output.float(), timestep_now, timestep_next)

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        latent=_as_numpy(latent),
        initial_noise=_as_numpy(initial_noise),
        timesteps=_as_numpy(timesteps),
        stage3=_as_numpy(stage3),
        context=_as_numpy(context),
        model_output_0=_as_numpy(model_outputs[0]),
        model_output_1=_as_numpy(model_outputs[1]),
        output=_as_numpy(x_t),
        **{f"{WEIGHT_PREFIX}{name}": _as_numpy(value) for name, value in decoder.state_dict().items()},
    )
    metadata = {
        "artifact": f"{OUTPUT_STEM}.npz",
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "seed": seed,
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    (output_directory / f"{OUTPUT_STEM}.json").write_text(
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
