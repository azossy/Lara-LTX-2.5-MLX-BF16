#!/usr/bin/env python3
"""Generate CUDA BF16 vectors for production DiffVAE padding and crop geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ltx_core.model.video_vae import diffusion_tiling
from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from ltx_core.model.video_vae.transformer import EagerSdpaAttention

DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 25_081_914
STAGE_CHANNELS = (16, 16, 16, 16, 16)
STAGE_DEPTHS = (1, 1, 1, 1, 1)
STAGE_KERNELS = ((3, 3, 3),) * 5
UPSAMPLES = (((1, 2, 2), 1), ((2, 1, 1), 1), ((2, 2, 2), 1), ((2, 2, 2), 1))
WEIGHT_PREFIX = "diffvae_padding_weight__"
OUTPUT_STEM = "diffvae_padding_bf16_cuda"


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def generate(output_directory: Path, device_name: str, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("[LARA-RUNTIME-002] CUDA is unavailable. Start a GPU instance and retry.")
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    decoder = DiffusionVideoDecoder(
        in_channels=4,
        out_channels=3,
        patch_size=4,
        head_dim=16,
        stage_channels=STAGE_CHANNELS,
        stage_depths=STAGE_DEPTHS,
        stage_kernels=STAGE_KERNELS,
        upsamples=UPSAMPLES,
        stage5_kernel=(3, 3, 3),
        t_emb_dim=16,
        default_num_inference_steps=1,
    ).to(device=device, dtype=torch.bfloat16)
    for module in decoder.modules():
        if hasattr(module, "attention_function"):
            module.attention_function = EagerSdpaAttention()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.bfloat16) * 0.02
            )
        latent = torch.randn((1, 4, 1, 1, 1), generator=generator, device=device, dtype=torch.bfloat16)
        initial_noise = torch.randn((1, 3, 1, 32, 32), generator=generator, device=device, dtype=torch.bfloat16)
        padded_latent, (_, height_pad, width_pad) = diffusion_tiling.ensure_min_latent_shape(
            latent, decoder.stage_min_tile_sizes
        )
        noise, _ = diffusion_tiling.resize_axis(initial_noise, 2, 17, mode="repeat_last")
        noise, _ = diffusion_tiling.resize_axis(noise, 3, 96, mode="symmetric")
        noise, _ = diffusion_tiling.resize_axis(noise, 4, 96, mode="symmetric")
        ghosted = diffusion_tiling.pad_trailing_latent_for_natten_border(
            padded_latent, decoder._natten_trailing_pad_latent_frames
        )
        stage3 = decoder.forward_stages_1_to_3(ghosted)
        context = decoder.forward_stage_4(stage3, pad_trailing=True)
        timestep = torch.ones((1,), device=device, dtype=torch.float32)
        model_output = decoder.forward_diff_step(decoder._context_and_x_for_diff_step(context, noise), timestep)
        pixels = decoder._euler_step(noise, model_output.float(), timestep, torch.zeros_like(timestep))
        output = diffusion_tiling.crop_pixels_to_content(
            pixels,
            1,
            32,
            32,
            h_pad=height_pad,
            w_pad=width_pad,
            spatial_scale=(32, 32),
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / f"{OUTPUT_STEM}.npz",
        latent=_as_numpy(latent),
        initial_noise=_as_numpy(initial_noise),
        output=_as_numpy(output),
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
