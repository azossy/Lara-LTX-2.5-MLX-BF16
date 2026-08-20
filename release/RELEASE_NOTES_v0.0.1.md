# Lara-LTX 0.0.1 engineering preview

Lara-LTX 0.0.1 is the first installable, native MLX/Metal BF16 engineering
preview of the official LTX-2.5 DEV 22B two-stage video-and-audio pipeline for
Apple Silicon. It runs in-process through one Python implementation; no CUDA,
server, daemon or PyTorch inference fallback is used on Mac.

## Included

- Direct `LTXPipeline.from_pretrained(...)` Python API.
- Thin `lara-ltx generate` CLI using the same pipeline.
- Optional thin ComfyUI loader, text-to-video and save nodes.
- Official packed Gemma 4 conditioning and video/audio prompt connectors.
- Forty-eight resident AV transformer blocks with stage-local distilled LoRA.
- HQ Res2S two-stage sampling and x2 latent refinement.
- Diffusion Video VAE, Audio VAE, primary vocoder, BWE and H.264/AAC output.
- Localized, stable error codes and configuration-driven runtime settings.
- Lightweight Hugging Face descriptor pinned to the gated upstream BF16 pack;
  the official checkpoint is not duplicated.

## Measured target

The primary test machine is an M5 Max with 128 GB unified memory. The checked-in
320x512/17-frame public-API smoke produced synchronized H.264/AAC output in
73.4 seconds with a 40,749,191,990-byte MLX peak. These figures are measured
for that reduced workload and are not estimates for a canonical 1920x1088 run.

## Numerical status

Mapped weights, prompt connectors, representative and full transformer
boundaries, output heads, spatial upscaling, video decoding and complete audio
decoding have checkpoint-backed CUDA-versus-MLX evidence. The expanded trace
also identified and fixed the AV cross-timestep multiplier: both production
multipliers now come from official checkpoint metadata and equal 1000.
Checkpoint FP32 AdaLN tables are now cast to each BF16 runtime stream before
modulation, preventing silent full-block FP32 upcasting. Tanh GELU evaluates
its activation in FP32 and returns the original BF16 stream dtype, matching the
captured CUDA module boundary.

This remains an engineering preview, not a final CUDA-quality-parity claim.
Exact-input final transformer/output-head boundaries meet component criteria,
but CUDA/Metal BF16 differences are amplified by guidance and accumulate across
the stochastic trajectory, so the frozen final-latent gate does not yet pass.
The four-case CUDA audiovisual corpus is provided for continued perceptual and
blind review. Matching MLX outputs and paired diagnostics are complete for two
cases. The 512x512/33-frame MLX case was terminated by the operating system
after about 70 minutes with heavy swap growth on M5 Max 128 GB, so that setting
is explicitly unsupported in this preview. Thresholds are not weakened to
relabel the failing trajectory or incomplete quality corpus as a pass.

## Model access and license

Users must accept access to `Lightricks/LTX-2.5` and provide a Hugging Face read
token. The runtime is distributed under the LTX-2.x Community License
Agreement; see `LICENSE.md` and `NOTICE.md`. Commercial Entities, as defined by
that agreement, must obtain the required paid license before commercial use.
