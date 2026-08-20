# Architecture

## Product boundary

Lara is a native, in-process Apple Silicon model runtime. The MLX/Metal model
execution path is the product; it does not depend on a network server.

```text
Direct Python API (primary)
       ▲          ▲
       │          │
thin CLI      thin ComfyUI nodes (optional, late gate)
       └──────┬───┘
              ▼
       Lara MLX Pipeline
              │
   ┌──────────┼───────────┐
   ▼          ▼           ▼
Loader    Conditioning  Sampler
   │          │           │
   └──────► Transformer ◄──┘
              │
        Upscaler / VAE
              │
          Media result
              │
              ▼
         MLX → Metal
```

HTTP servers, daemons, batching services, authentication, databases and custom
web applications are outside the repository's architecture.

## Canonical computation

Lara mirrors the official `TI2VidTwoStagesHQPipeline` at pinned upstream commit
`400fd31054597515f47125691032c04b1c3ee24e`:

1. LTX-specific Gemma 4 12B produces positive and negative conditioning.
2. The full 22B DEV transformer performs guided stage-one BF16 denoising.
3. The BF16 latent spatial upscaler doubles spatial resolution.
4. The DEV transformer with the official distilled LoRA performs refinement.
5. The diffusion video VAE decodes video latents.
6. The audio VAE and vocoder decode synchronized audio.
7. The media layer writes and muxes the final local result.

CUDA/PyTorch is the behavioral specification. Community ports may inform
operator design but cannot override measured upstream behavior.

## Core responsibility boundaries

- `lara_ltx.models` and loader code: streaming checkpoint parsing, key mapping
  and BF16 parameter ownership.
- `lara_ltx.conditioning`: tokenizer, Gemma 4 encoder and LTX projections.
- `lara_ltx.transformer`: transformer blocks, attention, RoPE, AdaLN and output
  projection.
- `lara_ltx.sampling`: scheduler, CFG/STG and res_2s integration.
- `lara_ltx.upscaler`: spatial latent upscaler.
- `lara_ltx.vae`: diffusion video decoder, audio VAE, vocoder and tiling.
- `lara_ltx.ops`: reusable MLX reference and fused operations.
- `lara_ltx.metal`: only custom kernels proven necessary by profiling.
- `lara_ltx.pipeline`: the one orchestration path used by every interface.
- `lara_ltx.media`: result object, frame/audio conversion and save handoff.
- `lara_ltx.parity`: tensor metadata, comparison metrics and reports.

## Interface dependency rule

```text
CLI ─────────┐
             ├──► public Python API ─► pipeline ─► MLX/Metal core
ComfyUI ─────┘
```

Dependencies may point only from adapters toward the public API. Core modules
must not import CLI, ComfyUI or any serving integration. The ComfyUI package may
convert node inputs/outputs and cache a pipeline instance, but it may not own
checkpoint mapping, transformer, scheduler, VAE or Metal implementations.

The final release exposes three official entry points: direct Python for
developers, the thin CLI for terminal users, and optional ComfyUI custom nodes
for general Mac users. ComfyUI is never a required dependency of the model or
the Python/CLI installation.

## Runtime constraints

- Normal Mac inference has no PyTorch or CUDA dependency.
- Primary checkpoint and activations remain BF16 unless an operation requires a
  documented higher-precision accumulator.
- Weight loading is shard-at-a-time and avoids simultaneous duplicate model
  copies. The evaluated active component remains resident for its pipeline
  phase; the runtime does not swap every layer from disk on every denoising
  pass.
- MLX evaluation boundaries and object lifetimes are explicit to control unified
  memory.
- Unsupported capabilities fail with localized stable error codes.
- Optimized and custom-kernel paths always retain a reference path for parity.

## Component residency

```text
Gemma conditioning
  → retain embeddings, release Gemma
DEV transformer + stage-1 LoRA
  → retain latents, release stage-1 temporaries
spatial upscaler
  → retain high-resolution latents, release upscaler
DEV transformer + stage-2 LoRA
  → retain final video/audio latents, release transformer
Diffusion video VAE → audio VAE/vocoder → streaming media writer
```

The stage-specific LoRA strengths mean the correct low-memory baseline may
reload and fuse the transformer shard-by-shard between stages. Keeping an
unfused LoRA resident is an optimization candidate, not an architectural
assumption.

At the canonical 121-frame 1088 x 1920 target, the stage-two latent grid is
16 x 34 x 60 = 32,640 video tokens. Fused SDPA is mandatory and an explicit
`tokens x tokens` score allocation is forbidden. The initial reference path
evaluates each transformer block so large feed-forward intermediates can be
released before the next block graph grows.
