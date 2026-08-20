# Failure-prevention plan review

## Decision

The project is technically feasible on the primary M5 Max 128 GB target, but
only if memory is managed at **component and pipeline-stage boundaries**. It
must not depend on per-layer disk swapping or an assumed out-of-core MLX
runtime. The release path remains BF16 and quality-first.

The minimum successful product is the official LTX-2.5 DEV 22B
`TI2VidTwoStagesHQPipeline` behavior for text-to-video with synchronized audio,
plus the optional-image conditioning path after it passes the same parity
gates. Python is the primary interface. CLI and ComfyUI remain thin adapters.

## What was removed or deferred

- Per-layer weight swapping during every denoising pass is removed from the
  primary design. MLX file loads can be lazy, but this is not a stable
  substitute for true out-of-core execution; repeated storage faults would be
  catastrophic across many layers and denoising passes.
- Multi-Mac, distributed inference, tensor parallelism and multi-node work are
  excluded. They do not help the single-Mac product and would multiply parity
  and deployment risk.
- Custom Metal kernels are not a pre-implementation workstream. A kernel is
  allowed only after an MLX reference operation is correct and a profiler shows
  that the operation materially limits end-to-end runtime.
- Specialized upstream applications such as retake, Dub-It, HDR IC-LoRA,
  audio-to-video and training are not v1 release requirements. They are
  separate products, not prerequisites for the canonical HQ T2V/I2V pipeline.
- Prompt enhancement is deferred until the ordinary LTX conditioning path is
  correct. It must not force the Gemma model to remain resident during
  diffusion.
- A large permanent full-tensor dump for every layer is removed. The normal
  parity corpus stores compact metadata/checksums and a selected layer set;
  full tensors are captured only around the first divergence.

## What was added

### 1. Early feasibility gate

Before completing the full port, the following probes must pass on the target
Mac:

1. A real Metal operation in the exact pinned Python/MLX environment.
2. Fused SDPA at the canonical stage-one and stage-two token shapes without an
   explicit quadratic score tensor.
3. A representative 4096-wide transformer block or equivalent synthetic
   workload with measured peak memory and evaluation boundaries.
4. An allocation-safe safetensors header/key inspection of every component.
5. An MLX reference implementation of one 3D neighborhood-attention case that
   matches a CUDA golden before the complete Diffusion VAE is ported.

A full port may continue only when no required operation is unowned and the
measured memory model fits the target working-set budget. A slow probe is a
performance risk to investigate; an allocation failure or quadratic attention
materialization is a correctness/feasibility blocker.

### 2. Shape-derived memory model

The canonical HQ preset is 121 frames at 1088 x 1920 output. With the official
VAE factors `(time=8, height=32, width=32)` and patch size 1:

| Stage | Pixel shape | Latent grid | Video tokens |
|---|---:|---:|---:|
| Stage 1 | 121 x 544 x 960 | 16 x 17 x 30 | 8,160 |
| Stage 2 | 121 x 1088 x 1920 | 16 x 34 x 60 | 32,640 |

At 32 heads x 128 dimensions, a stage-two BF16 Q, K or V tensor is about
255 MiB. Q/K/V together are about 765 MiB before output and workspace. A dense
attention-score tensor would be far larger and is forbidden. The 4x
feed-forward expansion is also a multi-GiB transient at this token count, so a
48-block lazy graph must not accumulate unchecked.

The first correct path evaluates at each transformer block boundary. Later
profiling may safely group or compile stable blocks, but only while peak memory
and parity remain within the recorded limits.

### 3. Component residency schedule

Only the active component set may remain evaluated in unified memory:

| Phase | Resident | Required transition |
|---|---|---|
| Conditioning | Gemma 4 + tokenizer + LTX projection | Evaluate and retain only positive/negative conditioning; release Gemma and clear reclaimable cache |
| Stage 1 | DEV transformer + stage-one LoRA application + conditioning/latents | Release all stage-one-only temporaries before upscaling |
| Upscale | Spatial upscaler + stage-one latents | Release upscaler after the high-resolution latent exists |
| Stage 2 | DEV transformer + stage-two LoRA application + conditioning/latents | Release transformer before pixel/audio decode |
| Decode | Diffusion video VAE, then audio VAE/vocoder | Yield encoded chunks to media output; do not retain all decoded frames twice |

The official distilled LoRA uses different strengths in the two stages. The
safe baseline reloads and fuses the base transformer shard-by-shard at each
stage, so it never needs a second complete transformer copy. A resident
unfused-LoRA path is allowed only if it is both parity-correct and faster after
measurement.

`mx.clear_cache()` is a stage-boundary tool, not an inner-loop operation.
Repeated inner-loop cache clearing would trade memory predictability for a
large allocation and synchronization penalty.

### 4. Diffusion VAE solution path

The Diffusion VAE is the highest implementation-risk component because the
official fast paths use NATTEN, Triton or Blackwell-specific kernels. The
portable path will preserve the official 3D neighborhood semantics as follows:

1. Port the upstream eager fallback's window-bound and tile-selection rules.
2. Flatten each bounded `(T,H,W)` query/key tile into MLX fused SDPA batches.
3. Keep additive masks bounded by a configurable element budget.
4. Reassemble output tiles without creating a global attention matrix.
5. Validate kernels `(3,7,7)`, `(3,5,5)` and `(3,3,3)`, edge windows, trailing
   temporal padding and tiling overlap against CUDA.
6. Profile the correct reference path. Write a custom Metal neighborhood
   kernel only if this operation remains the dominant unresolved bottleneck.

This brings VAE risk into P0/P2 probes instead of discovering it after the
transformer has already been ported.

### 5. Gemma 4 reuse decision

MLX-LM now contains a native Gemma 4 implementation. Lara should reuse its
proven architecture and operator decisions as the starting point, but must
adapt it to the exact LTX checkpoint contract: no LM-head logits during normal
conditioning, all required hidden states, the LTX tokenizer behavior, and the
LTX video/audio projection connectors. The exact LTX checkpoint remains the
source of truth, and the adapter receives its own CUDA parity tests.

This avoids a needless second Gemma 4 port without making generic text
generation or an MLX-LM server part of Lara.

### 6. Observability and stop conditions

Every major phase records elapsed time, active/cache/peak MLX memory, input
shape and component lifecycle. A generation fails before allocation when its
shape-derived estimate exceeds the configured budget and reports a localized
error code with a smaller safe shape.

Development stops for redesign, rather than masking the problem, if any of the
following occurs:

- maximum-shape SDPA allocates a quadratic score tensor;
- the component-residency schedule cannot leave the configured system reserve;
- an upstream operation has no parity-preserving MLX/reference implementation;
- a required checkpoint key cannot be mapped exactly;
- the only path to completion requires quantizing the primary BF16 release;
- a performance optimization causes a frozen quality/parity regression.

## Evidence sources

Measured target-Mac evidence is recorded in `docs/MLX_FEASIBILITY_RESULTS.md`.
The maximum 32,640-token BF16 fused-SDPA and synthetic 4096-wide block probes
passed without quadratic score storage. Checkpoint-backed and VAE probes remain
open.

- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX fused scaled dot-product attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
- [MLX memory limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)
- [MLX cache control](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)
- [MLX wired-memory limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
- [MLX-LM Gemma 4 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
- Pinned upstream `TI2VidTwoStagesHQPipeline`, `DiffusionVideoDecoder` and eager
  3D neighborhood-attention fallback under `upstream/LTX-2`.
