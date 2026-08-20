# Development and porting plan

## Scope authority

`Lara-LTX-2.5-MLX-BF16_Serving_Integration_Scope.md` is the controlling scope
directive. The project delivers a native Apple Silicon MLX/Metal port of the
official LTX-2.5 DEV 22B BF16 model. It is not a serving-engine project.

The first-class product is one in-process inference implementation:

```text
Python API (primary) ─┬─ optional CLI
                      └─ optional thin ComfyUI adapter
                                │
                                ▼
                    Lara MLX inference pipeline
                                │
                                ▼
                         MLX → Metal → Apple GPU
```

The CLI and ComfyUI adapter may only call the public Python pipeline. They must
never introduce a second PyTorch/MPS or model-execution path.

## Included scope

- Official DEV 22B BF16 checkpoint loading and explicit weight mapping.
- Gemma 4 conditioning and LTX projections.
- Video/audio transformer, attention, RoPE, AdaLN and modality interaction.
- Scheduler, guidance, latent generation and deterministic-input handling.
- Spatial latent upscaling and two-stage HQ refinement.
- Diffusion video VAE, audio VAE, vocoder and media output.
- MLX operations and narrowly justified custom Metal kernels.
- CUDA tensor/stage/video golden references and regression tooling.
- Apple unified-memory correctness, profiling and optimization.
- Small stable direct Python API, followed by a thin CLI.
- Optional ComfyUI custom nodes after the core pipeline is stable.

## Explicitly excluded

- HTTP, REST or OpenAI-compatible servers.
- vLLM/Ollama-style daemons or continuous batching.
- Authentication, accounts, databases, job backends and rate limiting.
- Multi-user, distributed, multi-node, Kubernetes or cloud serving.
- SaaS infrastructure, proprietary web dashboards or custom Web UI.
- A separate PyTorch/MPS implementation inside ComfyUI.

Any future serving integration must live outside the core package and consume
the same public Python API. It is not a release requirement for this project.

## Priority and gates

Work proceeds in the following strict order. A later gate may be scaffolded,
but it cannot displace an unmet earlier gate.

The failure analysis and bottleneck decisions in
`docs/FAILURE_PREVENTION_REVIEW.md` are binding implementation constraints.
In particular, component-level residency is the default; per-layer disk
swapping is not an accepted primary runtime design.

### P0 — Official CUDA Golden Reference

Deliverables:

- Pinned source, model revision, dependency lock and hardware manifest.
- Verified DEV 22B BF16 component files with size and SHA-256 manifests.
- Successful official PyTorch/CUDA two-stage BF16 run without quantization.
- Fixed prompt/seed/resolution matrix and serialized initial noise.
- Primitive, conditioning, representative block, sampler, latent, decoder,
  audio and final-media captures.
- A parallel target-Mac feasibility record covering real Metal execution,
  stage-one/stage-two fused SDPA shapes, block-boundary peak memory and one 3D
  neighborhood-attention reference case.

Exit criteria:

- The exact environment and every reference artifact are reproducible.
- The baseline produces valid video and synchronized audio on CUDA.
- Tensor dumps include dtype, shape, statistics and checksums.
- Maximum-shape SDPA does not materialize a quadratic attention-score tensor,
  and no required MLX operation remains without an owned implementation path.

### P1 — Correct MLX weight loading

Deliverables:

- Allocation-safe safetensors inspection and shard-at-a-time component loading
  that avoids duplicate full-model residency.
- Shard-scoped mapped weight batches so model construction can apply and release
  each validated source shard before advancing.
- Versioned source-key to MLX-parameter mapping table.
- Header-derived mapping templates in which every target and layout transform
  must be assigned explicitly before conversion.
- Pre-load mapping validation that rejects unresolved or duplicate targets,
  unsupported transforms and mismatches with the MLX component key set.
- Explicit layout transforms with shape/dtype validation.
- BF16 preservation report and unmapped/duplicate-key failures.
- Component-aware lazy loading and the measured Gemma → transformer/upscaler
  → VAE residency schedule. Per-layer disk swapping inside denoising is not
  used as the normal path.
- Stage-local, shard-wise distilled-LoRA application with no second full
  transformer copy; any dynamic unfused alternative requires parity and
  performance evidence.

Exit criteria:

- Every required DEV transformer key is accounted for exactly once.
- Loaded MLX tensors retain expected BF16 values and shapes.
- Failure messages use documented stable error codes.

### P2 — Core MLX forward path

Implementation order:

1. RMSNorm, GELU, linear layers and Split RoPE.
2. Fused memory-efficient self/cross attention.
3. AdaLN, feed-forward and output modulation.
4. Video transformer blocks.
5. Audio transformer blocks and audio/video cross-attention.
6. Patchification, timestep embeddings, conditioning and output projection.
7. Bounded 3D neighborhood-attention reference needed by Diffusion VAE.

Exit criteria:

- One representative block runs from mapped checkpoint weights on MLX.
- Video and audio branches execute without PyTorch in the Mac path.
- Primitive and block comparisons meet provisional BF16 tolerances.
- The canonical 32,640-token shape fits the configured working-set budget with
  explicit evaluation boundaries.

### P3 — End-to-end local generation

Deliverables:

- Gemma 4 tokenizer/encoder/projection path, adapted from the proven MLX-LM
  Gemma 4 architecture where compatible and returning the exact LTX hidden
  states without normal-path LM logits.
- Scheduler, CFG/STG and res_2s sampler.
- Stage-one denoising, spatial upscaling and stage-two refinement.
- Diffusion video decoder, audio VAE/vocoder and MP4 muxing.
- An internal in-process runner used by tests; no server process.

Exit criteria:

- A fixed prompt produces a playable local video with synchronized audio.
- Repeated seeded runs satisfy the documented determinism contract.
- Normal Mac inference imports neither PyTorch nor CUDA packages.
- Text-to-video with synchronized audio passes first; the optional-image path
  is then enabled through the same HQ pipeline only after conditioning parity.

### P4 — Tensor and stage parity

Deliverables:

- Automated CUDA-versus-MLX reports at all major checkpoints.
- Per-layer bisect tooling for the first divergent transformer block.
- Separate RNG parity and computation parity reports.

Exit criteria:

- No unexplained major systematic divergence remains.
- Tolerances are evidence-based and frozen in versioned test configuration.

### P5 — BF16 output-quality parity

Deliverables:

- Multi-prompt, seed, duration and resolution regression corpus.
- Frame, temporal-motion, perceptual and audio-sync measurements.
- Blind side-by-side review records and known-difference documentation.

Exit criteria:

- End-to-end differences are perceptually immaterial for the acceptance set.
- No major capability is removed or silently approximated.

### P6 — MLX/Metal performance optimization

Optimization order:

1. Correct per-block evaluation boundaries and component lifetime reduction.
2. MLX built-in fused operators and `mx.compile` where numerically safe.
3. Chunking/tiling that preserves observable output behavior.
4. Custom Metal kernels only for measured unresolved bottlenecks.

Exit criteria:

- M5 Max 128 GB peak memory, load time, generation time and GPU utilization
  are recorded.
- Repeated generation has no unbounded unified-memory growth.
- Every optimization passes the complete parity and quality suite.
- Stage timing and MLX active/cache/peak memory are recorded separately;
  `mx.clear_cache()` is not used in transformer inner loops.

### P7 — Stable direct Python API

Target surface:

```python
from lara_ltx import LTXPipeline

pipeline = LTXPipeline.from_pretrained("LaraAI/Lara-LTX-2.5-MLX-BF16")
video = pipeline(prompt="A cinematic aerial shot", seed=42)
video.save("output.mp4")
```

The exact arguments may evolve before this gate, but the public surface must
remain small, typed and in-process. Loading, generation and saving must require
no external daemon.

### P8 — Thin CLI

- Implement `lara-ltx generate` as argument parsing around `LTXPipeline`.
- Do not duplicate loading, scheduling, generation or media logic.
- Verify installation and generation from a clean environment.

### P9 — Thin ComfyUI adapter

- Prefer a separate `ComfyUI-LaraLTX` integration package/repository.
- Initial nodes: model loader, text-to-video and decode/output.
- Add image-to-video only after the core image-conditioning path passes parity.
- Reuse loaded pipeline state when safe.
- Keep transformer, checkpoint mapping, VAE, scheduler and Metal code in
  `lara_ltx`; never copy them into the plugin.

### P10 — GitHub and Hugging Face release

- Publish the verified source, tests and release notes to GitHub.
- Publish the verified BF16 MLX model package, loading metadata and model card
  to Hugging Face.
- Use the same release version, source commit and compatibility statement on
  both services.
- Put a short Quick Start before lengthy implementation details.
- Present the three official usage paths: Python, CLI and optional ComfyUI.
- State plainly: no server required, no CUDA required for normal Mac inference,
  and ComfyUI is optional.
- Document supported Mac hardware, measured memory/load/generation benchmarks
  and CUDA-versus-MLX quality results.
- Do not publish performance or quality claims before the corresponding P5/P6
  evidence is recorded.
- Verify all install/load/generate instructions in a clean environment.
- Verify the public GitHub repository, tagged release assets and Hugging Face
  model repository after publication.

## Current execution status

| Item | State | Next evidence |
|---|---|---|
| Upstream source/model pin | Complete | Re-verify at every golden run |
| Project config, localized errors and manifests | Implemented | Unit/lint pass |
| Tensor comparison metrics | Implemented | CUDA/MLX artifact report |
| MLX RMSNorm, GELU, Split RoPE and fused SDPA | CUDA-BF16 parity passed | Checkpoint-backed component parity |
| MLX video/audio AV block | Video, video text cross-AdaLN and non-cross-AdaLN AV CUDA-BF16 parity passed | Checkpoint-backed block parity |
| Failure/bottleneck review | Complete | SDPA/block probes passed; execute 3D-NA probe |
| M5 Max maximum-shape feasibility | SDPA and synthetic block passed | Checkpoint-backed block and DiffVAE 3D-NA parity |
| CUDA environment on Blackwell | P0 baseline and internal boundary trace captured | Consume the golden tensors for checkpoint-backed MLX component parity |
| Gated official BF16 weights | Downloaded and verified: size, SHA-256 and reviewed mixed-precision header policy | Run the official two-stage BF16 golden pipeline |
| Official HQ golden runner | P0 smoke run captured and media-validated | Instrument and capture internal CUDA boundaries |
| Diffusion VAE checkpoint mapping | Implemented and header-audited | Load all 407 mapped targets on MLX and run checkpoint parity |
| Mixed-precision shard loading | Requested-tensor policy implemented | Exercise official BF16/F32 mapped targets on target Mac MLX runtime |
| End-to-end MLX generation | Pending P3 | Fixed-prompt playable MP4 |
| Python API / CLI / ComfyUI / HF packaging | Deferred to P7/P8/P9/P10 | Core parity gates first |

## Repository direction

Core code remains in the installable `lara_ltx` package and converges on these
boundaries:

```text
lara_ltx/
├── models/          # model structures and mapped parameters
├── pipeline/        # the single end-to-end inference path
├── conditioning/    # tokenizer, Gemma 4 and projections
├── transformer/     # attention, RoPE, AdaLN and blocks
├── sampling/        # scheduler, guidance and samplers
├── upscaler/        # latent spatial refinement
├── vae/             # video/audio decode and vocoder
├── ops/             # portable MLX operators
├── metal/           # only parity-tested custom kernels
├── media/           # frames/audio result and encoding handoff
└── cli/             # thin wrapper around pipeline
```

Tests are separated into unit, CUDA/MLX parity, end-to-end regression, quality
and performance groups. Golden artifacts record their upstream revision and
environment; large weight files are never committed to Git.

## Change-control rules

- Correctness and BF16 quality precede performance and integration work.
- Quantization is not permitted in the primary release path.
- An unsupported operation must fail explicitly; it cannot silently fall back
  to a lower-quality approximation.
- Sampling or conditioning changes require a golden comparison and documented
  rationale.
- Custom Metal code requires a benchmark, a reference implementation and
  numerical tests.
- No serving-engine feature may enter the core dependency graph.
- No feature may assume true out-of-core MLX execution unless an upstream MLX
  capability and a target-Mac benchmark prove it.
- Specialized upstream pipelines outside canonical HQ T2V/I2V are post-v1
  work and cannot delay the release-critical path.
