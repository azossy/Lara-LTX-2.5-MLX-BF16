# Release acceptance criteria

A public release is accepted only when all required sections below pass. The
checklist is evidence-driven; marketing text cannot substitute for test output.

## Scope and architecture

- [x] The product is one direct in-process MLX/Metal inference pipeline.
- [x] The core has no HTTP server, daemon, serving engine, SaaS or Web UI.
- [x] CLI and ComfyUI depend on the public Python API, never the reverse.
- [x] No duplicate PyTorch/MPS inference implementation exists.

## Core model

- [x] Official LTX-2.5 DEV 22B BF16 weights load completely and exactly.
- [x] Normal Mac inference has no PyTorch or CUDA dependency.
- [x] Transformer, conditioning, scheduler, upscaler, video decoder, audio VAE
      and vocoder execute with MLX/Metal.
- [x] End-to-end local video and synchronized-audio generation succeeds.
- [x] Unsupported operations fail explicitly instead of silently degrading.
- [x] Deterministic-seed behavior is documented and tested.
- [ ] Canonical HQ text-to-video and synchronized audio pass before optional
      image conditioning is exposed.
- [x] Normal generation does not depend on per-layer disk swapping.

## Numerical and quality evidence

- [x] A reproducible official CUDA Golden Reference corpus exists.
- [ ] Primitive, mapped-weight, conditioning, block, sampler, latent, decoder
      and audio parity tests pass their frozen tolerances.
- [ ] Multiple prompts, seeds, durations and resolutions are covered.
- [ ] Programmatic video, temporal and audio-sync metrics are reviewed.
- [ ] Blind side-by-side review is complete.
- [ ] No known major systematic quality regression remains.
- [ ] Published parity language matches the measured evidence.

## Apple Silicon

- [x] M5 Max 128 GB end-to-end generation is verified.
- [ ] Peak unified memory and model-load time are measured.
- [x] Generation time is measured.
- [ ] Metal GPU utilization is measured.
- [x] Repeated reduced-smoke generation does not show unbounded memory growth.
- [ ] Every claimed supported Mac and minimum-memory tier is actually tested.
- [x] Canonical 8,160- and 32,640-token fused-SDPA probes complete without an
      explicit quadratic score allocation.
- [x] Gemma, transformer/upscaler and decoder residency transitions preserve
      the configured macOS memory reserve.
- [ ] Per-stage active/cache/peak MLX memory and elapsed time are published.

## Official user interfaces

- [x] Python `from_pretrained`, generation and `save()` work without a server.
- [x] The CLI produces the same result through the same pipeline.
- [x] ComfyUI is documented as optional, not a core dependency.
- [x] Thin ComfyUI loader, text-to-video and decode/output workflows pass.
- [ ] Image-to-video is exposed only if the core conditioning path passes parity.

## Packaging and documentation

- [ ] GitHub contains the verified source, tests, documentation, license,
      release notes and an immutable tagged release.
- [ ] Hugging Face packaging loads from a clean supported Mac environment.
- [ ] The model card opens with the Apple Silicon/no-server/no-CUDA message.
- [x] Quick Start precedes lengthy technical detail.
- [x] Python, CLI and optional ComfyUI paths are immediately distinguishable.
- [ ] Supported hardware, benchmarks and CUDA-versus-MLX quality results are
      populated with measured data rather than estimates.
- [x] License, attribution, modifications and known limitations are complete.
- [x] All user-facing failures include an error code, cause and recovery action.
- [ ] GitHub and Hugging Face identify the same release version and source
      commit, and link to each other.
- [ ] Public downloads and installation instructions are re-tested after both
      publications complete.
