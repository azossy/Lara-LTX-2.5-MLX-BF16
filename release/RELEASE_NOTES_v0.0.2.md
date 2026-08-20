# Lara-LTX 0.0.2 resource-safety preview

Lara-LTX 0.0.2 is a safety-focused engineering-preview update to the native
MLX/Metal BF16 LTX-2.5 runtime for Apple Silicon. It preserves the same direct
Python API, thin CLI and optional ComfyUI adapter while preventing an
unverified default workload from driving macOS into extreme swap and OOM.

## Changes

- Change the packaged Python, CLI and ComfyUI defaults to the measured
  512x320/17-frame workload.
- Add a schema-versioned, configuration-driven resource policy containing the
  minimum unified-memory capacity, maximum Stage-2 video-token envelope and
  recommended output grid.
- Check the recommended grid before Hugging Face checkpoint resolution and
  check the actual requested grid before model execution.
- Report localized `LARA-RUNTIME-010` errors with the requested token count,
  configured limit, detected capacity, minimum capacity and recommended grid.
- Keep expert overrides explicit through a separately reviewed custom profile;
  no automatic quantization, reduced-precision fallback or silent grid change
  is performed.

## Measured envelope

The supported preview target remains M5 Max with 128 GB unified memory. The
512x320/17-frame public API workload completed with a 40,749,191,990-byte MLX
peak. A 320x512/25-frame quality case also completed and defines the current
640-token Stage-2 ceiling. The 512x512/33-frame case uses 1,280 Stage-2 tokens,
grew swap beyond 70 GB and was terminated by macOS after about 70 minutes; it
is rejected by the packaged profile.

## Scope and numerical status

This remains an engineering preview, not a final CUDA-quality-parity claim.
The resource preflight does not alter weights, arithmetic, schedules or
existing parity thresholds. The strict stochastic final-latent gate and the
remaining quality-corpus work stay open and are documented without weakening
their acceptance criteria.

Users must accept access to `Lightricks/LTX-2.5` and provide a Hugging Face read
token. The runtime remains subject to the LTX-2.x Community License Agreement;
see `LICENSE.md` and `NOTICE.md`.
