# Lara-LTX 0.0.3 quality-corpus and profiling preview

Lara-LTX 0.0.3 completes the four-case MLX quality corpus, removes the measured
high-resolution decoder cliff and adds phase-level profiling to the native
MLX/Metal BF16 LTX-2.5 runtime for Apple Silicon.

## Changes

- Reproduce the captured CUDA AdaLN boundary exactly by computing RMSNorm in
  FP32 and restoring the input dtype before BF16 modulation.
- Preserve CUDA-aligned FP32 RoPE arithmetic and the existing exact-input
  attention operation gates.
- Increase the versioned decoder activation budget to 20 GiB and the measured
  Stage-2 ceiling to 1,280 tokens. This replaces a pathological 51,200-tile
  decoder plan with one stage-5 tile for the maximum quality-corpus case.
- Complete all four MLX quality generations and paired frame, temporal and
  audio diagnostics at four seeds, four resolutions and three frame counts.
- Add `profile=True` to the Python pipeline. The returned `video.metrics`
  records elapsed time and MLX active/cache/peak memory for text conditioning,
  transformer loading, both sampler stages, latent upscale, video decode,
  Audio VAE and vocoder/BWE.

## Measured results

On the M5 Max 128 GB target, the 512x512/33-frame case completes in 91.98
seconds at a 40,247,895,618-byte MLX peak. The 640x384/25-frame case completes
in 77.97 seconds at 40,121,296,374 bytes. Two same-process 320x512/17-frame
smoke runs complete in 46.85 and 46.94 seconds, peak at 39,877,127,464 bytes,
produce byte-identical MP4 files and show zero released-memory growth.

## Numerical and quality status

This remains an engineering preview, not a same-seed CUDA-reproduction claim.
The combined stochastic replay ends at video/audio latent NRMSE
`0.1916`/`0.1110`, above the frozen final-latent thresholds. All four MLX
outputs pass a non-blind frame-sequence usability audit, but paired metrics and
visual review show materially different same-seed composition. Independent
blind review is not claimed and no tolerance was weakened.

Users must accept access to `Lightricks/LTX-2.5` and provide a Hugging Face read
token. The runtime remains subject to the LTX-2.x Community License Agreement;
see `LICENSE.md` and `NOTICE.md`.
