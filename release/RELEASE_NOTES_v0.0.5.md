# Lara-LTX 0.0.5 cinematic comparison and evaluator hardening

Lara-LTX 0.0.5 publishes the matched 10-second CUDA/MLX cinematic demo, native
outputs, transparent 4K presentation upscales, and raw objective evaluator
reports under the `LaraAI-Labs` Hugging Face organization.

## Included evidence

- Same prompt, negative prompt, seed, 512×320 source grid, 241 frames, 24 fps
  and 15 steps for CUDA and MLX/Metal.
- Native CUDA and MLX MP4 files plus individual and side-by-side 3840×2160
  Lanczos presentation exports, explicitly labeled as non-native 4K.
- VBench, VideoScore2, Audiobox Aesthetics and LAION CLAP reports with pinned
  evaluator source/model revisions.
- Observed generation metadata: CUDA 74.05 seconds and MLX 2,221.30 seconds.
  These are deployment observations on different systems, not a controlled
  hardware microbenchmark.

## Reliability changes

- VideoScore2 supports numbered descriptive model responses while parsing only
  the final answer after a tagged reasoning block.
- The evaluator uses a configurable pinned `decord==0.6.0` video reader instead
  of the removed `torchvision.io.read_video` API.
- AutoDL sharded downloads are manifest-driven, resumable and verify byte size
  plus SHA-256 before evaluator startup.
- Hugging Face release identity and default examples now resolve to
  `LaraAI-Labs/Lara-LTX-2.5-MLX-BF16`.
