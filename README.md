# Lara-LTX-2.5-MLX-BF16

> **LTX-2.5 DEV 22B BF16 for Apple Silicon**  
> Native MLX/Metal port for local video generation on Mac.
>
> **No server required. No CUDA required. Runs locally on Apple Silicon.**

Full precision. No quantization. Original quality first.

## Choose how you want to use it

- **Python API** — for developers.
- **CLI** — for terminal users.
- **ComfyUI + Lara custom nodes** — optional GUI for general Mac users.

**ComfyUI is optional.** Python and CLI run the model directly without it.

## Development status

The direct Python pipeline now runs the complete prompt-to-MP4 lifecycle with
the official seven-file BF16 pack: packed Gemma 4 plus the official eight-layer
video/audio prompt connectors, 48 resident AV
transformer blocks, HQ two-stage res_2s sampling, x2 latent refinement,
Diffusion Video VAE, Audio VAE/vocoder/BWE and H.264/AAC muxing. The fixed
320x512/17-frame public-API smoke passes on M5 Max with a measured
40,749,191,990-byte MLX peak and 73.4-second total runtime.

This is an engineering preview, not a same-seed CUDA-reproduction claim. The
four-case CUDA and MLX quality corpora, paired diagnostics and expanded
stochastic trace are complete. Exact CUDA inputs prove the
checkpoint-metadata-driven input path, AdaLN boundary, attention operations,
final transformer boundary and output heads at component tolerance. Deep
block-31 and block-39 traces add 352 passing exact-input Attention comparisons,
with worst NRMSE below `0.00396`, ruling out a discrete deep Attention or
mapping defect. The strict full-trajectory final-latent gate remains open
because small cross-backend BF16 differences accumulate through guidance and
stochastic sampling. All four MLX
outputs passed a non-blind frame-sequence usability audit, but their composition
differs materially from CUDA and independent blind review remains open.

This is not a serving-engine project. The deliverable is one direct Python MLX
pipeline, followed only after core parity by a thin CLI and optional thin
ComfyUI adapter. HTTP servers, daemons, SaaS backends and a duplicate
PyTorch/MPS ComfyUI port are explicitly out of scope.

## Quick Start

Requirements: Apple Silicon, macOS, Python 3.12, 128 GB unified memory,
`ffmpeg`, and accepted access to `Lightricks/LTX-2.5` on Hugging Face. The
primary tested machine is M5 Max 128 GB. The packaged profile defaults to the
measured 512x320/17-frame workload.

```bash
git clone https://github.com/azossy/Lara-LTX-2.5-MLX-BF16.git
cd Lara-LTX-2.5-MLX-BF16
uv sync
export HF_TOKEN="your_read_token"
```

### Python

```python
from lara_ltx import LTXPipeline

pipe = LTXPipeline.from_pretrained("challychoi/Lara-LTX-2.5-MLX-BF16")
video = pipe(prompt="A cinematic aerial shot of Seoul at night.", seed=42)
video.save("output.mp4")
```

Pass `profile=True` to collect text, both sampling stages, latent upscale,
video decode, audio VAE and vocoder elapsed/MLX-memory metrics in
`video.metrics`.

For the already downloaded official pack, pass its directory instead:

```python
pipe = LTXPipeline.from_pretrained("/path/to/LTX-2.5")
```

### CLI

```bash
lara-ltx generate \
  --model challychoi/Lara-LTX-2.5-MLX-BF16 \
  --prompt "A cinematic aerial shot of Seoul at night" \
  --output output.mp4
```

Generation dimensions, seed, frame rate, frame count, steps, cache directory
and a custom TOML profile are optional CLI overrides. Run `lara-ltx generate
--help` for the complete interface. Dimensions must be divisible by 64 and the
frame count must follow `8*k+1`. Before resolving or loading the checkpoint,
the runtime checks physical unified memory and the profile's measured Stage-2
token envelope. Unsupported grids fail with `LARA-RUNTIME-010` instead of
silently swapping until macOS terminates the process.

### ComfyUI

The separate thin adapter is available under `integrations/ComfyUI-LaraLTX`.
Install `lara-ltx` in ComfyUI's Python environment, copy that directory into
`ComfyUI/custom_nodes`, and restart ComfyUI. Its loader, text-to-video and save
nodes invoke the exact same Python pipeline; the tested API workflow is under
the adapter's `examples` directory.

## Supported Mac hardware

The primary acceptance target is M5 Max with 128 GB unified memory. Additional
supported devices and minimum memory will be published only after measured P6
testing. The current engineering preview is measured through 512x512/33 frames
and 640x384/25 frames. A previously undersized decoder budget selected 51,200
stage-5 halo tiles and caused swap runaway; the reviewed 20 GiB activation
budget reduces the maximum case to one stage-5 tile. The packaged resource
policy and 1,280-token Stage-2 ceiling live in `hq.toml`, not API code. A custom
profile may change or disable enforcement, but doing so is not a
supported-hardware claim and never enables implicit quantization or fallback.

## Benchmarks

| Workload | Steps | Result | MLX peak | Total time |
|---|---:|---|---:|---:|
| Public API smoke, 320x512, 17 frames, synchronized audio | Stage 1: 2; Stage 2: 3 | 17-frame H.264 + 48 kHz stereo AAC | 40,749,191,990 B | 73.4 s |
| HQ fixed prompt, 320x512, 17 frames, synchronized audio | Stage 1: 15; Stage 2: 3 | 17-frame H.264 + 48 kHz stereo AAC | 40,749,319,162 B | 154.7 s |
| Same-process repeat (second run), same grid | Stage 1: 2; Stage 2: 3 | Byte-identical MP4; 0 B released-memory growth | 39,877,127,464 B | 46.9 s |
| Quality corpus, 512x512, 33 frames | Stage 1: 15; Stage 2: 3 | H.264 + synchronized AAC | 40,247,895,618 B | 92.0 s |
| Quality corpus, 640x384, 25 frames | Stage 1: 15; Stage 2: 3 | H.264 + synchronized AAC | 40,121,296,374 B | 78.0 s |

Artifacts: `golden/mlx_public_pipeline_smoke_report.json`,
`golden/mlx_hq_fixed_seed_report.json` and
`golden/mlx_repeated_pipeline_report.json` and
`golden/quality/mlx_high_resolution_profile_v7.json`. These measured grids are
not estimates for the canonical 1920x1088 workload.

The opt-in profile's second repeated run measured 13.1 seconds for Gemma text
conditioning, 13.2 seconds for resident-transformer loading, 4.0 seconds for
Stage 1, 0.05 seconds for latent upscale, 14.1 seconds for Stage 2, 1.64 seconds
for video decode, 0.08 seconds for Audio VAE and 0.36 seconds for vocoder/BWE.
The report records active, cache and peak MLX bytes for every phase.

## CUDA vs MLX quality comparison

Checkpoint-backed component, prompt-connector, decode and public MP4 results
are recorded. The expanded CUDA trace has been replayed: the corrected exact
input boundary passes, while the frozen final-latent threshold does not. The
versioned four-case CUDA and MLX corpora are complete at four seeds, four
resolutions and three frame counts. Paired frame, temporal and audio diagnostics
confirm that same-seed CUDA/MLX outputs are different stochastic realizations:
frame cosine ranges from `0.416` to `0.901`, while temporal and raw-audio
cosines remain near zero. A first/middle/last-frame audit found all four MLX
candidates coherent and usable, including the 512x512 fox and 640x384 ocean
cases, but it was not an independent blind review. Candidate usability is
therefore reported separately from CUDA reproduction, which remains failed.

Evidence: `golden/quality/mlx_generation_v7.json`,
`golden/quality/cuda_mlx_quality_corpus_v7.json`,
`golden/quality/visual_review_v7.json` and
`golden/quality/mlx_high_resolution_profile_v7.json`. Exact-input deep
Attention evidence is recorded in
`golden/mlx_attention_internal_block31_v7.json` and
`golden/mlx_attention_internal_block39_v7.json`.

## Canonical references

- Official source: `Lightricks/LTX-2`
- Official model: `Lightricks/LTX-2.5`
- Reference pipeline: `TI2VidTwoStagesHQPipeline`
- Reference precision: BF16

LTX-2.5 was developed by Lightricks. Lara is an independent, unofficial Apple
Silicon port and is not endorsed by or affiliated with Lightricks.

## License and use restrictions

This derivative runtime is distributed under the
[LTX-2.x Community License Agreement](LICENSE.md), including Section 4 and
Attachment A. The upstream acceptable-use policy is incorporated by reference.
Commercial Entities, as defined by that agreement, must obtain the required
paid license before commercial use. See [NOTICE.md](NOTICE.md) for attribution
and a summary of modifications. These files are notices, not legal advice; the
complete agreement controls.

## Development order

1. Produce the official CUDA golden reference.
2. Implement exact streaming BF16 weight loading and mapping.
3. Complete the core MLX transformer/conditioning forward path.
4. Integrate end-to-end video and synchronized-audio generation.
5. Pass tensor, stage and output-quality parity gates.
6. Optimize MLX/Metal only after correctness.
7. Expose the small direct Python API, then the thin CLI.
8. Add the optional thin ComfyUI adapter.
9. Publish the verified, mutually traceable release to GitHub and Hugging Face.

See `docs/ARCHITECTURE.md`, `docs/PORTING_PLAN.md`, `docs/TODO.md`,
`docs/PARITY_PLAN.md`, `docs/RISK_REGISTER.md`, and
`docs/RELEASE_ACCEPTANCE.md`. Target-Mac P1/P2 verification is defined in
`docs/TARGET_MAC_PARITY_RUNBOOK.md`. The controlling scope directive is
`Lara-LTX-2.5-MLX-BF16_Serving_Integration_Scope.md`.
