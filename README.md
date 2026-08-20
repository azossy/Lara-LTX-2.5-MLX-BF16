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

This is an engineering preview, not yet a final CUDA-quality-parity release.
The multi-prompt quality corpus, blind review and canonical 1920x1088/121-frame
performance gate remain open; published quality claims stay limited to the
checked-in component and boundary evidence.

This is not a serving-engine project. The deliverable is one direct Python MLX
pipeline, followed only after core parity by a thin CLI and optional thin
ComfyUI adapter. HTTP servers, daemons, SaaS backends and a duplicate
PyTorch/MPS ComfyUI port are explicitly out of scope.

## Quick Start

Requirements: Apple Silicon, macOS, Python 3.12, enough unified memory for the
BF16 workload, `ffmpeg`, and accepted access to `Lightricks/LTX-2.5` on Hugging
Face. The primary tested machine is M5 Max 128 GB.

```bash
git clone https://github.com/LaraAI/Lara-LTX-2.5-MLX-BF16.git
cd Lara-LTX-2.5-MLX-BF16
uv sync
export HF_TOKEN="your_read_token"
```

### Python

```python
from lara_ltx import LTXPipeline

pipe = LTXPipeline.from_pretrained("LaraAI/Lara-LTX-2.5-MLX-BF16")
video = pipe(prompt="A cinematic aerial shot of Seoul at night.", seed=42)
video.save("output.mp4")
```

For the already downloaded official pack, pass its directory instead:

```python
pipe = LTXPipeline.from_pretrained("/path/to/LTX-2.5")
```

### CLI

```bash
lara-ltx generate \
  --model LaraAI/Lara-LTX-2.5-MLX-BF16 \
  --prompt "A cinematic aerial shot of Seoul at night" \
  --output output.mp4
```

Generation dimensions, seed, frame rate, frame count, steps, cache directory
and a custom TOML profile are optional CLI overrides. Run `lara-ltx generate
--help` for the complete interface. Dimensions must be divisible by 64 and the
frame count must follow `8*k+1`.

### ComfyUI

The separate thin adapter is available under `integrations/ComfyUI-LaraLTX`.
Install `lara-ltx` in ComfyUI's Python environment, copy that directory into
`ComfyUI/custom_nodes`, and restart ComfyUI. Its loader, text-to-video and save
nodes invoke the exact same Python pipeline; the tested API workflow is under
the adapter's `examples` directory.

## Supported Mac hardware

The primary acceptance target is M5 Max with 128 GB unified memory. Additional
supported devices and minimum memory will be published only after measured P6
testing.

## Benchmarks

| Workload | Steps | Result | MLX peak | Total time |
|---|---:|---|---:|---:|
| Public API smoke, 320x512, 17 frames, synchronized audio | Stage 1: 2; Stage 2: 3 | 17-frame H.264 + 48 kHz stereo AAC | 40,749,191,990 B | 73.4 s |
| HQ fixed prompt, 320x512, 17 frames, synchronized audio | Stage 1: 15; Stage 2: 3 | 17-frame H.264 + 48 kHz stereo AAC | 40,749,319,162 B | 154.7 s |
| Same-process repeat (second run), same grid | Stage 1: 2; Stage 2: 3 | Byte-identical MP4; 0 B released-memory growth | 40,749,188,278 B | 69.1 s |

Artifacts: `golden/mlx_public_pipeline_smoke_report.json`,
`golden/mlx_hq_fixed_seed_report.json` and
`golden/mlx_repeated_pipeline_report.json`. These 320x512 workloads validate
the lifecycle and are not estimates for the canonical 1920x1088 workload.

## CUDA vs MLX quality comparison

Checkpoint-backed component, prompt-connector, decode and public MP4 results
are recorded. Exact stochastic full-trajectory replay needs one expanded CUDA
trace; multi-prompt perceptual comparison and blind review remain pending P5.
Plausible-looking output alone is not reported as parity.

## Canonical references

- Official source: `Lightricks/LTX-2`
- Official model: `Lightricks/LTX-2.5`
- Reference pipeline: `TI2VidTwoStagesHQPipeline`
- Reference precision: BF16

LTX-2.5 was developed by Lightricks. Lara is an independent, unofficial Apple
Silicon port and is not endorsed by or affiliated with Lightricks.

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
