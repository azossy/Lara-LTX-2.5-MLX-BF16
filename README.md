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

P0 CUDA golden-reference work and P1 bounded BF16 checkpoint loading are
complete. Real target-Mac reports now pass for the mapped AV block, Gemma LTX
feature projections, transformer output heads and stage-local LoRA fusion. P2
conditioning/patchification and P3 end-to-end integration remain; no playable
MLX model output or end-to-end parity is claimed yet.

This is not a serving-engine project. The deliverable is one direct Python MLX
pipeline, followed only after core parity by a thin CLI and optional thin
ComfyUI adapter. HTTP servers, daemons, SaaS backends and a duplicate
PyTorch/MPS ComfyUI port are explicitly out of scope.

## Quick Start

The commands below define the target release experience. They are not yet a
working release contract while P1–P6 parity work is in progress. The P0 CUDA
fixed-seed reference, boundary trace and media validation are complete.

### Python

```python
from lara_ltx import LTXPipeline

pipe = LTXPipeline.from_pretrained("LaraAI/Lara-LTX-2.5-MLX-BF16")
video = pipe(prompt="A cinematic aerial shot of Seoul at night.", seed=42)
video.save("output.mp4")
```

### CLI

```bash
lara-ltx generate \
  --model LaraAI/Lara-LTX-2.5-MLX-BF16 \
  --prompt "A cinematic aerial shot of Seoul at night" \
  --output output.mp4
```

### ComfyUI

The release will provide a separate thin `ComfyUI-LaraLTX` adapter that invokes
the exact same Python pipeline. It will not contain another model
implementation. Installation and workflow JSON will be added only after P9 is
verified.

## Supported Mac hardware

The primary acceptance target is M5 Max with 128 GB unified memory. Additional
supported devices and minimum memory will be published only after measured P6
testing.

## Benchmarks

Peak memory, model-load time, generation time and Metal utilization are pending
P6 measurement. No estimated performance figures are claimed.

## CUDA vs MLX quality comparison

Checkpoint-backed component tensor results are recorded; complete conditioning,
stage and output-quality results remain pending P2, P4 and P5.
Plausible-looking output alone will not be reported as parity.

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
