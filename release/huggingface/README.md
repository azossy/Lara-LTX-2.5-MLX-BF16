---
license: other
license_name: ltx-2.x-community-license
license_link: https://github.com/azossy/Lara-LTX-2.5-MLX-BF16/blob/main/LICENSE.md
base_model: Lightricks/LTX-2.5
library_name: lara-ltx
pipeline_tag: text-to-video
tags:
  - mlx
  - apple-silicon
  - video-generation
  - audio-generation
---

# Lara-LTX-2.5-MLX-BF16

Native MLX/Metal BF16 runtime for local LTX-2.5 video and synchronized-audio
generation on Apple Silicon. No CUDA or server process is required on Mac.

Release `0.0.5` matches the GitHub source tag `v0.0.5`.

> This repository is a release descriptor. The exact official gated BF16 files
> remain in `Lightricks/LTX-2.5`; accepting upstream access is required. The
> Lara loader follows `lara_ltx_model.toml` to that pinned source revision and
> does not publish an unnecessary second copy of the official checkpoint.

## Quick Start

Requirements: Apple Silicon, macOS, Python 3.12, `ffmpeg`, 128 GB unified
memory, and accepted access to the gated upstream model. The packaged profile
defaults to the measured 512x320/17-frame workload and rejects unverified
memory or Stage-2 token grids before checkpoint download or loading.

```bash
git clone https://github.com/azossy/Lara-LTX-2.5-MLX-BF16.git
cd Lara-LTX-2.5-MLX-BF16
uv sync
export HF_TOKEN="your_read_token"
```

```python
from lara_ltx import LTXPipeline

pipe = LTXPipeline.from_pretrained("LaraAI-Labs/Lara-LTX-2.5-MLX-BF16")
video = pipe(prompt="A cinematic aerial shot of Seoul at night.", seed=42)
video.save("output.mp4")
```

The CLI and optional thin ComfyUI adapter invoke this same in-process Python
pipeline. See the GitHub README for measured hardware results, installation,
configuration, known limitations and verified parity evidence.

## CUDA vs MLX: matched 10-second cinematic demo

Both native outputs use the same prompt, negative prompt, seed `25082110`,
512×320 source grid, 241 frames, 24 fps and 15 inference steps. The scene is a
photorealistic close-up of an adult East Asian dancer with wind-driven hair,
facial micro-expressions and eye acting. Cross-backend stochastic generation is
not expected to produce byte-identical or composition-identical videos.

### 4K side-by-side presentation

<video controls playsinline width="100%" src="https://huggingface.co/LaraAI-Labs/Lara-LTX-2.5-MLX-BF16/resolve/main/demo/presentation_4k/cinematic_dancer_closeup_cuda_mlx_side_by_side_4k.mp4"></video>

The side-by-side file has CUDA on the left and MLX/Metal on the right. It carries
separate CUDA and MLX audio tracks. All 4K files are deterministic Lanczos
presentation upscales to 3840×2160, not native 4K model generations, and were
never used as evaluator inputs.

| Native output | Video | 4K presentation |
|---|---|---|
| CUDA | [Play/download](demo/native/cinematic_dancer_closeup_cuda.mp4) | [Play/download](demo/presentation_4k/cinematic_dancer_closeup_cuda_4k.mp4) |
| MLX/Metal | [Play/download](demo/native/cinematic_dancer_closeup_mlx.mp4) | [Play/download](demo/presentation_4k/cinematic_dancer_closeup_mlx_4k.mp4) |

### Measured generation time

| Backend run | Elapsed | Generated frames/s |
|---|---:|---:|
| CUDA | 74.05 s | 3.2547 |
| MLX/Metal | 2,221.30 s | 0.1085 |

CUDA was 30.00× faster in these two observed runs. This is a deployment result,
not a controlled hardware benchmark: the runs used different systems and the
table must not be read as an intrinsic CUDA-versus-Metal hardware ratio.

### Objective evaluation on native outputs

| VBench metric | CUDA | MLX/Metal |
|---|---:|---:|
| Subject consistency | 0.8786 | 0.8799 |
| Background consistency | 0.9314 | 0.9151 |
| Motion smoothness | 0.9866 | 0.9877 |
| Dynamic degree | 1.0000 | 1.0000 |
| Aesthetic quality | 0.5204 | 0.5641 |
| Imaging quality | 0.6663 | 0.6288 |

| VideoScore2 (1–5) | CUDA | MLX/Metal |
|---|---:|---:|
| Visual quality | 4 | 4 |
| Text alignment | 4 | 5 |
| Physical/common-sense consistency | 4 | 4 |

| Audio metric | CUDA | MLX/Metal |
|---|---:|---:|
| Audiobox CE | 3.0175 | 7.6048 |
| Audiobox CU | 5.5600 | 8.3217 |
| Audiobox PC | 2.3645 | 5.6478 |
| Audiobox PQ | 6.1328 | 8.3602 |
| LAION CLAP cosine | 0.2976 | -0.0937 |

The metrics measure different properties and are not combined into one winner.
Inspect the videos directly alongside the raw, revision-pinned reports:
[VideoScore2](evaluation/videoscore2.json),
[CUDA VBench](evaluation/vbench_cuda.json),
[MLX VBench](evaluation/vbench_mlx.json),
[CUDA Audiobox](evaluation/audiobox_cuda.jsonl),
[MLX Audiobox](evaluation/audiobox_mlx.jsonl),
[CUDA CLAP](evaluation/clap_cuda.json), and
[MLX CLAP](evaluation/clap_mlx.json).

## Status

This is an engineering preview, not a final CUDA-quality-parity claim. Exact
CUDA inputs verify the checkpoint-metadata-driven transformer input path and
final transformer/output-head component boundaries. The v6 CUDA trace contains
408 named block-0 Attention boundaries across all six modules and three
guidance passes; payload deduplication stores 118 unique tensors plus 290
aliases. All 176 block-0 MLX exact-input sub-operation comparisons pass the
frozen component gate. The v7 sequence trace replays blocks 24-39 from the
verified block-23 boundary, reproduces all 12 CUDA source outputs exactly, and
adds passing 176-operation MLX reports at both blocks 31 and 39. Their worst
NRMSE values are `0.00396` and `0.00395`.

CUDA-aligned FP32 RMSNorm and RoPE accumulation reproduce the captured AdaLN
boundary exactly. The complete stochastic replay ends at video/audio latent
NRMSE `0.1916`/`0.1110`; the deep exact-input results rule out a discrete
attention implementation or mapping failure, while cross-backend BF16
rounding differences still accumulate. The frozen final-latent gate remains
open without weakening its thresholds.
All four MLX quality-corpus cases and paired CUDA diagnostics are complete. A
non-blind frame-sequence audit accepts candidate usability but confirms that
same-seed composition differs materially, so independent blind parity is not
claimed. The reviewed decoder profile completes 512x512/33 frames in 92.0
seconds at a 40.25 GB MLX peak instead of selecting the old pathological
51,200-tile plan. Resource-policy failures use localized `LARA-RUNTIME-010`
guidance and never silently quantize or select a fallback.

## License

This derivative is subject to the complete LTX-2.x Community License Agreement,
including Section 4 and Attachment A. Recipients must comply with all use
restrictions. Commercial Entities, as defined by the agreement, must obtain the
required paid license before commercial use. The full license and modification
notice are distributed with the source release.
