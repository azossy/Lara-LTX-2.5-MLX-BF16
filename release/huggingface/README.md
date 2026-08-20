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

Release `0.0.3` matches the GitHub source tag `v0.0.3`.

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

pipe = LTXPipeline.from_pretrained("challychoi/Lara-LTX-2.5-MLX-BF16")
video = pipe(prompt="A cinematic aerial shot of Seoul at night.", seed=42)
video.save("output.mp4")
```

The CLI and optional thin ComfyUI adapter invoke this same in-process Python
pipeline. See the GitHub README for measured hardware results, installation,
configuration, known limitations and verified parity evidence.

## Status

This is an engineering preview, not a final CUDA-quality-parity claim. Exact
CUDA inputs verify the checkpoint-metadata-driven transformer input path and
final transformer/output-head component boundaries. The v6 CUDA trace contains
408 named Attention boundaries across all six modules and three guidance
passes; payload deduplication stores 118 unique tensors plus 290 aliases. All
176 MLX exact-input sub-operation comparisons pass the frozen component gate.

CUDA-aligned FP32 RMSNorm and RoPE accumulation reproduce the captured AdaLN
boundary exactly. The complete stochastic replay ends at video/audio latent
NRMSE `0.1916`/`0.1110`; cross-backend BF16 differences still accumulate and
the frozen final-latent gate remains open without weakening its thresholds.
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
