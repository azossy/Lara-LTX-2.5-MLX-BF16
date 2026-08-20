# Lara-LTX-2.5-MLX-BF16
## Serving / Integration Scope Directive

**Project:** `Lara-LTX-2.5-MLX-BF16`  
**Purpose:** Port LTX-2.5 DEV 22B BF16 to Apple Silicon using MLX/Metal while preserving original BF16 quality and enabling direct local video inference.  
**Status:** Development in progress  
**Audience:** Codex development agent / project contributors

---

# 1. Final Project Definition

`Lara-LTX-2.5-MLX-BF16` is **not a serving-engine project**.

The project goal is:

> **Port LTX-2.5 DEV 22B BF16 to Apple Silicon using MLX/Metal, preserve original BF16 quality as closely as technically possible, and enable direct local video inference on Mac.**

The project must focus on the **model execution path itself**:

- BF16 model loading
- weight mapping
- MLX/Metal inference
- transformer / attention implementation
- conditioning path
- latent/video generation
- VAE / decoder path
- custom Metal kernels only where required
- CUDA Golden Reference parity
- output-quality regression testing
- Apple Silicon memory/performance optimization

Do **not** expand the project into a general-purpose inference server or SaaS platform.

---

# 2. Serving Engine Is Explicitly Out of Scope

The following items are **NOT part of this project**:

- Custom HTTP inference server
- Custom REST API server
- OpenAI-compatible server
- vLLM-style serving engine
- Ollama-style daemon
- Multi-user serving
- Continuous batching server
- Authentication
- User/account management
- Database
- Job-management backend
- Rate limiting
- SaaS infrastructure
- Web dashboard
- Proprietary Web UI
- Cloud serving layer
- Distributed serving
- Kubernetes deployment
- Multi-node inference

If any of these features already exist in experimental code, they must not become a dependency of the core MLX port.

The core project must remain usable without any network server.

---

# 3. Official Execution Architecture

The intended architecture is:

```text
LTX-2.5 DEV 22B BF16
        │
        ▼
Lara-LTX-2.5-MLX-BF16
        │
        ▼
Lara MLX Inference Pipeline
        │
        ├── Model Loader
        ├── Weight Mapping
        ├── Conditioning
        ├── Transformer / Attention
        ├── Latent Generation
        ├── VAE / Decoder
        └── Required MLX / Metal Ops
        │
        ▼
MLX
        │
        ▼
Metal
        │
        ▼
Apple Silicon Unified Memory / GPU
```

The MLX/Metal implementation is the **core product**.

---

# 4. Primary User Interface: Direct Python Inference

The first-class integration path must be a clean Python API.

Target usage should be conceptually similar to:

```python
from lara_ltx import LTXPipeline

pipe = LTXPipeline.from_pretrained("LaraAI/Lara-LTX-2.5-MLX-BF16")

video = pipe(
    prompt="A cinematic aerial shot of a futuristic city at sunset.",
    seed=42,
)

video.save("output.mp4")
```

The exact API may differ according to the final implementation, but the following principles are mandatory:

1. No external serving engine is required.
2. Loading occurs directly inside the Python process.
3. MLX/Metal is used for the actual inference path.
4. The user can instantiate the pipeline, generate video, and save output without starting a server.
5. The public API should remain small and stable.

---

# 5. Optional CLI

A lightweight CLI is allowed and recommended because it makes local testing and Hugging Face usage easier.

Example target:

```bash
lara-ltx generate \
  --model LaraAI/Lara-LTX-2.5-MLX-BF16 \
  --prompt "A cinematic shot of ocean waves at sunrise" \
  --seed 42 \
  --output output.mp4
```

The CLI must be only a thin wrapper around the same Python inference pipeline.

Do not implement a second inference path specifically for the CLI.

```text
CLI
 ↓
Python API
 ↓
Lara MLX Pipeline
 ↓
MLX / Metal
```

---

# 6. Recommended GUI / Workflow Integration: ComfyUI

For normal end users, the recommended GUI environment is **ComfyUI**.

However, ComfyUI must act only as:

- UI
- workflow editor
- parameter input layer
- workflow orchestration layer
- optional existing API/frontend layer

The actual model execution must remain inside the Lara MLX/Metal pipeline.

Target architecture:

```text
ComfyUI
   │
   ▼
ComfyUI-LaraLTX
Thin Custom Nodes
   │
   ▼
Lara MLX Python API
   │
   ▼
Lara-LTX-2.5-MLX-BF16
   │
   ▼
MLX / Metal
   │
   ▼
Apple Silicon
```

---

# 7. Important ComfyUI Rule

Do **not** reimplement the model inside ComfyUI using PyTorch/MPS.

The ComfyUI integration must not become a second port.

Incorrect architecture:

```text
ComfyUI
   ↓
PyTorch / MPS LTX implementation
   ↓
Apple GPU
```

This defeats the purpose of the project.

Correct architecture:

```text
ComfyUI
   ↓
Thin LaraLTX Custom Node
   ↓
Existing Lara MLX Pipeline
   ↓
MLX / Metal
```

There must be **one inference implementation**.

---

# 8. Suggested ComfyUI Custom Nodes

Keep the initial node set intentionally small.

Recommended first version:

### 8.1 Lara LTX Model Loader

Responsibilities:

- Load `Lara-LTX-2.5-MLX-BF16`
- Initialize MLX model state
- Configure memory/runtime options
- Reuse already-loaded model when possible

Suggested output:

```text
LARA_LTX_MODEL
```

---

### 8.2 Lara LTX Text-to-Video

Inputs may include:

- Model
- Prompt
- Negative prompt, if supported by original model
- Width
- Height
- Frame count
- FPS
- Seed
- Steps
- Guidance/CFG values where applicable

Output:

```text
LARA_VIDEO_LATENTS
```

or final video frames depending on final internal architecture.

---

### 8.3 Lara LTX Image-to-Video

Add only when the core MLX port supports the corresponding LTX-2.5 image-conditioning path with verified parity.

Inputs:

- Model
- Input image
- Prompt
- Resolution
- Frames
- Seed
- Generation parameters

---

### 8.4 Lara LTX Decode / Video Output

Responsibilities:

- MLX VAE/decoder invocation
- frame conversion
- output handoff to standard ComfyUI video-saving nodes where practical

Avoid reinventing video encoding if ComfyUI already provides a stable downstream node.

---

# 9. ComfyUI Adapter Must Remain Thin

The custom-node package should ideally contain only:

```text
ComfyUI-LaraLTX/
├── __init__.py
├── nodes.py
├── adapters/
│   └── lara_pipeline.py
├── examples/
│   ├── text_to_video.json
│   └── image_to_video.json
└── README.md
```

The following code must **not** live inside the ComfyUI plugin:

- transformer implementation
- attention implementation
- Metal kernel implementation
- VAE implementation
- tensor conversion engine
- checkpoint parser
- weight-mapping logic
- core scheduler implementation

Those belong in the main `lara_ltx` package.

---

# 10. Recommended Repository Separation

Preferred architecture:

```text
lara-ltx/
│
├── src/
│   └── lara_ltx/
│       ├── models/
│       ├── pipeline/
│       ├── ops/
│       ├── metal/
│       ├── vae/
│       ├── conditioning/
│       ├── utils/
│       └── cli/
│
├── tests/
│   ├── parity/
│   ├── regression/
│   ├── quality/
│   └── performance/
│
├── scripts/
│   ├── convert/
│   ├── tensor_dump/
│   └── benchmark/
│
└── README.md
```

Optional separate integration repository:

```text
ComfyUI-LaraLTX/
```

This keeps the MLX port independent from the GUI.

---

# 11. Golden Reference Policy

The CUDA/PyTorch original must remain the source of truth.

Recommended validation path:

```text
Official LTX-2.5 DEV 22B BF16
PyTorch / CUDA
        │
        ▼
Golden Reference
        │
        ├── intermediate tensor dumps
        ├── latent outputs
        ├── conditioning outputs
        ├── decoder inputs/outputs
        ├── deterministic seeds
        └── final videos
        │
        ▼
Compare
        │
        ▼
Lara-LTX-2.5-MLX-BF16
MLX / Metal
```

A visually plausible video is **not sufficient evidence of correctness**.

Tensor-level and stage-level comparison should be used wherever technically possible.

---

# 12. Quality Requirement

The project goal is not:

> "Make LTX run on a Mac somehow."

The project goal is:

> **Preserve the original LTX-2.5 DEV 22B BF16 behavior and quality while replacing the PyTorch/CUDA execution path with an Apple Silicon MLX/Metal implementation.**

Therefore:

- do not quantize the primary BF16 release merely to make it fit
- do not silently replace unsupported operations with low-quality approximations
- do not remove model capabilities merely to simplify implementation
- do not alter sampling behavior without documenting and validating it
- do not trade major quality loss for benchmark speed

Performance optimization comes **after functional and numerical correctness**.

---

# 13. Release Acceptance Criteria

A public release should not be considered complete until the following are verified.

## Core

- [ ] Official LTX-2.5 DEV 22B BF16 weights load successfully
- [ ] MLX/Metal execution works on Apple Silicon
- [ ] End-to-end video generation works
- [ ] No PyTorch/CUDA dependency in the normal Mac inference path
- [ ] Required custom Metal operations are stable
- [ ] Deterministic seed behavior is tested where applicable

## Quality

- [ ] CUDA Golden Reference test set exists
- [ ] Major intermediate stages have parity tests
- [ ] Output video quality has been manually and programmatically reviewed
- [ ] No known major systematic quality regression remains
- [ ] Multiple prompts/seeds/resolutions are covered

## Apple Silicon

- [ ] M5 Max 128GB tested
- [ ] Peak unified-memory usage measured
- [ ] Model-load time measured
- [ ] Generation time measured
- [ ] Metal GPU utilization checked
- [ ] Memory leaks / repeated-generation growth tested

## User Experience

- [ ] Python direct inference works
- [ ] CLI generation works
- [ ] Hugging Face installation/loading instructions work from a clean environment

## ComfyUI

ComfyUI support may be released together with or immediately after the core MLX model.

- [ ] Thin custom-node adapter only
- [ ] Uses the exact same Lara MLX pipeline
- [ ] Text-to-video workflow tested
- [ ] Image-to-video workflow tested when supported
- [ ] Example workflow JSON supplied
- [ ] No duplicate PyTorch/MPS inference implementation

---

# 14. Development Priority

Codex should prioritize work in this order:

```text
P0  Official CUDA Golden Reference
 ↓
P1  Correct MLX weight loading
 ↓
P2  Core MLX forward path
 ↓
P3  End-to-end video generation
 ↓
P4  Tensor/stage parity
 ↓
P5  BF16 output-quality parity
 ↓
P6  MLX/Metal performance optimization
 ↓
P7  Python public API
 ↓
P8  CLI
 ↓
P9  ComfyUI thin adapter
```

Do not start server development.

Do not allow ComfyUI work to block core-model parity work.

---

# 15. Explicit Codex Instruction

The Codex agent working on this repository should interpret the project scope as follows:

> `Lara-LTX-2.5-MLX-BF16` is an Apple Silicon model-porting and local-inference project, not a model-serving platform. Implement only the MLX/Metal functionality required to load and execute LTX-2.5 DEV 22B BF16 locally while preserving the behavior and quality of the official PyTorch/CUDA implementation. The normal inference path must be callable directly through Python, with a lightweight CLI allowed. Do not build a custom HTTP server, REST API, SaaS backend, multi-user inference service, job server, authentication system, database, or proprietary web UI. For GUI/workflow usage, integrate with ComfyUI through a thin custom-node adapter that calls the existing Lara MLX pipeline directly. Do not create a separate PyTorch/MPS implementation inside ComfyUI.

---

# 16. Final Product Positioning

The technical positioning should remain simple:

> **LTX-2.5 DEV 22B BF16, ported to MLX/Metal for native local inference on Apple Silicon.**

Suggested release message:

> **LTX-2.5 DEV 22B BF16 on Apple Silicon.  
> Native MLX/Metal. Local inference. Original-quality first. ComfyUI ready.**

The value of Lara-LTX is the **Apple Silicon port itself**, not another serving stack.

---

# 17. Final Scope Summary

## Build

```text
LTX-2.5 DEV 22B BF16
        ↓
Lara MLX/Metal Port
        ↓
Python Direct Inference
        ↓
Optional CLI
        ↓
Optional Thin ComfyUI Adapter
```

## Do Not Build

```text
Custom Serving Engine
Custom REST Server
Custom Web UI
Multi-user Platform
SaaS Infrastructure
Ollama/vLLM Replacement
```

---

## Final Decision

**The project shall deliver `Lara-LTX-2.5-MLX-BF16` as a native Apple Silicon MLX/Metal local-inference implementation.**

**Serving-engine development is explicitly excluded.**

**Python direct inference is the primary execution interface.**

**ComfyUI is the recommended end-user GUI/workflow environment, connected through a thin custom-node adapter that directly invokes the Lara MLX pipeline.**

---

# 18. Distribution and User Execution Addendum

The final release provides three official use paths:

1. **Python API** for developers.
2. **CLI** for terminal users.
3. **ComfyUI + Lara custom nodes** as an optional GUI for general Mac users.

ComfyUI is optional. Python and CLI must remain independently usable, and all
three entry points must invoke the same Lara MLX pipeline.

Add the following final development priority after P9:

```text
P10  GitHub + Hugging Face release / user-first README / clean-install verification
```

The Hugging Face model card must lead with:

> **LTX-2.5 DEV 22B BF16 for Apple Silicon**  
> Native MLX/Metal port for local video generation on Mac.  
> **No server required. No CUDA required. Runs locally on Apple Silicon.**

Its user-facing order is:

1. What Lara-LTX is.
2. No server / no CUDA / Apple Silicon.
3. Quick Start.
4. Python.
5. CLI.
6. Optional ComfyUI.
7. Supported Mac hardware.
8. Measured benchmarks.
9. Measured CUDA-versus-MLX quality comparison.
10. Technical details.

Performance, supported-hardware and quality-parity claims may be published only
after their corresponding acceptance evidence exists. The project plan, TODO,
architecture, README draft and release acceptance criteria must all preserve
these rules.

After every release gate passes, publish the same verified release to both
GitHub and Hugging Face. The GitHub tag and Hugging Face revision must identify
the same source commit, compatibility range and release notes. Do not publish a
partially working build as the final release.
