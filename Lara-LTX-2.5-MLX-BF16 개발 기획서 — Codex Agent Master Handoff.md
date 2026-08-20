# Lara-LTX-2.5-MLX-BF16 개발 기획서

**Project Codename:** Lara  
**Official Model Name:** `Lara-LTX-2.5-MLX-BF16`  
**Target Platform:** Apple Silicon / MLX / Metal  
**Primary Target Machine:** MacBook Pro M5 Max / 128GB Unified Memory  
**CUDA Reference Server:** NVIDIA RTX PRO 6000 Blackwell 96GB ×2  
**Priority:** Quality Parity First → Performance Second  
**Project Type:** Full-precision inference runtime port  
**Quantization:** 금지  
**Retraining:** 초기 프로젝트 범위에서 금지  
**Distillation:** 금지

---

# 1. 프로젝트 목적

본 프로젝트의 목적은 Lightricks의 **LTX-2.5 DEV 22B BF16 원본 모델 품질을 최대한 그대로 보존하면서**, NVIDIA CUDA/PyTorch 중심의 공식 inference implementation을 Apple Silicon에 최적화된 **MLX/Metal 네이티브 런타임​**으로 재구현하는 것이다.

최종 결과물의 공식 명칭은:

> **Lara-LTX-2.5-MLX-BF16**

으로 한다.

본 프로젝트는 새로운 영상 생성 모델을 학습하는 프로젝트가 아니다.

다음 작업을 수행한다.

```text
Official LTX-2.5 DEV 22B BF16
            │
            │ PyTorch / CUDA
            ▼
      Golden Reference
            │
            │ Numerical Parity
            ▼
 Lara MLX Implementation
            │
            │ MLX / Metal
            ▼
 Apple Silicon Native Runtime
            │
            ▼
      M5 Max 128GB
```

가중치 자체의 의미를 변경하지 않는다.

초기 단계에서는 다음을 사용하지 않는다.

- INT4
- INT8
- Q4
- Q8
- FP8
- model pruning
- distillation
- low-rank approximation
- speculative approximation
- 품질 손실이 발생할 수 있는 임의의 fast-math

**BF16 원본 품질 보존이 최우선이다.**

---

# 2. 핵심 프로젝트 철학

프로젝트의 절대적인 개발 원칙은 다음과 같다.

> **Quality Parity First. Performance Second.**

빠르게 동작하는 Apple Silicon 버전을 만드는 것이 첫 번째 목표가 아니다.

첫 번째 목표는:

> **공식 CUDA/PyTorch LTX-2.5 DEV BF16과 Lara MLX BF16 implementation 사이에서 실질적인 출력 품질 동등성을 달성하는 것**

이다.

속도 최적화는 이 목표가 달성된 이후 진행한다.

따라서 Codex는 성능 향상을 이유로 정확도를 희생하는 최적화를 임의로 적용해서는 안 된다.

---

# 3. Upstream 기준

항상 **Lightricks 공식 LTX-2 repository와 LTX-2.5 공식 checkpoint를 canonical reference**로 사용한다.

현재 공식 repository에서 LTX-2.5는 component별 safetensors 구조이며 다음 구성이 존재한다.

### Transformer

`ltx-2.5-22b-dev-transformer-bf16.safetensors`

이 모델은 **full model**이며 guided two-stage pipeline에 사용된다.

Distilled Transformer는 별도 모델이므로 Lara의 기준 모델로 사용하지 않는다.

### Text Encoder

`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`

LTX용으로 fine-tuning된 Gemma 4 12B text encoder와 projection을 사용한다.

일반 Gemma checkpoint로 대체하지 않는다.

### Video VAE

`ltx-2.5-video-vae-bf16.safetensors`

최고품질 모드에서는 convolution VAE가 아니라 diffusion decoder VAE를 우선한다.

공식 LTX 문서에서도 diffusion decoder가 더 많은 VRAM과 decode 시간을 요구하지만 품질이 향상되는 경로라고 명시한다.

### Audio VAE

`ltx-2.5-audio-vae-bf16.safetensors`

### Spatial Upscaler

`ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`

### Distilled LoRA

공식 Two-Stage pipeline이 요구하는 경우 공식 LTX-2.5용 BF16 distilled LoRA를 그대로 지원한다.

이것은 Transformer를 Distilled 모델로 교체한다는 의미가 아니다.

공식 full DEV 모델을 기반으로 upstream pipeline의 동작을 그대로 재현해야 한다.

공식 LTX repository는 `TI2VidTwoStagesHQPipeline`을 two-stage generation + `res_2s` second-order sampler 기반의 더 높은 품질 pipeline으로 구분하고 있다.

---

# 4. 최종 목표 Pipeline

Lara가 최종적으로 우선 지원해야 할 최고품질 pipeline은:

> **LTX-2.5 DEV 22B BF16 + TI2VidTwoStagesHQ equivalent**

이다.

개념 구조:

```text
Prompt / Optional Image
        │
        ▼
Gemma 4 12B LTX BF16
        │
        ▼
Text Embeddings
        │
        ▼
Initial Noise / Latents
        │
        ▼
LTX-2.5 DEV 22B BF16
        │
        │ CFG / Guidance
        │ res_2s
        ▼
Stage 1 Latents
        │
        ▼
BF16 Spatial Upscaler
        │
        ▼
Stage 2 Refinement
        │
        ▼
Video Latents + Audio Latents
        │
        ├───────────────┐
        ▼               ▼
Video VAE BF16      Audio VAE BF16
        │               │
        ▼               ▼
Frames             Waveform
        │               │
        └──────┬────────┘
               ▼
         Final MP4
```

---

# 5. 하드웨어 환경

## 5.1 CUDA Golden Reference Server

DLauto에서 다음 환경을 사용한다.

### GPU

**RTX PRO 6000 Blackwell 96GB ×2**

NVIDIA 공식 사양상 RTX PRO 6000 Blackwell Server Edition은 GPU당 96GB GDDR7 메모리와 BF16 Tensor Core 연산을 제공한다.

### 권장 시스템

- GPU: RTX PRO 6000 Blackwell 96GB ×2
- CPU: 최소 32 vCPU
- 권장: 48 vCPU 이상
- RAM: 최소 256GB
- 권장: **512GB**
- SSD: 최소 1TB NVMe
- 권장: **2TB NVMe**
- OS: Ubuntu Linux
- CUDA/PyTorch: **설치 시점의 LTX 공식 repository 요구 버전을 우선**
- Python: upstream 요구사항 준수
- Git
- uv
- Hugging Face CLI

현재 공식 LTX PyTorch API 문서는 Python 3.12+, CUDA 12.7+, PyTorch 2.7 계열을 기준으로 안내하고 있지만, 버전은 프로젝트 시작 시 upstream repository를 다시 확인하여 lockfile에 고정한다.

---

# 6. GPU 2장의 역할

초기에는 두 GPU를 하나의 192GB GPU처럼 취급하지 않는다.

### GPU 0

**Pristine Golden Reference GPU**

사용 목적:

- 공식 LTX repository
- 공식 checkpoint
- 코드 변경 최소화
- Golden output 생성
- Golden tensor 생성
- 기준 benchmark

환경을 최대한 깨끗하게 유지한다.

### GPU 1

**Instrumentation / Development GPU**

사용 목적:

- forward hook
- intermediate tensor dump
- profiler
- modified reference code
- numerical experiments
- layer-by-layer validation
- regression generation
- batch benchmark

GPU 0의 canonical environment와 GPU 1의 experimental environment를 분리한다.

---

# 7. Apple Target Hardware

Primary target:

> **MacBook Pro M5 Max / 128GB Unified Memory**

Apple MLX는 Apple Silicon 전용 ML framework이며 CPU와 GPU가 동일한 unified memory의 array를 직접 접근하는 구조를 제공한다.

Lara는 이 unified-memory architecture를 적극적으로 활용해야 한다.

CUDA식의 불필요한 CPU↔GPU copy 모델을 그대로 흉내 내지 않는다.

---

# 8. 구현 언어 및 Runtime

초기 implementation:

- Python
- MLX
- NumPy
- safetensors
- ffmpeg

최적화 단계:

- MLX built-in fast operations
- `mx.compile()`
- `mx.fast`
- custom Metal kernel
- 필요 시 C++ MLX extension

MLX는 계산 그래프 compilation 및 operator fusion을 지원하고, Python/C++ API를 통해 custom Metal kernel 구현도 공식 지원한다.

---

# 9. Repository 구조

Codex는 신규 repository를 다음과 같은 구조로 설계한다.

```text
Lara-LTX-2.5-MLX-BF16/
│
├── README.md
├── LICENSE-LTX.md
├── NOTICE.md
├── pyproject.toml
├── uv.lock
│
├── configs/
│   ├── cuda_reference.yaml
│   ├── mlx_reference.yaml
│   ├── mlx_hq.yaml
│   └── parity.yaml
│
├── packages/
│   │
│   ├── lara_ltx_core/
│   │   ├── transformer/
│   │   ├── attention/
│   │   ├── rope/
│   │   ├── norms/
│   │   ├── embeddings/
│   │   ├── conditioning/
│   │   ├── schedulers/
│   │   ├── samplers/
│   │   ├── guidance/
│   │   ├── video_vae/
│   │   ├── audio_vae/
│   │   └── upscaler/
│   │
│   └── lara_ltx_pipelines/
│       ├── t2v_hq.py
│       ├── i2v_hq.py
│       ├── common.py
│       └── output.py
│
├── tools/
│   │
│   ├── convert/
│   │   ├── inspect_checkpoint.py
│   │   ├── convert_bf16_to_mlx.py
│   │   └── verify_conversion.py
│   │
│   ├── cuda_reference/
│   │   ├── generate.py
│   │   ├── dump_embeddings.py
│   │   ├── dump_latents.py
│   │   ├── dump_transformer.py
│   │   ├── dump_vae.py
│   │   └── manifest.py
│   │
│   └── parity/
│       ├── compare_tensor.py
│       ├── compare_layers.py
│       ├── compare_latents.py
│       ├── compare_frames.py
│       └── report.py
│
├── golden/
│   ├── manifests/
│   ├── prompts/
│   └── README.md
│
├── benchmarks/
│   ├── quality/
│   ├── performance/
│   ├── memory/
│   └── reports/
│
├── tests/
│   ├── unit/
│   ├── parity/
│   ├── integration/
│   └── regression/
│
└── docs/
    ├── ARCHITECTURE.md
    ├── PORTING.md
    ├── PARITY.md
    ├── PERFORMANCE.md
    ├── MODEL_CARD.md
    └── RELEASE.md
```

폴더 및 package 이름은 실제 upstream 구조를 확인한 뒤 변경할 수 있으나 책임 영역은 유지한다.

---

# 10. Phase 0 — Upstream Freeze

Codex가 가장 먼저 수행할 작업.

### 10.1 공식 repository clone

공식 Lightricks LTX-2 repository를 clone한다.

### 10.2 Revision 고정

기준 commit SHA를 기록한다.

```text
UPSTREAM_LTX_COMMIT=
UPSTREAM_MLX_VERSION=
PYTORCH_VERSION=
CUDA_VERSION=
```

을 manifest에 저장한다.

### 10.3 checkpoint hash 기록

모든 모델 파일의:

- filename
- file size
- SHA256
- dtype
- tensor count

를 기록한다.

### Phase 0 완료 조건

`golden/manifests/upstream.json`

파일 하나만 보면 정확한 reference environment를 재현할 수 있어야 한다.

---

# 11. Phase 1 — CUDA Golden Reference 구축

절대로 MLX 개발부터 시작하지 않는다.

먼저 GPU 0에서 official implementation을 정상 실행한다.

### 목표

LTX-2.5 DEV BF16 + highest-quality supported two-stage pipeline을 정상 생성한다.

초기 기준 영상:

- 512×320 정도의 작은 test
- 이후 704×448
- 이후 1024×576 또는 upstream이 권장하는 target
- 24fps
- short clip부터 시작

최종 품질 benchmark는 이후 확대한다.

### 중요

reference 생성에 FP8을 사용하지 않는다.

CPU/disk offloading 역시 처음에는 가능한 최소화한다.

---

# 12. Golden Test Set

최소 20개의 고정 prompt를 작성한다.

다음 영역을 포함한다.

### Human

- 얼굴 close-up
- 눈 깜빡임
- 머리 움직임
- 말하는 인물
- 손 움직임

### Motion

- 걷기
- 달리기
- 빠른 회전
- 복수 객체 motion
- cloth/hair motion

### Camera

- static
- slow pan
- dolly
- tracking
- handheld-style movement

### Physical scene

- 물
- 연기
- 불
- 바람
- reflection

### Lighting

- daylight
- indoor
- low-light
- backlight
- cinematic contrast

### Audio

- speech
- ambient
- synchronized physical sound
- silence / pause

각 test는 다음을 고정한다.

```text
test_id
prompt
seed
height
width
num_frames
fps
pipeline
sampler
step count
CFG
STG
initial noise
model revision
```

---

# 13. RNG 문제 해결

CUDA와 Metal의 RNG가 동일하다고 가정하지 않는다.

비교의 정확성을 위해:

> **Golden CUDA environment에서 initial noise/latent를 생성한 후 파일로 저장**

한다.

이 동일한 initial tensor를 CUDA와 MLX 양쪽에 넣는다.

따라서 다음을 구분한다.

```text
RNG parity
```

와

```text
model computation parity
```

를 혼동하지 않는다.

---

# 14. Tensor Dump System

CUDA reference pipeline에서 주요 tensor를 저장할 수 있도록 instrumentation layer를 작성한다.

모든 tensor를 무조건 저장해서 disk를 낭비하지 않는다.

Config 기반 selective dump를 지원한다.

예:

```yaml
dump:
  text_embedding: true
  initial_latent: true
  transformer:
    blocks:
      - 0
      - 1
      - 5
      - 10
      - 20
      - last
  stage1_final: true
  upscaled_latent: true
  stage2_final: true
  video_vae_output: true
  audio_vae_output: true
```

tensor와 함께 반드시 metadata를 기록한다.

- shape
- dtype
- min
- max
- mean
- std
- finite count
- checksum

---

# 15. Phase 2 — Weight Converter

`Lara-LTX-2.5-MLX-BF16`은 양자화 모델이 아니다.

따라서 conversion의 목표는:

> **BF16 numerical values를 변경하지 않고 MLX가 읽을 수 있는 layout으로 매핑하는 것**

이다.

필요한 작업:

1. original safetensors 읽기
2. key mapping
3. tensor shape mapping
4. transpose가 필요한 layer 확인
5. BF16 preservation
6. MLX-compatible serialization
7. manifest 생성
8. round-trip validation

### Conversion validation

가능한 모든 tensor에 대해:

```text
original_dtype == BF16
converted_dtype == BF16
shape equivalence
numerical equivalence
```

검증한다.

layout 변경 때문에 transpose가 필요한 tensor는 논리적으로 동일한 값인지 검증한다.

---

# 16. 기존 MLX 프로젝트 활용 원칙

Codex는 기존 Apple Silicon LTX port를 반드시 조사한다.

현재 공개된 MLX LTX implementation 가운데는 pure MLX로 two-stage와 HQ `res_2s`, CFG/STG, BF16 등을 구현한 프로젝트가 이미 존재한다. 다른 MLX Video 프로젝트 역시 dev/two-stage/HQ 경로를 구현하고 있다.

그러나 원칙은:

> **Reference, not blind dependency.**

기존 코드를 그대로 복사해 조립하는 것이 목적이 아니다.

다음 용도로 사용한다.

- architecture 이해
- weight mapping 참고
- unsupported MLX operation 해결법 참고
- Metal memory issue 파악
- sampler 구현 비교
- upstream parity 분석

Lara 구현의 기준은 항상 **Lightricks 공식 CUDA implementation**이다.

커뮤니티 implementation과 공식 implementation이 다르면 공식 implementation을 우선한다.

---

# 17. Phase 3 — MLX Transformer Port

가장 먼저 minimal Transformer forward를 구현한다.

초기 목표:

```text
CUDA Transformer input
        =
MLX Transformer input
```

을 사용했을 때 output numerical parity를 얻는 것.

다음 순서로 구현한다.

1. embeddings
2. normalization
3. QKV projections
4. RoPE / positional operations
5. attention
6. feed-forward
7. modality interaction
8. conditioning
9. output projection

각 block을 독립적으로 test할 수 있어야 한다.

---

# 18. Parity Metrics

단순히 `allclose()`만 사용하지 않는다.

각 tensor 비교에 다음 metric을 기록한다.

- max absolute error
- mean absolute error
- relative error
- RMSE
- cosine similarity
- NaN count
- Inf count

예:

```json
{
  "tensor": "transformer.block.12.output",
  "dtype": "bfloat16",
  "cosine_similarity": 0.99999,
  "max_abs_error": 0.0039,
  "mean_abs_error": 0.00012
}
```

Tolerance는 component별로 empirical하게 정의하되 처음부터 느슨하게 잡지 않는다.

---

# 19. Phase 4 — Text Encoder

Gemma 4 LTX text encoder도 local BF16로 구현한다.

목표:

> 동일 prompt → CUDA embedding과 MLX embedding이 충분히 동등

해야 한다.

Text encoder를 외부 API에 의존하지 않는다.

Lara 최종판은 **완전 로컬 실행**을 기본으로 한다.

---

# 20. Phase 5 — Scheduler / Guidance / Sampler

다음 component를 CUDA와 동일하게 구현한다.

- scheduler
- noise schedule
- CFG
- STG if required
- res_2s sampler
- stage transition
- conditioning

**sampler의 미세한 수치 차이는 전체 영상 결과를 크게 변화시킬 수 있으므로 매우 중요하다.**

이 단계에서 convenience approximation을 사용하지 않는다.

---

# 21. Phase 6 — Spatial Upscaler

BF16 spatial latent upscaler를 포팅한다.

검증:

```text
same latent
   ↓
CUDA Upscaler
vs
MLX Upscaler
```

output tensor parity를 측정한다.

---

# 22. Phase 7 — Video VAE

최고품질 버전의 핵심이다.

우선:

`ltx-2.5-video-vae-bf16`

Diffusion VAE를 목표로 한다.

공식 LTX 구현에서 Linux/CUDA에서는 diffusion video VAE용 NATTEN acceleration을 사용할 수 있지만 macOS에서는 다른 backend로 fallback한다. 따라서 Lara에서는 해당 계산을 MLX/Metal 방식으로 별도 최적화해야 한다.

### 개발 순서

1. MLX naive/reference implementation
2. CUDA parity
3. memory profiling
4. performance profiling
5. optimization

**1~2가 완료되기 전 3~5로 넘어가지 않는다.**

---

# 23. Phase 8 — Audio

Video parity가 확보된 이후:

- Audio VAE
- audio latent generation
- synchronized audio
- final mux

를 구현한다.

초기 MVP에서 audio를 생략하고 영상을 먼저 검증하는 것은 허용한다.

그러나 최종 `Lara-LTX-2.5-MLX-BF16` release에서는 LTX-2.5의 audio-video capability를 유지하는 것을 목표로 한다.

---

# 24. Phase 9 — Full HQ Pipeline

모든 component를 연결한다.

목표:

```text
LTX Official CUDA
DEV BF16
TwoStagesHQ
```

vs.

```text
Lara
DEV BF16
MLX
TwoStagesHQ equivalent
```

를 동일 조건에서 실행한다.

비교:

- output resolution
- frame count
- audio duration
- visual structure
- motion
- temporal consistency
- prompt adherence
- face stability
- fine details
- audio sync

---

# 25. 품질 검증

최종 output quality는 단순 PSNR 하나로 판단하지 않는다.

### Numerical

- intermediate tensor parity
- final latent parity

### Image

- frame-level pixel comparison
- LPIPS 계열 perceptual metric
- SSIM
- selected-frame visual comparison

### Temporal

- optical-flow consistency
- temporal perceptual difference
- motion trajectory comparison

### Human Review

Golden Test Set에 대해 side-by-side blind comparison을 한다.

최종 성공 기준:

> 평균 사용자가 어느 쪽이 CUDA이고 어느 쪽이 MLX인지 품질 차이만으로 안정적으로 구분하기 어려운 수준.

단, 완전히 bit-identical한 결과가 필수 목표는 아니다.

CUDA와 Metal의 kernel/rounding 차이로 bitwise identity는 현실적으로 필요하지 않다.

**Perceptual parity + strong numerical parity**를 목표로 한다.

---

# 26. 성능 최적화는 그 이후

Parity milestone이 통과하기 전에는 aggressive optimization을 금지한다.

Parity 완료 이후 다음 순서로 최적화한다.

### Level 1

MLX native operation 선택 개선

### Level 2

memory lifetime 최적화

### Level 3

lazy graph evaluation 위치 조정

### Level 4

`mx.compile()`

MLX의 `compile()`은 graph 최적화와 operator fusion을 통해 runtime과 memory 사용량을 개선할 수 있다.

### Level 5

MLX built-in fast op

가능하면 먼저 MLX가 이미 제공하는:

- scaled dot product attention
- RMS norm
- layer norm
- RoPE

등의 fast implementation을 조사한다.

### Level 6

Custom Metal Kernel

필요한 병목에 대해서만 custom Metal kernel을 작성한다.

MLX는 Python과 C++에서 custom Metal kernel을 공식 지원한다.

---

# 27. Metal Math Mode 정책

초기 Metal custom kernel은:

```text
math_mode = safe
```

를 기본값으로 사용한다.

MLX custom Metal kernel의 기본 safe math mode는 IEEE 특수값 처리를 보존한다. relaxed/fast mode도 지원하지만 처음부터 사용하지 않는다.

최적화 단계에서:

```text
safe
 ↓
relaxed
 ↓
fast
```

순으로 실험한다.

각 변경마다 Golden Test 전체 regression을 돌린다.

품질 또는 parity degradation이 발생하면 reject한다.

---

# 28. Memory 전략

M5 Max 128GB에서는 첫 릴리즈에서 low-memory operation을 지나치게 최적화하지 않는다.

우선 목표:

> **128GB급 Apple Silicon에서 최고품질 BF16 실행**

이다.

32GB Mac 지원을 위해 architecture를 망가뜨리지 않는다.

향후 별도 variant로:

```text
Lara-LTX-2.5-MLX-Q8
Lara-LTX-2.5-MLX-Q4
```

등을 만들 수 있으나 본 프로젝트 범위가 아니다.

---

# 29. Target Performance Metrics

v1.0은 속도 자체를 출시 gate로 삼지 않는다.

그러나 다음을 반드시 기록한다.

- model load time
- text encoding time
- stage 1 generation time
- upscaler time
- stage 2 time
- video VAE decode time
- audio decode time
- mux time
- total generation time
- peak unified memory
- GPU utilization
- CPU utilization
- thermal state

Test matrix:

```text
512×320
704×448
1024×576
1280×720
```

영상 길이:

```text
~2 sec
~5 sec
~10 sec
```

고해상도/장시간은 안정화 후 확대한다.

---

# 30. CLI

최종 사용 경험은 최대한 간단해야 한다.

예:

```bash
lara-ltx generate \
  --prompt "A cinematic shot..." \
  --pipeline hq \
  --height 576 \
  --width 1024 \
  --frames 121 \
  --seed 42 \
  --output output.mp4
```

기본값:

```text
dtype = BF16
quantization = none
pipeline = HQ
device = Apple GPU
```

로 한다.

사용자가 실수로 INT8/Q4 등을 켤 수 있는 숨겨진 자동 최적화를 넣지 않는다.

---

# 31. Reproducibility

각 생성 결과와 함께 JSON sidecar를 저장할 수 있도록 한다.

예:

```text
output.mp4
output.json
```

JSON에는:

- model revision
- Lara version
- MLX version
- macOS version
- chip
- memory
- prompt
- seed
- dimensions
- frame count
- sampler
- steps
- CFG
- STG
- model hashes
- elapsed time

를 기록한다.

---

# 32. Test Policy

모든 핵심 component는:

```text
Unit Test
    +
CUDA Parity Test
    +
Integration Test
    +
Regression Test
```

가 존재해야 한다.

새 성능 최적화 PR은:

> Golden regression test 통과 없이는 merge 금지

원칙으로 한다.

---

# 33. 실패 처리

OOM, Metal command failure, NaN, corrupted output 발생 시 조용히 fallback해서 품질을 변경하지 않는다.

예:

```text
BF16 OOM
→ 자동 Q8 전환
```

같은 동작은 **금지**한다.

대신 명확한 오류를 표시한다.

예:

```text
Generation aborted:
BF16 memory requirement exceeded.
No automatic quantization was applied because Lara is running in Quality-Preserve mode.
```

---

# 34. Logging / Profiling

다음 logging level을 지원한다.

```text
ERROR
WARN
INFO
DEBUG
TRACE
PARITY
PROFILE
```

PROFILE 모드에서는 각 pipeline stage의:

- execution time
- memory
- tensor shape
- graph evaluation
- Metal operation

정보를 기록한다.

---

# 35. Hugging Face 공개

최종 모델 repository 이름:

> **Lara-LTX-2.5-MLX-BF16**

Model Card 상단에는 명확하게 다음 사실을 표시한다.

- LTX-2.5는 Lightricks가 개발
- Lara는 비공식 Apple Silicon port
- Lightricks와 공식 제휴/승인 관계가 아님
- upstream model/license를 명시
- Lara에서 변경된 파일과 runtime 변경 사항을 설명
- BF16 full precision
- no quantization
- target Apple Silicon
- M5 Max 128GB에서 검증된 benchmark 공개

---

# 36. 라이선스

이 항목은 반드시 지켜야 한다.

현재 LTX-2.5는 **LTX-2.x Community License Agreement**를 적용받으며, 해당 라이선스는 LTX 기반 derivative architectures와 수정된 accompanying code 등도 derivative 범위에 포함한다. 배포 시 원 라이선스 제공, 변경 파일 표시, attribution 유지 등 별도 조건이 존재한다. 또한 Lightricks 상표나 공식 endorsement를 암시할 권리를 제공하지 않는다.

따라서 Codex는:

**임의로 전체 프로젝트를 MIT/Apache 등으로 선언하지 않는다.**

공개 전에:

1. upstream license 전체 보존
2. NOTICE
3. modification notices
4. attribution
5. model-card disclaimer
6. distribution requirements

를 확인한다.

법률적 최종 판단이 필요한 부분은 코드로 추측하지 말고 `LEGAL_REVIEW_REQUIRED.md`에 명확하게 표시한다.

---

# 37. Git 운영

Branch:

```text
main
develop
feature/*
perf/*
parity/*
```

주요 milestone마다 tag:

```text
v0.1-cuda-reference
v0.2-weight-conversion
v0.3-transformer-parity
v0.4-vae-parity
v0.5-hq-pipeline
v0.6-metal-optimized
v1.0
```

---

# 38. Milestone

## M0 — Environment

- RTX PRO 6000 ×2 정상 인식
- 공식 LTX 실행
- M5 Max MLX 실행
- 모든 version freeze

## M1 — Golden Reference

- DEV BF16 HQ generation 성공
- Golden Test Set 구축
- tensor dump 완료

## M2 — Weight Conversion

- 모든 BF16 weights 변환
- dtype/shape/hash manifest
- conversion verification

## M3 — Transformer

- MLX Transformer forward
- block-by-block parity

## M4 — Text Encoder

- Gemma embedding parity

## M5 — Sampler / Guidance

- scheduler
- CFG
- res_2s
- stage logic parity

## M6 — VAE / Upscaler

- upscaler parity
- video decode parity

## M7 — Audio

- audio generation/decode

## M8 — Full HQ

- end-to-end generation
- CUDA vs MLX quality parity

## M9 — Performance

- profiling
- mx.compile
- MLX fast ops
- custom Metal kernels

## M10 — Release

- documentation
- benchmark
- model card
- license
- Hugging Face release

---

# 39. v1.0 Definition of Done

다음 조건을 모두 만족해야 `v1.0`으로 선언한다.

### Functional

- M5 Max 128GB에서 완전 로컬 실행
- T2V HQ 정상 동작
- I2V HQ 정상 동작 또는 명확하게 다음 milestone로 문서화
- video + audio 최종 출력
- BF16
- no quantization

### Quality

- CUDA Golden Test Set와 numerical parity 검사 통과
- perceptual quality review 통과
- obvious quality regression 없음

### Reliability

- 반복 generation 안정
- memory leak 없음
- NaN propagation 없음
- deterministic input support

### Performance

최소한 실사용 가능한 generation 완료.

단, 성능 때문에 품질을 희생하지 않는다.

### Documentation

- 설치
- 모델 다운로드
- CUDA reference
- Apple execution
- benchmark
- troubleshooting
- architecture
- license

모두 문서화한다.

---

# 40. Codex 행동 원칙

Codex는 다음 규칙을 항상 따른다.

### 1.

코드를 작성하기 전에 upstream LTX implementation을 먼저 읽는다.

### 2.

추측으로 architecture를 구현하지 않는다.

### 3.

CUDA implementation을 canonical specification으로 간주한다.

### 4.

기존 MLX port는 참고 자료로 활용하되 무조건 신뢰하지 않는다.

### 5.

최적화 전에 parity test를 작성한다.

### 6.

성능 향상 후 반드시 regression test를 실행한다.

### 7.

BF16을 임의로 FP16/FP8/INT8로 변환하지 않는다.

### 8.

사용자가 명시하지 않는 한 모델을 fine-tune하지 않는다.

### 9.

실패를 숨기기 위한 silent fallback을 만들지 않는다.

### 10.

작업 과정에서 발견한 upstream bug, Mac-specific limitation, numerical discrepancy는 모두 문서화한다.

---

# 41. 우선순위

개발 우선순위는 절대 다음 순서를 바꾸지 않는다.

```text
1. Correctness
       ↓
2. Numerical Parity
       ↓
3. Perceptual Quality
       ↓
4. Stability
       ↓
5. Memory Efficiency
       ↓
6. Performance
       ↓
7. Convenience
```

속도 때문에 1~4를 희생하지 않는다.

---

# 42. Codex 첫 실행 작업

Codex는 프로젝트 시작 시 바로 코딩부터 하지 말고 다음 작업부터 수행한다.

## Task 1

Lightricks/LTX-2 최신 repository 조사.

다음 파일과 package를 분석한다.

- ltx-core
- ltx-pipelines
- TI2VidTwoStagesHQPipeline
- Transformer architecture
- Gemma connector
- video VAE
- audio VAE
- upscaler
- sampler
- guidance
- checkpoint loader

## Task 2

Apple MLX 최신 API 조사.

특히:

- BF16
- SDPA
- RoPE
- normalization
- compile
- streams
- memory management
- Metal kernel
- safetensors

지원 여부를 확인한다.

## Task 3

기존 LTX MLX port를 조사해:

```text
UPSTREAM_CUDA_VS_EXISTING_MLX.md
```

를 작성한다.

각 component에 대해:

```text
Official CUDA implementation
Existing MLX implementation
Difference
Known limitation
Lara implementation decision
```

을 정리한다.

## Task 4

다음 문서를 만든다.

```text
docs/ARCHITECTURE.md
docs/PORTING_PLAN.md
docs/PARITY_PLAN.md
docs/RISK_REGISTER.md
```

## Task 5

그 이후에 repository skeleton과 CUDA Golden Reference tooling을 구현한다.

---

# 43. 첫 번째 실제 성공 기준

프로젝트의 첫 번째 큰 성공은:

> **Mac에서 영상 하나가 생성되는 것**

이 아니다.

다음이 성공 기준이다.

```text
Same Input Tensor
       │
       ├── Official CUDA Transformer
       │
       └── Lara MLX Transformer
                │
                ▼
        Strong Numerical Parity
```

이 지점을 확보하면 나머지는 엔지니어링 문제로 바뀐다.

---

# 44. 두 번째 성공 기준

다음 단계는:

```text
Official CUDA
LTX-2.5 DEV BF16
TwoStagesHQ
        │
        ▼
    Golden Video
```

와

```text
Lara MLX
LTX-2.5 DEV BF16
TwoStagesHQ equivalent
        │
        ▼
     Lara Video
```

가 사실상 동일한 품질을 갖는 것이다.

---

# 45. 최종 비전

`Lara-LTX-2.5-MLX-BF16`은 단순한 “Mac에서도 돌아가는 LTX”가 되어서는 안 된다.

목표는:

> **LTX-2.5 DEV 22B의 full-precision BF16 품질을 Apple Silicon에서 최대한 원본에 가깝게 구현하고, Apple의 MLX/Metal/Unified Memory architecture를 활용하여 M5 Max급 시스템에서 실용적으로 사용할 수 있게 만든 최고품질 Apple-native LTX runtime**

이다.

최종적으로 Hugging Face에서 사용자에게 전달해야 할 메시지도 명확해야 한다.

> **Full Precision.  
> No Quantization.  
> Apple Silicon Native.  
> Quality Parity First.**

---

# 46. Codex에 대한 최종 명령

이 프로젝트를 단기 데모 또는 proof-of-concept 수준으로 구현하지 않는다.

설계, 테스트, numerical validation, reproducibility, logging, benchmark, licensing, documentation을 처음부터 포함한다.

모든 최적화는 측정 가능해야 한다.

모든 numerical deviation은 설명 가능해야 한다.

모든 release는 재현 가능해야 한다.

**최종 목표는 단순 실행 성공이 아니라, 신뢰할 수 있는 Apple Silicon용 production-grade LTX-2.5 BF16 inference implementation을 만드는 것이다.**

프로젝트 공식 명칭:

# `Lara-LTX-2.5-MLX-BF16`

을 모든 source code, documentation, model card 및 benchmark에서 일관되게 사용한다.