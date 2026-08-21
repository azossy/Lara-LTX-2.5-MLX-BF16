# Quality evaluation

The release publishes raw native outputs, deterministic presentation exports,
machine-readable generation metadata, and evaluator reports. It does not reduce
the comparison to one composite score: each metric measures a different property
and can disagree legitimately.

## Reproducibility contract

- CUDA and MLX use the same prompt, negative prompt, seed, source grid, frame
rate, frame count, and inference-step count.
- VideoScore2 uses its pinned model revision, loads the evaluator once for the
  matched pair, and resets the same recorded sampling seed before each case.
  The report retains both parsed scores and the raw model response.
- Native outputs remain available and are hashed before presentation processing.
- “4K” files are Lanczos presentation upscales to 3840×2160. They are explicitly
  labeled as non-native 4K and are never used as evaluator inputs.
- Source repositories, revisions, checkpoint URLs, and SHA-256 digests are pinned
  in `configs/quality/external_evaluators.toml`.
- CLAP compares audio with shared audio-only descriptions. Feeding the full visual
  generation prompt to an audio/text model would dilute the measurement.
- CLAP receives the first 10.000 seconds as 48 kHz mono PCM because the pinned
  non-fusion checkpoint has a fixed 480,000-sample input window. Native files
  remain untrimmed and Audiobox evaluates their complete audio streams.
- CUDA and MLX audio remain separate. The side-by-side file carries both tracks;
  it does not mix them into a new waveform. Presentation-only tracks receive
  silence padding when required so a slightly shorter AAC stream cannot remove
  the final video frame.

## Metrics

VBench reports subject consistency, background consistency, motion smoothness,
dynamic degree, aesthetic quality, and imaging quality. Audiobox Aesthetics
reports Content Enjoyment (CE), Content Usefulness (CU), Production Complexity
(PC), and Production Quality (PQ). LAION CLAP reports cosine similarity between
the generated audio and its fixed audio description. VideoScore2 reports visual
quality, text alignment, and physical/common-sense consistency for the final
matched cinematic pair.

These automated metrics are evidence, not ground truth. The Hugging Face model
card therefore presents the two videos together with the raw reports so viewers
can inspect hair motion, facial identity, eye acting, temporal stability, audio,
and artifacts directly.

## Published cinematic pair

The public pair uses seed `25082110`, 512×320, 241 frames, 24 fps and 15
inference steps. Both outputs are 10.042-second H.264/AAC files. CUDA generation
took 74.0478 seconds (3.2547 generated frames/s); MLX/Metal generation took
2,221.3035 seconds (0.1085 generated frames/s). The observed ratio is 29.9982×.
Because the two deployments use different systems, this is not an intrinsic
CUDA-versus-Metal hardware benchmark.

### VBench

| Metric | CUDA | MLX/Metal |
|---|---:|---:|
| Subject consistency | 0.878626 | 0.879897 |
| Background consistency | 0.931389 | 0.915084 |
| Motion smoothness | 0.986575 | 0.987705 |
| Dynamic degree | 1.000000 | 1.000000 |
| Aesthetic quality | 0.520446 | 0.564146 |
| Imaging quality | 0.666255 | 0.628761 |

### VideoScore2

| Dimension (1–5) | CUDA | MLX/Metal |
|---|---:|---:|
| Visual quality | 4 | 4 |
| Text-to-video alignment | 4 | 5 |
| Physical/common-sense consistency | 4 | 4 |

VideoScore2 is pinned to model revision
`09a2732cb64fa566a1f332f978368292ce5c295c` and source revision
`a88168af50b1dd98f0c3d973620ac495daab2de4`. The runner uses configurable
`decord==0.6.0` video decoding because current torchvision no longer exposes
`torchvision.io.read_video`. It parses the model's final response after any
tagged reasoning block and retains the complete raw response in the report.

### Audio

| Metric | CUDA | MLX/Metal |
|---|---:|---:|
| Audiobox Content Enjoyment (CE) | 3.017473 | 7.604846 |
| Audiobox Content Usefulness (CU) | 5.559971 | 8.321693 |
| Audiobox Production Complexity (PC) | 2.364491 | 5.647840 |
| Audiobox Production Quality (PQ) | 6.132813 | 8.360165 |
| LAION CLAP cosine similarity | 0.297631 | -0.093657 |

CLAP uses the same audio-only text description for both cases. A negative CLAP
cosine is reported as measured and is not hidden or converted to a pass/fail
claim. Audiobox and CLAP measure different audio properties and may disagree.

### Public artifacts

The native videos, 4K presentation exports and raw reports are published at
[`LaraAI-Labs/Lara-LTX-2.5-MLX-BF16`](https://huggingface.co/LaraAI-Labs/Lara-LTX-2.5-MLX-BF16).
GitHub retains the reproducible manifests, evaluator configuration, runners and
summary documentation without adding large generated videos to source history.
