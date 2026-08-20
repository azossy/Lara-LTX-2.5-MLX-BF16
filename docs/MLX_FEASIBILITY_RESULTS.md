# MLX feasibility results

## Environment

- Device: Apple M5 Max
- Physical unified memory: 137,438,953,472 bytes (128 GiB)
- MLX recommended working set: 115,448,725,504 bytes
- Python: 3.12.13
- MLX / MLX Metal: 0.32.0
- Probe dtype: BF16

The configured working-set budget resolves to MLX's recommended working set,
which is lower than both the physical-memory fraction and reserve-derived
limits. The probes therefore do not assume that all 128 GiB is available to the
process.

## Fused SDPA

Artifact: `golden/benchmarks/mlx_attention_feasibility.json`

| Shape | Elapsed | Estimated Q/K/V/output | Measured peak |
|---|---:|---:|---:|
| 8,160 tokens x 32 heads x 128 | 0.031 s | 267,386,880 B | 267,386,904 B |
| 32,640 tokens x 32 heads x 128 | 0.551 s | 1,069,547,520 B | 1,069,547,544 B |

The measured peak tracks Q/K/V/output storage rather than an explicit
`tokens x tokens` score matrix. This closes the immediate quadratic-memory
feasibility risk for unmasked video self-attention on MLX 0.32.0. It does not
yet prove the complete high-token transformer stage. The real checkpoint-backed
block-0 cross-modal gate is recorded separately below.

## Synthetic maximum-shape transformer block

Artifact: `golden/benchmarks/mlx_transformer_block_feasibility.json`

The probe runs BF16 RMSNorm, four 4096-wide self-attention projections, fused
SDPA, the output projection, residuals and a 4096 → 16384 → 4096 GELU FFN at
32,640 tokens.

| Measure | Result |
|---|---:|
| Attention section | 0.577 s |
| FFN section | 0.244 s |
| Whole synthetic block | 0.821 s |
| Active memory before evaluation | 670,040,120 B |
| Peak evaluation memory | 4,680,908,870 B |
| Active memory after intermediate references are released | 937,427,000 B |
| Allocator cache after the block | 4,813,029,390 B |

The allocator cache is intentionally not cleared inside the block; it is
reusable by following blocks and may be cleared at a component lifecycle
boundary. The active-memory drop demonstrates why explicit Python references
and MLX evaluation boundaries matter.

## Diffusion VAE 3D neighborhood attention

Artifact: `golden/cuda/primitive_bf16_cuda.npz`

The bounded MLX implementation was tested against the pinned upstream CUDA
eager `na3d` reference at BF16 for shape `(1, 3, 4, 5, 2, 4)` and kernel
`(3, 3, 3)`. It preserves shifted edge windows and causal-axis support, uses a
configured local-mask budget, and evaluates every query tile at the fused-SDPA
boundary so lazy graphs cannot retain all tile masks and K/V slices.

The CUDA-versus-MLX assertion passes at the project BF16 primitive tolerance
(`2e-2` absolute and relative). This closes the semantic reference gate; a
checkpoint-backed DiffVAE decode and tile-performance measurement remain
required before release.

## Video transformer block path

The CUDA golden archive covers a stable-scale BF16 video-only transformer
block: self-attention, text cross-attention, AdaLN gates and feed-forward
residual. It also covers video text cross-AdaLN and simultaneous audio/video
blocks with bidirectional cross-attention. MLX uses the upstream checkpoint key
layout and passes the same primitive tolerance with identical weights and
inputs for each of these paths. Checkpoint-backed transformer execution and
conditioning inputs remain separate gates.

## Checkpoint-backed target-Mac components

The complete seven-file official pack is size/SHA-256 verified in
`golden/manifests/local_bf16_pack_verification.json`. Selected-range BF16
loading then produced passing CUDA-versus-MLX reports for the exact 84-key AV
block-0 mapping, four Gemma LTX feature projections and six video/audio output
head tensors. Their component-load peaks are approximately 0.77 GB, 2.31 GB and
1.62 MB respectively. All outputs are finite, normalized RMSE stays below
0.0086, and cosine similarity exceeds 0.99996.

## Native Gemma 4 text conditioning

Artifact: `golden/mlx_gemma_text_full_report.json`

The pinned MLX-LM 0.31.3 text core exactly exposes the packed checkpoint's 666
`model.*` BF16 parameters. Lara separately loads the four LTX feature
projections and all five embedded tokenizer/config assets, collects the
Hugging Face-compatible 49 hidden states at controlled layer evaluation
boundaries, and never constructs the tied 262,144-token logits projection.
The full 1,024-token left-padded path produces finite `(1, 1024, 4096)` video
and `(1, 1024, 2048)` audio conditioning with a measured 26,126,823,208-byte
weight-load peak. This is a checkpoint-backed Metal execution result; full
CUDA-versus-MLX text-core tensor parity remains a P4 evidence task.

## Scheduler, guidance and res_2s

Artifact: `golden/mlx_sampling_report.json`

The MLX scheduler reproduces all 16 stage-1 sigma entries in the P0 CUDA HQ
boundary artifact with a maximum absolute error of `1.1920928955078125e-07`.
Guidance plans one combined transformer batch containing only the required
conditioned, unconditioned, STG-perturbed and AV-isolated passes, then attaches
per-block keep masks to the existing MLX transformer streams. The full res_2s
loop includes midpoint evaluation, optional anchor refinement, independent SDE
noise streams, clean-latent masking and terminal denoising. Synthetic loop
tests pass on Metal; denoiser-integrated stage parity remains a P4 task.

## Spatial latent upscaler

Artifact: `golden/mlx_spatial_upscaler_report.json`

The production MLX module strict-loads all 72 official BF16 upscaler weights
plus the two VAE channel-statistic tensors. It reproduces the source Conv3d,
float32-accumulating GroupNorm and SiLU, framewise Conv2d and source-order x2
pixel shuffle without converting the checkpoint. The CUDA-captured
denormalized input is bit-exact. The unnormalized x2 output reaches NRMSE
`0.006794` and cosine `0.999977`; the re-normalized output reaches NRMSE
`0.012722` and cosine `0.999919`. All three frozen criteria pass with a measured
995,736,328-byte component-load peak.

## Audio VAE decoder

Artifact: `golden/mlx_audio_vae_decoder_report.json`

The native MLX decoder loads the 56 official decoder weights plus both
patchified latent-statistic tensors. It reproduces height-causal Conv2d,
pixel normalization, FP32 SiLU evaluation, nearest-neighbor causal upsampling,
the upstream first-frame removal and exact variable-length crop. The fixed
CUDA `(1, 8, 18, 16)` latent produces a finite `(1, 2, 69, 64)` spectrogram
with NRMSE `0.003531`, cosine `0.999994`, and a 63,839,212-byte weight-load
peak.

## Waveform vocoder and bandwidth extension

Artifact: `golden/mlx_vocoder_bwe_report.json`

The native MLX waveform path strict-validates and loads all 1,227 official BF16
generator and STFT tensors. Both BigVGAN-v2 generators reproduce the
anti-aliased SnakeBeta activation in FP32, while the checkpoint-backed causal
STFT feeds the BWE residual generator and a deterministic Hann-sinc path
upsamples the 16 kHz skip signal to 48 kHz. The fixed CUDA spectrogram produces
a finite stereo waveform with NRMSE `0.004552`, cosine `0.999993`, and a
1,753,922,396-byte execution peak.

## Sequential 48-block transformer runtime

Artifacts: `golden/mlx_transformer_sequence_block0_report.json`,
`golden/mlx_transformer_sequence_48block_stage1_smoke.json`, and
`golden/mlx_transformer_sequence_48block_stage2_smoke.json`

The checkpoint-backed provider parses the 42 GB checkpoint header once, then
strict-loads, optionally fuses LoRA into, executes and releases exactly one
84-tensor AV block at a time. Block 0 retains the frozen CUDA parity result
(video NRMSE `0.008269`, audio NRMSE `0.005531`). The full 48-block stage-1
and stage-2 smoke runs each load 4,032 tensors and fuse all 1,632 block-local
LoRA pairs at strengths `0.25` and `0.5`; both streams remain finite, active
block residency stays below 773,816,712 bytes, and the maximum execution peak
is 2,153,945,408 bytes.

The low-memory provider above is a validation and reduced-memory fallback, not
the production default: reloading 42 GB for every denoiser evaluation would be
an unacceptable I/O bottleneck. The production resident provider instead
loads the 48 blocks once into 37,131,012,104 bytes and evaluates only the lazy
activation graph block by block. Its stage executions peak at 38,511,394,752
bytes. Between HQ stages it rereads and replaces only LoRA-targeted base
weights one block at a time, switching strength from `0.25` to `0.5` at a
38,234,978,314-byte peak without a duplicate transformer state. Evidence is in
`golden/mlx_resident_transformer_two_stage_report.json`.

The 28 LoRA pairs outside the block stack are also covered. The production
input/output loaders use the reviewed mapping source keys to fuse 26 timestep,
patch and AV-cross input weights plus both modality output projections. Stage
strengths `0.25` and `0.5` each produce finite `[1, 2, 128]` outputs at a
1,724,801,034-byte isolated peak. Reports are
`golden/mlx_transformer_io_lora_stage1_report.json` and
`golden/mlx_transformer_io_lora_stage2_report.json`; together with the 1,632
block-local pairs, all 1,660 official adapter pairs now have an execution path.

## Interpretation

- Maximum-shape fused attention is not the current memory blocker.
- A full 48-block runtime must still evaluate/release at controlled boundaries;
  building one giant lazy graph remains unnecessary risk.
- The synthetic block does not contain real 42 GB transformer weights, text
  cross-attention, audio blocks, audio/video cross-attention, AdaLN, guidance
  multiplicity or sampler overhead. Its timing is a feasibility measurement,
  not an end-to-end generation estimate or release benchmark.
- With component-level residency, the transformer checkpoint plus the measured
  working space has substantial headroom under the 115.45 GB budget. The exact
  peak must be re-measured with mapped checkpoint weights and stage-specific
  LoRA handling.
- The remaining high-risk technical paths are the full 48-block/high-token
  orchestration, two-stage component lifecycle, DiffVAE tile performance, and
  audiovisual mux synchronization. Full Gemma 4 conditioning, spatial x2
  upscaling and 48 kHz waveform reconstruction are now executable; Gemma's
  CUDA-versus-MLX hidden-state report remains evidence work.
