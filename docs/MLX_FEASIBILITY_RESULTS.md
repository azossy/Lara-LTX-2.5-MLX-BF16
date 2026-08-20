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
  orchestration, DiffVAE tile-performance measurement, and full Gemma 4
  conditioning rather than its already-verified LTX projections.
