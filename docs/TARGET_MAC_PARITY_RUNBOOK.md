# Target-Mac checkpoint parity runbook

This runbook is the required execution gate for P1/P2. Run it only on the
pinned Apple Silicon environment (Python 3.12 and MLX 0.32.0) with the
authorized, hash-verified BF16 model pack. It must not be run on AutoDL: CUDA
cannot validate Metal execution.

## Preconditions

1. Verify the model pack against `golden/manifests/upstream.json`.
2. Install this project in its pinned Python environment.
3. Set `LARA_MODEL_ROOT` to the directory containing the model-pack component
   folders, and set `LARA_REPORT_DIR` to a writable directory outside the
   repository if reports contain machine-specific measurements.
4. Do not run another model workload concurrently. The reports include MLX
   active and peak memory only for the component-loading section.

```sh
export LARA_MODEL_ROOT=/path/to/LTX-2.5
export LARA_REPORT_DIR=/path/to/lara-parity-reports
mkdir -p "$LARA_REPORT_DIR"
```

## P1/P2 checkpoint block gate

This command maps exactly the 84 approved block-0 tensors, evaluates the
fixed CUDA BF16 boundary, and writes video/audio metrics plus component-load
memory measurements.

```sh
.venv/bin/python tools/parity/compare_mlx_checkpoint_block.py \
  --transformer-checkpoint "$LARA_MODEL_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --mapping golden/manifests/transformer_block_0_mapping.json \
  --cuda-reference golden/cuda_checkpoint_block_0.npz \
  --report "$LARA_REPORT_DIR/checkpoint_block_0.json"
```

The command succeeds only when both video and audio outputs are finite, have
normalized RMSE at most `2e-2`, and cosine similarity at least `0.9999`.
Maximum absolute error remains diagnostic because BF16 output scale varies by
component. Record `weight_load_memory` from the generated JSON; it is the P1
evidence that this component stays within the target Mac's configured
working-set budget.

## P1/P2 Gemma feature-projection gate

This command uses safetensors' selected-key mmap path. It materializes only the
four reviewed LTX projection entries, never tokenizer/config assets or Gemma
language-model weights.

```sh
.venv/bin/python tools/parity/compare_mlx_gemma_feature.py \
  --gemma-checkpoint "$LARA_MODEL_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --mapping golden/manifests/gemma_feature_mapping.json \
  --cuda-reference golden/cuda_gemma_feature_reference.npz \
  --report "$LARA_REPORT_DIR/gemma_feature.json"
```

The command uses the same finite-output, normalized-RMSE and cosine criteria as
the block gate. Preserve the report and compare its
`weight_load_memory.peak_bytes` with the target machine's documented working
set. A failed criterion or over-budget result leaves P1/P2 open; do not weaken
the criteria or substitute quantized weights.

## P2 transformer output-head gate

This command loads the six reviewed video/audio output modulation and
projection tensors and compares the final BF16 heads with their CUDA capture.

```sh
.venv/bin/python tools/parity/compare_mlx_transformer_output.py \
  --transformer-checkpoint "$LARA_MODEL_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --mapping golden/manifests/transformer_output_mapping.json \
  --cuda-reference golden/cuda_transformer_output_reference.npz \
  --report "$LARA_REPORT_DIR/transformer_output.json"
```

Both streams must pass the same frozen checkpoint-backed BF16 criteria.

## P1 stage-local LoRA gate

The fixed CUDA LoRA artifact contains one official base/A/B tensor set, so this
gate does not require the full model pack. It proves both stage strengths use
the one-weight-at-a-time path without a duplicate transformer state.

```sh
.venv/bin/python tools/parity/compare_mlx_lora_fusion.py \
  --cuda-reference golden/cuda_lora_fusion_reference.npz \
  --cuda-report golden/cuda_lora_fusion_reference.json \
  --report golden/mlx_lora_fusion_report.json
```

The checked-in target-Mac report passes both stages: stage 1 has zero maximum
absolute error, stage 2 has `3.814697265625e-06`, and peak fusion memory is
`1,050,624` bytes.

## Evidence handling

- Preserve the checkpoint-block, Gemma-feature, transformer-output and LoRA JSON reports with the
  commit and MLX environment record.
- On a failure, retain the JSON report and re-run only after checking the
  mapping manifest, checkpoint hash, and Python/MLX versions.
- Do not publish a release or claim end-to-end parity merely because these component reports pass
  on the target Mac and the remaining P2 image-conditioning gate is complete.
