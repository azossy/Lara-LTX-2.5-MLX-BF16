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

## P3 native Gemma 4 conditioning gate

This command loads all 666 text-core tensors and four LTX projections, builds
the tokenizer from the five embedded byte assets, skips LM logits and executes
the production 1,024-token hidden-state path.

```sh
.venv/bin/python tools/parity/validate_mlx_gemma_text.py \
  --gemma-checkpoint "$LARA_MODEL_ROOT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --text-mapping golden/manifests/gemma_text_mapping.json \
  --feature-mapping golden/manifests/gemma_feature_mapping.json \
  --prompt "A cinematic ocean at sunrise with synchronized waves and distant seabirds." \
  --sequence-length 1024 \
  --report "$LARA_REPORT_DIR/gemma_text.json"
```

The checked-in report produces 49 hidden states and finite video/audio
features at a 26,126,823,208-byte load peak. This gate proves native execution,
not cross-backend numerical parity; that comparison remains in P4.

## P3 scheduler and guidance gate

```sh
.venv/bin/python tools/parity/validate_mlx_sampling.py \
  --cuda-reference golden/cuda_hq_boundary_trace_v2.npz \
  --report "$LARA_REPORT_DIR/sampling.json"
```

The scheduler must match the captured HQ stage-1 sigmas within `2e-7`, the
combined CFG/STG/AV-isolation arithmetic must be exact, and the res_2s
coefficients must remain finite.

## P3 bounded 48-block transformer gate

```sh
.venv/bin/python tools/parity/validate_mlx_transformer_sequence.py \
  --transformer-checkpoint "$LARA_MODEL_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --cuda-reference golden/cuda_checkpoint_block_0.npz \
  --block-count 48 \
  --lora-checkpoint "$LARA_MODEL_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --lora-strength 0.25 \
  --report "$LARA_REPORT_DIR/transformer_sequence_stage1.json"
```

The gate must execute 48 records, load 84 tensors per block, fuse 34 LoRA pairs
per block, produce finite video/audio outputs, and keep active block residency
bounded rather than accumulating prior blocks. CUDA numerical comparison is
also required in the one-block form without `--lora-checkpoint`; later blocks
use this command as a lifecycle smoke gate until denoiser-integrated P4
boundaries are captured.

Repeat the same command with `--lora-strength 0.5` and a distinct report path
for the official stage-2 lifecycle gate.

For the production one-load path, run both stages in one process:

```sh
.venv/bin/python tools/parity/validate_mlx_resident_transformer.py \
  --transformer-checkpoint "$LARA_MODEL_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --lora-checkpoint "$LARA_MODEL_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --reference golden/cuda_checkpoint_block_0.npz \
  --report "$LARA_REPORT_DIR/resident_transformer_two_stage.json"
```

The resident report must contain 48 blocks, finite outputs for both strengths,
no second transformer state, and a measured peak within the target Mac budget.

Validate the remaining input/output LoRA pairs for each stage:

```sh
.venv/bin/python tools/parity/validate_mlx_transformer_io_lora.py \
  --transformer-checkpoint "$LARA_MODEL_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --lora-checkpoint "$LARA_MODEL_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --input-mapping golden/manifests/transformer_input_mapping.json \
  --output-mapping golden/manifests/transformer_output_mapping.json \
  --lora-strength 0.25 \
  --report "$LARA_REPORT_DIR/transformer_io_lora_stage1.json"
```

Repeat with strength `0.5`. Each report must fuse 26 input and two output pairs
and produce finite video/audio outputs.

Validate the complete input-to-denoised resident path across both stages:

```sh
.venv/bin/python tools/parity/validate_mlx_integrated_denoiser.py \
  --transformer-checkpoint "$LARA_MODEL_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --lora-checkpoint "$LARA_MODEL_ROOT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --input-mapping golden/manifests/transformer_input_mapping.json \
  --output-mapping golden/manifests/transformer_output_mapping.json \
  --block-count 48 \
  --token-count 2 \
  --context-token-count 3 \
  --sigma 0.5 \
  --seed 25081900 \
  --stage-lora-strength 0.25 \
  --stage-lora-strength 0.5 \
  --require-all-lora-pairs \
  --report "$LARA_REPORT_DIR/integrated_denoiser_two_stage.json"
```

The report must cover all 1,660 LoRA pairs, execute the configured number of
resident blocks on every stage, produce finite `[1, 2, 128]` denoised video
and audio tokens, and remain within the target Mac memory budget. All runtime
controls are explicit arguments so the same gate can use smaller diagnostic
shapes without changing source code.

## P3 spatial latent-upscaler gate

This command strict-loads all 72 official upscaler tensors and both VAE
channel-statistic tensors, then compares the denormalized input, unnormalized
x2 output and re-normalized x2 output with the AutoDL CUDA BF16 capture.

```sh
.venv/bin/python tools/parity/compare_mlx_spatial_upscaler.py \
  --upscaler-checkpoint "$LARA_MODEL_ROOT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --video-vae-checkpoint "$LARA_MODEL_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" \
  --mapping golden/manifests/spatial_upscaler_mapping.json \
  --cuda-reference golden/cuda_spatial_upscaler_reference.npz \
  --report "$LARA_REPORT_DIR/spatial_upscaler.json"
```

Every boundary must be finite, have normalized RMSE at most `2e-2`, and have
cosine similarity at least `0.9999`. The checked-in target-Mac report passes
with 74 loaded tensors and a 995,736,328-byte weight-load peak.

## P3 Audio VAE decoder gate

```sh
.venv/bin/python tools/parity/compare_mlx_audio_vae_decoder.py \
  --checkpoint "$LARA_MODEL_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --mapping golden/manifests/audio_vae_mapping.json \
  --cuda-reference golden/cuda_audio_decode_reference.npz \
  --report "$LARA_REPORT_DIR/audio_vae_decoder.json"
```

The decoder must strict-load 56 model tensors and two latent-statistic tensors.
Its decoded spectrogram must be finite, have normalized RMSE at most `2e-2`,
and cosine similarity at least `0.9999`. The checked-in Metal report passes at
NRMSE `0.003531` and cosine `0.999994`.

## P3 waveform vocoder and BWE gate

```sh
.venv/bin/python tools/parity/compare_mlx_vocoder_bwe.py \
  --checkpoint "$LARA_MODEL_ROOT/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --mapping golden/manifests/vocoder_mapping.json \
  --cuda-reference golden/cuda_audio_decode_reference.npz \
  --report "$LARA_REPORT_DIR/vocoder_bwe.json"
```

This gate loads the 1,227 primary-generator, causal-STFT and BWE-generator
tensors, reconstructs the stereo 48 kHz waveform and compares the complete
user-visible output. It must be finite, have normalized RMSE at most `2e-2`,
and cosine similarity at least `0.9999`. The checked-in Metal report passes at
NRMSE `0.004552`, cosine `0.999993`, and a 1,753,922,396-byte peak.

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

## P2 transformer-input loading and execution gate

This gate strict-loads all 53 official input-projection and conditioning
tensors and executes a bounded simultaneous video/audio preparation on Metal.

```sh
.venv/bin/python tools/parity/validate_mlx_transformer_input.py \
  --transformer-checkpoint "$LARA_MODEL_ROOT/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --mapping golden/manifests/transformer_input_mapping.json \
  --report "$LARA_REPORT_DIR/transformer_input.json"
```

The checked-in target-Mac report loads 53 BF16 tensors, produces all 26
required finite outputs and records an 853,353,292-byte peak.

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

- Preserve the checkpoint-block, Gemma-feature, full Gemma-text, sampling,
  spatial-upscaler, Audio VAE decoder, waveform-vocoder/BWE,
  transformer-input, transformer-output and LoRA JSON reports with the commit
  and MLX environment record.
- On a failure, retain the JSON report and re-run only after checking the
  mapping manifest, checkpoint hash, and Python/MLX versions.
- Do not publish a release or claim end-to-end parity merely because these
  component reports pass on the target Mac. Optional image conditioning remains
  gated behind the canonical HQ T2V plus synchronized-audio pipeline.
