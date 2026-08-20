# Active execution plan

This document is an execution view of the controlling scope directive and
`docs/PORTING_PLAN.md`. It does not relax any release or parity gate.

## Current active gate: P4 stochastic trajectory parity and P5 quality corpus

| Order | Work | Completion evidence | State |
|---|---|---|---|
| 1 | Transfer the pinned DEV BF16 transformer | Expected byte size and SHA-256 | Complete |
| 2 | Transfer and verify every remaining required component | Every manifest entry has expected size, recorded SHA-256, and a header inventory with BF16 payloads plus only explicitly reviewed upstream FP32 modulation tables and Gemma tokenizer/config byte assets | Complete |
| 3 | Run official two-stage HQ CUDA BF16 without quantization | Fixed-seed MP4, command log, output SHA-256 and environment record | Complete: 512x320, 17 frames, seed 25081900 |
| 4 | Validate generated media | Non-empty playable MP4, video/audio stream metadata, fixed-run report | Complete: H.264 video and AAC audio confirmed |
| 5 | Capture core CUDA reference artifacts | Serialized conditioning, sampler, latent, decoder and audio boundary tensors | Complete: expanded v3 SDE/denoiser trace, v4 seven-block/three-pass deep trace, v5 block-0 module trace and four-shard v6 Attention sub-operation trace are local and SHA-256 verified |
| 6 | Capture the versioned CUDA quality corpus | Four prompts/seeds, four resolutions, three frame counts and synchronized-audio MP4s | Complete: all four H.264/AAC cases are local and validated |
| 7 | Transfer and verify the official pack on the target Mac | All seven entries match pinned byte sizes and SHA-256 values | Complete: `golden/manifests/local_bf16_pack_verification.json` |
| 8 | Execute target-Mac checkpoint-backed P1/P2 gates | Block-0, Gemma projection, output head and LoRA reports pass frozen BF16 criteria with bounded loading | Complete: all four target-Mac reports pass |

P0 is complete for the fixed smoke baseline. A plausible MP4 alone is not a
parity result; completion required the model-pack verification, internal tensor
trace and repeated-run determinism evidence recorded above.

The CUDA golden runner defaults to the upstream-compatible `chunked_eager`
Diffusion VAE mode. The Blackwell DSL mode is opt-in only when the exact
datacenter GPU and `nvidia-cutlass-dsl` dependency gate have been verified.

## Next implementation gates

| Priority | Work | Exit evidence | Dependency |
|---|---|---|---|
| P1 | Inspect all real BF16 shards; create reviewed mapping manifests for transformer, Gemma/projection, audio VAE, upscaler, duration head and distilled LoRA | Every source key is accounted for exactly once, with validated shape/dtype/layout transforms | Header-derived templates for all seven components; Diffusion VAE decoder, DurationHead, all 48 transformer blocks, Gemma V2's four LTX feature projections, the complete 72-tensor x2 latent spatial upscaler, the 102-tensor Audio VAE core, and the 1,227-tensor waveform vocoder/BWE plus STFT bases have reviewed mappings. Gemma U8 tokenizer/config assets remain on their own loader path |
| P1 | Implement bounded component-level MLX loading and LoRA application | No duplicate full-model residency; BF16 values and shapes verified | Complete on target M5 Max: block-0, Gemma projection and output-head loads pass frozen CUDA-versus-MLX BF16 criteria at peaks of 773,563,888, 2,312,122,440 and 1,622,608 bytes; stage-local LoRA peaks at 1,050,624 bytes |
| P2 | Complete checkpoint-backed transformer conditioning, patchification and remaining modulation paths | Real mapped block CUDA-versus-MLX tensor report plus strict official input-weight execution | Complete: mapped block-0, Gemma projection and both output heads pass the frozen Metal criteria. The exact 53-key input component now strict-loads the official BF16 checkpoint and executes projection, timestep/prompt AdaLN, keyframe marker, masks, main/cross RoPE and bidirectional AV modulation on Metal at an 853,353,292-byte measured peak |
| P3 | Port Gemma conditioning, scheduler, CFG/STG/res2s, spatial upscaler, audio VAE/vocoder and single in-process media pipeline | Fixed prompt produces a playable MLX MP4 with audio | Complete: official Gemma prompt connectors now pass CUDA boundary parity; corrected public Python and installed CLI produce byte-identical H.264/AAC output at a 40,749,191,990-byte MLX peak |
| P4/P5 | Replay the expanded stochastic CUDA trace, then run the multi-prompt audiovisual quality corpus | Frozen final-latent tolerances, documented differences and acceptance report | Block-0 CUDA capture and all 176 exact-input Attention comparisons pass. FP32 RMSNorm reproduces the captured AdaLN boundary exactly; the combined full replay ends at video/audio NRMSE `0.1916`/`0.1110`, so same-seed CUDA reproduction remains open. An FP32 attention sigmoid/gate experiment was rejected because full video/audio NRMSE regressed to `0.7320`/`0.2236`. The next bounded CUDA diagnostic captures block 31/39 sub-operations. All four MLX corpus cases and paired diagnostics are complete; the deterministic blind A/B kit is prepared but independent review remains open. |
| P6 | Measure and optimize on the target Apple Silicon hardware | Peak memory, latency and Metal utilization reports | The 20 GiB decoder activation budget eliminates pathological 51,200-tile execution and completes the maximum measured grid in 92.0 seconds at 40.25 GB. The public opt-in profiler records component load, stage latency and MLX active/cache/peak memory; repeated generation is byte-identical with zero released-memory growth. Privileged macOS GPU duty-cycle percentage is not claimed. |
| P7/P8 | Stabilize Python API and thin CLI | Clean local installation and localized user-facing errors | Complete: the v0.0.3 wheel was installed into an isolated Python 3.12 environment; public API, CLI help and actual 512x320/17 H.264/AAC generation passed |
| P9 | Build separate thin ComfyUI adapter | Adapter only calls the public Python pipeline | Complete: three nodes, localized resources, output confinement and tested API workflow |
| P10 | Publish GitHub and Hugging Face releases | Tagged, traceable, license-compliant verified public artifacts | v0.0.3 is published and cross-linked on both services; remaining numerical/blind-quality gates are tracked without overstating parity |

## Non-negotiable controls

- Do not substitute quantized or different model files for the pinned DEV BF16
  pack.
- Do not add server, daemon, HTTP API, cloud job system or a second PyTorch/MPS
  execution path.
- Do not publish source or model artifacts until the upstream license and
  attribution obligations are packaged and every release acceptance gate has
  evidence.
- Do not claim CUDA/MLX output parity before P4/P5 evidence exists.
