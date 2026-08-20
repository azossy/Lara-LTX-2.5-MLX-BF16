# Active execution plan

This document is an execution view of the controlling scope directive and
`docs/PORTING_PLAN.md`. It does not relax any release or parity gate.

## Current active gate: P3 end-to-end MLX pipeline assembly

| Order | Work | Completion evidence | State |
|---|---|---|---|
| 1 | Transfer the pinned DEV BF16 transformer | Expected byte size and SHA-256 | Complete |
| 2 | Transfer and verify every remaining required component | Every manifest entry has expected size, recorded SHA-256, and a header inventory with BF16 payloads plus only explicitly reviewed upstream FP32 modulation tables and Gemma tokenizer/config byte assets | Complete |
| 3 | Run official two-stage HQ CUDA BF16 without quantization | Fixed-seed MP4, command log, output SHA-256 and environment record | Complete: 512x320, 17 frames, seed 25081900 |
| 4 | Validate generated media | Non-empty playable MP4, video/audio stream metadata, fixed-run report | Complete: H.264 video and AAC audio confirmed |
| 5 | Capture missing internal CUDA reference artifacts | Serialized conditioning, sampler, latent, decoder and audio boundary tensors | Complete: 25-boundary trace plus repeated-run determinism report |
| 6 | Transfer and verify the official pack on the target Mac | All seven entries match pinned byte sizes and SHA-256 values | Complete: `golden/manifests/local_bf16_pack_verification.json` |
| 7 | Execute target-Mac checkpoint-backed P1/P2 gates | Block-0, Gemma projection, output head and LoRA reports pass frozen BF16 criteria with bounded loading | Complete: all four target-Mac reports pass |

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
| P3 | Port Gemma conditioning, scheduler, CFG/STG/res_2s, spatial upscaler, audio VAE/vocoder and single in-process media pipeline | Fixed prompt produces a playable MLX MP4 with audio | Native Gemma 4 conditioning, scheduler/guidance/res_2s, the 72-weight x2 upscaler, 58-tensor Audio VAE decoder, complete 1,227-tensor vocoder/BWE path, resident 48-block transformer, and all 1,660 stage-local LoRA pairs are executable. Denoiser wiring, two-stage latent lifecycle and media mux remain |
| P4/P5 | Run deterministic tensor/stage reports and multi-prompt audiovisual quality corpus | Evidence-based tolerances, documented differences and acceptance report | P3 end-to-end path |
| P6 | Measure and optimize on the target Apple Silicon hardware | Peak memory, latency and Metal utilization reports | Correct P3/P4 path |
| P7/P8 | Stabilize Python API and thin CLI | Clean local installation and localized user-facing errors | P3-P6 acceptance |
| P9 | Build separate thin ComfyUI adapter | Adapter only calls the public Python pipeline | P7/P8 acceptance |
| P10 | Publish GitHub and Hugging Face releases | Tagged, traceable, license-compliant verified public artifacts | All preceding release gates |

## Non-negotiable controls

- Do not substitute quantized or different model files for the pinned DEV BF16
  pack.
- Do not add server, daemon, HTTP API, cloud job system or a second PyTorch/MPS
  execution path.
- Do not publish source or model artifacts until the upstream license and
  attribution obligations are packaged and every release acceptance gate has
  evidence.
- Do not claim CUDA/MLX output parity before P4/P5 evidence exists.
