# Execution TODO

This list follows the controlling scope directive. Work on later priorities may
be scaffolded, but an unmet earlier gate remains the active project priority.
The live execution sequence and evidence required at each gate are maintained
in `docs/ACTIVE_EXECUTION_PLAN.md`.

## P0 — CUDA Golden Reference

- [x] Pin official source commit and model revision.
- [x] Capture sanitized local MLX environment metadata.
- [x] Finish the pinned Blackwell-compatible PyTorch/CUDA environment.
- [x] Prove real `sm_120` CUDA computation on both reference GPUs.
- [x] Obtain authorized access to the official gated LTX-2.5 BF16 files.
- [x] Provision enough persistent storage for the complete split pack and
      golden artifacts without risking the active GPU allocation (300 GB data
      disk verified after restart).
- [x] Download and verify every required DEV BF16 component.
- [x] Archive the resolved CUDA lock and full environment manifest.
- [x] Run the official two-stage HQ pipeline in BF16 without quantization and
      preserve the fixed-seed 512x320/17-frame MP4, command report and stream
      validation report.
- [x] Capture deterministic conditioning, sampler, latent, decoder, audio and
      final-video golden artifacts (primitive plus small video/AV transformer
      block artifacts are captured); freeze the measured CUDA vocoder tolerance.
- [x] Capture the fixed-P0-latent CUDA Audio VAE → primary vocoder → BWE
      waveform boundaries; confirm the final waveform's maximum difference from
      the original P0 trace remains at the documented BF16 tolerance.
- [x] Capture fixed-P0-latent Video VAE statistics and x2 spatial-upscaler
      boundaries; confirm the re-normalized upscaled latent is bitwise equal to
      the original P0 stage-transition boundary.
- [x] Capture the checkpoint-backed CUDA Gemma V2 feature path with deterministic
      hidden states, including per-token RMS flattening, masked-token zeroing,
      and both video/audio projection outputs.
- [x] Run a real target-Mac Metal smoke test in the exact Python 3.12 / MLX
      environment.
- [x] Benchmark fused SDPA at 8,160 and 32,640 video tokens with the canonical
      32 heads x 128 head dimension; record time and peak memory.
- [x] Prove a synthetic block-boundary evaluation policy at the canonical 4096 hidden
      width without retaining a 48-block lazy graph.
- [x] Port and CUDA-parity-test one bounded 3D neighborhood-attention reference
      case before the full Diffusion VAE implementation.

## P1 — MLX weight loading (complete)

- [x] Implement allocation-safe safetensors header inspection.
- [x] Implement BF16 MLX loading for one validated shard at a time, returning
      only requested tensors after header-derived dtype/shape verification.
- [x] Implement deterministic, header-derived checkpoint mapping templates with
      cross-shard duplicate-key rejection.
- [x] Require an explicit, one-to-one reviewed target mapping and reject
      unsupported transforms or component-key-set mismatches before loading.
- [x] Implement shard-at-a-time mapped weight batches without duplicate
      full-model residency or per-layer denoising-time disk swapping.
- [x] Enforce requested-tensor dtype policy during shard loading: BF16 payloads
      plus explicitly mapped F32 auxiliary weights are accepted; embedded U8
      tokenizer/config assets are never treated as MLX model weights.
- [x] Define and version the official Diffusion VAE decoder mapping: 407
      target keys, 144 explicit QKV-split rules and one audited `type_emb`
      omission in `golden/manifests/vae_bf16_decoder_mapping.json`.
- [x] Strict-load the official VAE decoder through the pinned CUDA reference:
      all 407 mapped targets load with zero missing and unexpected keys in
      `golden/manifests/vae_cuda_strict_load.json`.
- [x] Create header-derived review templates for all seven pinned model
      components, including the transformer, Gemma/projection, both VAEs,
      upscaler, duration head and distilled LoRA.
- [x] Define the standalone DurationHead mapping: strip the official prefix and
      split PyTorch fused attention QKV tensors into three reviewed MLX targets.
- [x] Implement the MLX DurationHead with the exact 19-key parameter contract,
      and regenerate/validate its manifest against the authorized official BF16
      shard on AutoDL.
- [x] Validate all 1,660 official distilled-LoRA A/B pairs against their base
      transformer weights and preserve the resulting rank/layout manifest.
- [x] Validate all 48 production AV transformer blocks: each has 84 mapped
      parameters, six reviewed FP32 AdaLN tables, and BF16 operational weights.
- [x] Isolate and validate the four BF16 Gemma V2 LTX feature-extractor
      projections directly from the authorized shard; tokenizer/config U8 assets
      and Gemma language-model tensors remain outside this component mapping.
- [x] Validate the complete 72-tensor BF16 x2 latent spatial-upscaler mapping,
      including its frame-wise 2D pixel-shuffle projection and 3D residual paths.
- [x] Separate and validate all 102 BF16 Audio VAE core tensors from the shared
      Audio VAE/vocoder pack; retain the 1,227 vocoder/BWE tensors for their
      dedicated waveform-decoding loader.
- [x] Validate all 1,227 BF16 waveform-vocoder/BWE tensors and three exact
      STFT basis tensors against the official checkpoint configuration.
- [x] Capture CUDA BF16 stage-1/stage-2 LoRA fusion tensors for a fixed official
      adapter/base pair; compare them against the one-weight-at-a-time MLX
      fusion implementation on the target Apple Silicon environment.
- [x] Re-audit every mapped official component shape, dtype, layout transform
      and duplicate/missing key: 48 transformer blocks × 84 rules, Gemma (4),
      x2 upscaler (72), Audio VAE core (102), vocoder/BWE (1,227), DurationHead,
      distilled LoRA and Diffusion VAE decoder manifests all remain valid.
- [x] Prove BF16 value preservation and bounded peak load memory on the target
      M5 Max: the hash-verified checkpoint-backed block, output-head and Gemma
      comparisons pass the frozen cross-backend BF16 criteria with measured
      component-load peaks of 773,563,888, 1,622,608 and 2,312,122,440 bytes.
- [x] Implement stage-local distilled-LoRA application without holding a second
      complete transformer copy; the target-Mac CUDA comparison passes for both
      stage strengths with a measured 1,050,624-byte fusion peak.

## P2 — MLX/Metal core inference

- [x] Scaffold RMSNorm, tanh GELU, Split RoPE and fused MLX SDPA.
- [x] Pass local unit tests and CUDA BF16 primitive parity.
- [x] Implement and CUDA-BF16-parity-test the video-only self-attention, text
      cross-attention and AdaLN-gated feed-forward block path.
- [x] Implement and CUDA-BF16-parity-test video text cross-AdaLN modulation.
- [x] Implement AV prompt-side cross-AdaLN and capture a deterministic CUDA BF16
      reference for its simultaneous audio/video path; execute the MLX comparison
      on the target Apple Silicon environment.
- [x] Complete video/audio output modulation, exact six-key checkpoint mapping
      and target-Mac CUDA-versus-MLX BF16 comparison for both streams.
- [x] Complete video transformer blocks with cross-AdaLN. Image-conditioned
      latent construction is a pipeline conditioning concern, not a separate
      transformer-block path, and remains deferred until canonical HQ T2V with
      synchronized audio passes as required by the release gate.
- [x] Implement and CUDA-BF16-parity-test the non-cross-AdaLN audio block and
      simultaneous bidirectional audio/video cross-attention.
- [x] Complete patchification, timestep embeddings and conditioning inputs:
      the video/audio preprocessor implements patchified-token projection,
      1000x timestep AdaLN, prompt sigma AdaLN, keyframe marker, masks, main
      and cross RoPE, bidirectional AV scale/shift and gate assembly. Its exact
      53-key official BF16 mapping strict-loads and executes on Metal with a
      measured 853,353,292-byte peak.
- [x] Execute the prepared target-Mac comparison for the checkpoint-backed CUDA
      BF16 AV block boundary and its exact 84-key mapping.
- [x] Provide a target-Mac runner that loads the official mapped block-0 weights
      one component batch at a time and reports CUDA-versus-MLX video/audio
      metrics and component-load memory from the fixed checkpoint-backed
      reference.
- [x] Provide a target-Mac runner that loads the four mapped Gemma V2 LTX
      feature projections and reports CUDA-versus-MLX video/audio feature
      metrics and component-load memory from the fixed checkpoint-backed
      reference.
- [x] Implement bounded MLX 3D neighborhood attention for Diffusion VAE.
- [x] Implement and CUDA-BF16-parity-test the causal 3D Diffusion VAE ResNet
      block with PixelNorm/GroupNorm and channel-changing residual projection.
- [x] Implement and CUDA-BF16-parity-test VAE depth-to-space upsampling and
      space-to-depth downsampling, including temporal first-frame handling and
      residual paths.

## P3 — End-to-end video generation

- [x] Adapt the pinned native MLX-LM Gemma 4 text core to the packed LTX
      tokenizer, contiguous left-padding masks, 49 hidden states and dual
      projections without constructing LM logits. All 666 text weights plus
      four feature weights strict-map from the official shard; the full
      1,024-token Metal execution passes with a 26,126,823,208-byte load peak.
- [x] Port the token-shifted scheduler, batched CFG/STG/AV-isolation guidance,
      block perturbation masks and full two-evaluation res_2s/SDE loop. The
      15-step schedule matches the P0 CUDA HQ artifact within `1.2e-7` and the
      deterministic Metal contract suite passes.
- [x] Implement sequential checkpoint-backed execution for all 48 production
      AV transformer blocks with per-block graph materialization and immediate
      release. The full stage-1 and stage-2 smoke runs each strict-load 4,032
      tensors, fuse all 1,632 block-local LoRA pairs at strengths `0.25` and
      `0.5`, hold active block memory below 773,816,712 bytes, and produce
      finite video/audio outputs at a 2,153,945,408-byte peak.
- [x] Add the production resident-weight mode so denoiser steps do not reread
      the 42 GB checkpoint. All 48 blocks occupy 37,131,012,104 bytes; stage-1
      and stage-2 execution peak at 38,511,394,752 bytes. Rebuilding only the
      LoRA-targeted weights from the base checkpoint switches `0.25` to `0.5`
      in place at a 38,234,978,314-byte peak without a second model state.
- [x] Fuse the remaining 28 non-block LoRA pairs into the official input
      preprocessor (26) and video/audio output projections (2). Both `0.25`
      and `0.5` stage strengths produce finite `[1, 2, 128]` modality outputs
      at a measured 1,724,801,034-byte isolated component peak.
- [x] Port the complete 72-weight spatial latent upscaler, VAE statistic
      denormalization/renormalization and source-order pixel shuffle. All three
      CUDA boundaries pass; the normalized output has NRMSE `0.01272` and
      cosine similarity `0.999919` at a 995,736,328-byte load peak.
- [ ] Wire the validated upscaler into stage-2 transformer refinement.
- [ ] Complete diffusion video VAE: non-attention Conv decoder subset and
      CUDA-BF16 assembly parity, per-frame attention, and timestep-conditioned
      ResNet/mid-block parity are complete. Production Diffusion VAE shared
      layers, deterministic NABlock/stage, and stage-5 combined block parity are
      complete. Full untiled decoder orchestration and the two-step diffusion
      loop now pass CUDA BF16 parity; stage-5 halo tiling is implemented and
      parity-tested. Stage-4 halo tiling plus production minimum-size padding,
      trailing-frame ghosting, and exact output cropping are implemented. A
      conservative activation-budget selector now chooses both high-resolution
      tile sizes without quantization or weight swapping; live Metal free-memory
      discovery remains pipeline integration work.
- [x] Port and checkpoint-back the causal pixel-normalized Audio VAE decoder.
      All 56 decoder weights plus two normalization statistics load from the
      official shard; the CUDA spectrogram boundary passes with NRMSE
      `0.003531` and cosine `0.999994` at a 63,839,212-byte load peak.
- [x] Port the 1,227-tensor primary vocoder, causal checkpoint-backed mel STFT,
      BWE residual generator and 16-to-48 kHz sinc reconstruction. The complete
      Metal waveform path passes the CUDA reference with NRMSE `0.004552` and
      cosine similarity `0.999993` at a 1,753,922,396-byte peak.
- [ ] Wire decoded 48 kHz stereo waveforms into the end-to-end media mux and
      validate synchronization against the fixed CUDA clip.
- [ ] Produce a playable fixed-prompt MP4 through one in-process MLX pipeline.
- [ ] Pass HQ text-to-video plus synchronized audio before enabling optional
      image conditioning through that same pipeline.

## P4 — Tensor and stage parity

- [ ] Automate CUDA-versus-MLX reports for every major checkpoint.
- [ ] Add first-divergent-layer bisect tooling.
- [ ] Freeze evidence-based tolerances and deterministic inputs.

## P5 — BF16 quality parity

- [ ] Build the multi-prompt/seed/duration/resolution acceptance corpus.
- [ ] Measure frame, temporal, perceptual and audio-sync quality.
- [ ] Complete blind review and document known differences.

## P6 — Apple Silicon optimization

- [ ] Measure M5 Max 128 GB peak unified memory and repeated-run growth.
- [ ] Measure load time, generation time and Metal utilization.
- [ ] Apply safe MLX evaluation, lifetime, fused-op and compile optimizations.
- [ ] Add custom Metal kernels only for measured unresolved bottlenecks.
- [ ] Record per-stage active/cache/peak MLX memory and stage elapsed time.
- [ ] Verify cache clearing occurs only at lifecycle boundaries, not in hot
      transformer loops.

## P7 — Python API

- [ ] Stabilize `LTXPipeline.from_pretrained` and generation arguments.
- [ ] Provide an in-process media result with `save()`.
- [ ] Verify no PyTorch, CUDA, daemon or network server is required on Mac.

## P8 — CLI

- [ ] Implement `lara-ltx generate` strictly as a Python API wrapper.
- [ ] Verify clean installation, helpful localized failures and video output.

## P9 — Optional ComfyUI adapter

- [ ] Create the separate thin `ComfyUI-LaraLTX` package.
- [ ] Add loader, text-to-video and decode/output nodes.
- [ ] Add image-to-video only after core image-conditioning parity.
- [ ] Supply tested example workflows.
- [ ] Verify the plugin contains no model, VAE, scheduler, mapping or Metal copy.

## P10 — GitHub and Hugging Face release

- [ ] Publish the verified source, tests, documentation and tagged release to
      GitHub.
- [ ] Publish the verified MLX BF16 package and loading metadata.
- [ ] Put Quick Start and the three official usage paths at the top.
- [ ] State that no server/CUDA is required on Mac and ComfyUI is optional.
- [ ] Publish only measured supported-hardware, benchmark and quality results.
- [ ] Verify every README command from a clean environment.
- [ ] Keep the GitHub tag, Hugging Face revision and compatibility metadata
      mutually traceable.
- [ ] Verify both public repository pages and downloadable artifacts after
      publication.
- [ ] Complete the release acceptance checklist.

## Permanently out of scope

- [ ] Do not add an HTTP/REST/OpenAI-compatible server or daemon.
- [ ] Do not add multi-user serving, continuous batching or a job backend.
- [ ] Do not add accounts, authentication, database, rate limiting or SaaS.
- [ ] Do not add a proprietary Web UI or cloud-serving stack.
- [ ] Do not implement a second PyTorch/MPS model path for ComfyUI.
- [ ] Do not add per-layer disk swapping, multi-Mac/distributed inference, or
      specialized upstream application pipelines to the v1 critical path.
