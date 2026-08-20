# Error codes

| Code | Message key | Cause | User action | Location |
|---|---|---|---|---|
| LARA-GENERAL-001 | `LARA-GENERAL-001` | Unclassified failure | Check logs and retry | Global fallback |
| LARA-CONFIG-001 | `LARA-CONFIG-001` | Missing, unreadable or invalid TOML | Verify path and syntax | `lara_ltx.config` |
| LARA-CONFIG-002 | `LARA-CONFIG-002` | Required setting absent | Restore the key from reference config | `lara_ltx.config` |
| LARA-CONFIG-003 | `LARA-CONFIG-003` | Setting has an invalid range or type | Restore a documented positive/ranged value | `lara_ltx.config` |
| LARA-UPSTREAM-001 | `LARA-UPSTREAM-001` | Source HEAD differs from pinned commit | Fetch and checkout the pinned commit | Upstream freeze tooling |
| LARA-MODEL-001 | `LARA-MODEL-001` | Required checkpoint unavailable | Accept model terms, authenticate and download | Model loader/downloader |
| LARA-MODEL-002 | `LARA-MODEL-002` | Size, dtype or checksum mismatch | Replace only the affected file | Manifest verifier |
| LARA-MODEL-003 | `LARA-MODEL-003` | Invalid or unreadable safetensors header | Replace the affected checkpoint | `lara_ltx.models.checkpoint` |
| LARA-MODEL-004 | `LARA-MODEL-004` | Unsupported checkpoint dtype | Use the official BF16 checkpoint | `lara_ltx.models.checkpoint` |
| LARA-MODEL-005 | `LARA-MODEL-005` | Tensor shape/dtype byte count mismatch | Replace the corrupted checkpoint | `lara_ltx.models.checkpoint` |
| LARA-MODEL-006 | `LARA-MODEL-006` | Tensor payload exceeds file bounds | Replace the truncated checkpoint | `lara_ltx.models.checkpoint` |
| LARA-MODEL-007 | `LARA-MODEL-007` | A requested checkpoint tensor violates the reviewed allowed dtype set | Use the pinned BF16 payload or explicitly reviewed F32 auxiliary tensor; never map serialized tokenizer/config assets | `lara_ltx.models.loading` |
| LARA-MODEL-008 | `LARA-MODEL-008` | Tensor payload byte ranges overlap | Replace the malformed checkpoint | `lara_ltx.models.checkpoint` |
| LARA-MODEL-009 | `LARA-MODEL-009` | Mapping template output failed | Check output path and permissions | Mapping template tool |
| LARA-MODEL-010 | `LARA-MODEL-010` | Required tensor missing from a shard | Verify pinned model revision and mapping manifest | `lara_ltx.models.loading` |
| LARA-MODEL-011 | `LARA-MODEL-011` | MLX-loaded tensor dtype or shape differs from validated safetensors descriptor | Replace the shard or correct mapping | `lara_ltx.models.loading` |
| LARA-MODEL-012 | `LARA-MODEL-012` | A source tensor key appears in more than one shard | Use a non-overlapping official shard set | `lara_ltx.models.mapping` |
| LARA-MODEL-013 | `LARA-MODEL-013` | Hugging Face access probe could not reach the model service | Check network connectivity and retry; the probe downloads no payload | `tools/models/probe_access.py` |
| LARA-MODEL-014 | `LARA-MODEL-014` | Mapping file is unreadable, has the wrong schema, or contains an unassigned target | Assign every reviewed target and retry | `lara_ltx.models.mapping` |
| LARA-MODEL-015 | `LARA-MODEL-015` | Multiple source tensors map to one MLX target | Keep exactly one source for each target | `lara_ltx.models.mapping` |
| LARA-MODEL-016 | `LARA-MODEL-016` | Mapping requests an unsupported transform | Use a documented transform and review its shape | `lara_ltx.models.mapping` |
| LARA-MODEL-017 | `LARA-MODEL-017` | Mapping targets differ from the component parameter keys | Correct missing/unexpected target assignments | `lara_ltx.models.mapping` |
| LARA-MODEL-018 | `LARA-MODEL-018` | One source tensor appears in multiple mapping rules | Keep exactly one reviewed rule per source tensor | `lara_ltx.models.mapping` |
| LARA-MODEL-019 | `LARA-MODEL-019` | A requested matrix transform received a non-2D tensor | Correct the transform or source mapping | `lara_ltx.models.loading` |
| LARA-MODEL-020 | `LARA-MODEL-020` | Transformed tensor shape differs from target component shape | Correct the mapping transform or component configuration | `lara_ltx.models.loading` |
| LARA-MODEL-021 | `LARA-MODEL-021` | A shard referenced by mapping is unavailable | Provide the verified source shard before construction | `lara_ltx.models.loading` |
| LARA-MODEL-022 | `LARA-MODEL-022` | Multiple provided shards have the same filename | Use uniquely named official shards | `lara_ltx.models.loading` |
| LARA-MODEL-023 | `LARA-MODEL-023` | A fused QKV tensor has no axis-zero three-way split | Use the matching official checkpoint revision and QKV mapping rule | `lara_ltx.models.loading` |
| LARA-MODEL-024 | `LARA-MODEL-024` | A static gate cannot be folded into its target linear tensor | Use the matching checkpoint revision and gate mapping rule | `lara_ltx.models.loading` |
| LARA-MODEL-025 | `LARA-MODEL-025` | DurationHead source shape conflicts with the reviewed component layout | Use the pinned official shard and matching manifest | `lara_ltx.models.mapping` |
| LARA-MODEL-026 | `LARA-MODEL-026` | LoRA A/B pair is incomplete or incompatible with a pinned base transformer weight | Use the matching official BF16 adapter and base checkpoint | `lara_ltx.models.lora` |
| LARA-MODEL-027 | `LARA-MODEL-027` | LoRA stage strength is non-finite or negative | Use a finite non-negative stage strength | `lara_ltx.models.lora` |
| LARA-MODEL-028 | `LARA-MODEL-028` | Transformer-block checkpoint entry is missing or incompatible with the reviewed AV layout | Use the pinned 22B BF16 transformer checkpoint | `lara_ltx.models.transformer_block` |
| LARA-MODEL-029 | `LARA-MODEL-029` | Gemma feature-extractor checkpoint entry is missing or incompatible with the reviewed BF16 projection layout | Use the pinned Gemma checkpoint and keep tokenizer assets separate | `lara_ltx.models.gemma_feature` |
| LARA-MODEL-030 | `LARA-MODEL-030` | Latent spatial-upscaler entry is missing or incompatible with the reviewed x2 BF16 layout | Use the pinned official upscaler checkpoint | `lara_ltx.models.spatial_upscaler` |
| LARA-MODEL-031 | `LARA-MODEL-031` | Audio VAE core entry is missing or incompatible with the reviewed BF16 layout | Use the pinned Audio VAE checkpoint; load vocoder/BWE separately | `lara_ltx.models.audio_vae` |
| LARA-MODEL-032 | `LARA-MODEL-032` | Audio vocoder/BWE entry is missing or incompatible with the reviewed BF16 waveform layout | Use the pinned Audio VAE/vocoder checkpoint | `lara_ltx.models.vocoder` |
| LARA-MODEL-033 | `LARA-MODEL-033` | Transformer output entry is missing or incompatible with the reviewed video/audio projection layout | Use the pinned official 22B BF16 transformer checkpoint | `lara_ltx.models.transformer_output` |
| LARA-MODEL-034 | `LARA-MODEL-034` | Transformer input entry is missing or incompatible with the reviewed video/audio conditioning layout | Use the pinned official 22B BF16 transformer checkpoint | `lara_ltx.models.transformer_input` |
| LARA-MODEL-035 | `LARA-MODEL-035` | Packed Gemma 4 text configuration or one of its 666 text-core mappings is incompatible | Use the pinned official BF16 encoder and regenerate the reviewed text-core manifest | `lara_ltx.models.gemma_text` |
| LARA-MODEL-036 | `LARA-MODEL-036` | A packed Gemma tokenizer or processor byte asset is missing, truncated or invalid | Restore the pinned encoder, verify its checksum and retry | `lara_ltx.models.gemma_assets` |
| LARA-PARITY-002 | `LARA-PARITY-002` | Fixed CUDA audio-boundary input/capture key is unavailable or invalid | Restore the approved boundary artifact and retry | `tools/parity/capture_cuda_audio_decode.py` |
| LARA-PARITY-003 | `LARA-PARITY-003` | Fixed CUDA LoRA fusion input/capture key is unavailable or invalid | Restore the approved LoRA boundary artifact and retry | `tools/parity/compare_mlx_lora_fusion.py` |
| LARA-RUNTIME-001 | `LARA-RUNTIME-001` | BF16 workload exceeds memory | Reduce dimensions or use a larger-memory target | Runtime |
| LARA-RUNTIME-002 | `LARA-RUNTIME-002` | CUDA is unavailable | Start a GPU instance and retry | Golden generator |
| LARA-RUNTIME-003 | `LARA-RUNTIME-003` | Metal is unavailable | Use a supported Apple Silicon Mac and pinned MLX environment | Metal feasibility probes |
| LARA-RUNTIME-004 | `LARA-RUNTIME-004` | Estimated probe allocation exceeds configured budget | Reduce shape or free memory | Metal feasibility probes |
| LARA-RUNTIME-005 | `LARA-RUNTIME-005` | Official CUDA golden-pipeline inputs, grid or output are invalid, or the reference run failed | Inspect the captured log, verify the pinned BF16 pack and supported dimensions, then retry | `tools/parity/run_cuda_hq_golden.py` |
| LARA-RUNTIME-006 | `LARA-RUNTIME-006` | CUDA reference media is missing, invalid, or lacks video/audio streams | Inspect the pipeline log, regenerate the artifact and validate it again | `tools/parity/validate_cuda_media.py` |
| LARA-RUNTIME-007 | `LARA-RUNTIME-007` | The pinned native MLX-LM text runtime is not installed | Install locked dependencies on supported Apple Silicon | `lara_ltx.text_encoder.gemma4` |
| LARA-SAMPLING-001 | `LARA-SAMPLING-001` | Scheduler steps, token count, shifts or terminal are invalid | Use the documented positive and finite schedule values | `lara_ltx.sampling.scheduler` |
| LARA-SAMPLING-002 | `LARA-SAMPLING-002` | CFG/STG scales, prediction passes or perturbation indices are invalid | Correct the guidance configuration and required passes | `lara_ltx.sampling.guidance`, `lara_ltx.sampling.perturbations` |
| LARA-SAMPLING-003 | `LARA-SAMPLING-003` | res_2s schedule, modality, eta or SDE step is invalid | Supply a decreasing schedule and valid modality states/sampler settings | `lara_ltx.sampling.res2s` |
| LARA-PARITY-001 | `LARA-PARITY-001` | Compared tensor shapes differ | Verify layout mapping | `lara_ltx.parity` |
| LARA-TENSOR-001 | `LARA-TENSOR-001` | RoPE cosine/sine shapes differ | Rebuild both tensors from one grid | `lara_ltx.transformer.rope` |
| LARA-TENSOR-002 | `LARA-TENSOR-002` | RoPE batch cannot broadcast | Rebuild frequencies for the input batch | `lara_ltx.transformer.rope` |
| LARA-TENSOR-003 | `LARA-TENSOR-003` | Attention head dimension is odd | Use an even head dimension | `lara_ltx.transformer.rope` |
| LARA-TENSOR-004 | `LARA-TENSOR-004` | Unknown RoPE layout | Select split or interleaved | `lara_ltx.transformer.rope` |
| LARA-TENSOR-005 | `LARA-TENSOR-005` | Grid and max-position dimensions differ | Correct position metadata | `lara_ltx.transformer.rope` |
| LARA-TENSOR-006 | `LARA-TENSOR-006` | AdaLN table and timestep shapes conflict | Verify block configuration and embedding | `lara_ltx.transformer.adaln` |
| LARA-TENSOR-007 | `LARA-TENSOR-007` | Invalid 3D neighborhood-attention layout, kernel, causal axes, or mask budget | Verify Q/K/V shape and runtime memory configuration | `lara_ltx.video_vae.neighborhood_attention` |
| LARA-TENSOR-008 | `LARA-TENSOR-008` | AV cross-attention modulation tensors are incomplete | Recreate the stream preprocessing outputs | `lara_ltx.transformer.blocks` |
| LARA-TENSOR-009 | `LARA-TENSOR-009` | Unsupported VAE normalization layer | Select pixel normalization or group normalization | `lara_ltx.video_vae.resnet` |
| LARA-TENSOR-010 | `LARA-TENSOR-010` | VAE depth-to-space stride and channel count are incompatible | Use a stride whose volume divides the input channel count | `lara_ltx.video_vae.sampling` |
| LARA-TENSOR-011 | `LARA-TENSOR-011` | VAE space-to-depth configuration or input dimensions are incompatible | Use valid channel counts, stride, and divisible input dimensions | `lara_ltx.video_vae.sampling` |
| LARA-TENSOR-012 | `LARA-TENSOR-012` | VAE patch size, shape, or channel layout is invalid | Use positive patch sizes that divide the input layout | `lara_ltx.video_vae.ops` |
| LARA-TENSOR-013 | `LARA-TENSOR-013` | VAE decoder block is unsupported or has incompatible channels | Use a reviewed supported block and valid channel multiplier | `lara_ltx.video_vae.conv_decoder` |
| LARA-TENSOR-014 | `LARA-TENSOR-014` | Timestep-conditioned VAE block is missing its embedding | Recreate decoder timestep conditioning and retry | `lara_ltx.video_vae.resnet` |
| LARA-TENSOR-015 | `LARA-TENSOR-015` | Diffusion VAE pixel-shuffle layout is incompatible | Use a valid 3D stride and divisible channel reduction | `lara_ltx.video_vae.diffusion_layers` |
| LARA-TENSOR-016 | `LARA-TENSOR-016` | Diffusion VAE decoder configuration or input layout is invalid | Verify stage dimensions, timestep schedule, and context/noise shapes | `lara_ltx.video_vae.diffusion_decoder` |
| LARA-TENSOR-017 | `LARA-TENSOR-017` | DurationHead configuration is structurally invalid | Use positive dimensions and a hidden dimension divisible by head count | `lara_ltx.duration_head_contract` |
| LARA-TENSOR-018 | `LARA-TENSOR-018` | DurationHead has no usable modality or receives incompatible connector tensors | Provide video and/or audio rank-3 tokens with configured widths and matching batches | `lara_ltx.duration_head` |
| LARA-TENSOR-019 | `LARA-TENSOR-019` | Gemma feature hidden states or attention mask have an invalid shape | Use [batch, tokens, 3840, 49] hidden states and matching rank-2 mask | `lara_ltx.models.gemma_feature_runtime` |
| LARA-TENSOR-020 | `LARA-TENSOR-020` | Transformer output hidden state or embedded timestep has an invalid shape | Use matching rank-3 tensors with the configured hidden width | `lara_ltx.transformer.output` |
| LARA-TENSOR-021 | `LARA-TENSOR-021` | Transformer conditioning layout, sigma or keyframe mask is invalid | Recreate patchified latents, per-token timesteps, position bounds, projected context and masks | `lara_ltx.transformer.input` |
| LARA-TENSOR-022 | `LARA-TENSOR-022` | Gemma prompt, token layout or left-padding mask is invalid | Provide non-empty prompts and matching rank-2 IDs with contiguous left padding | `lara_ltx.models.gemma_assets`, `lara_ltx.text_encoder.gemma4` |
| LARA-TENSOR-023 | `LARA-TENSOR-023` | Spatial-upscaler configuration or NCTHW latent layout is invalid | Use the reviewed x2 configuration and a rank-5 128-channel latent | `lara_ltx.video_vae.spatial_upscaler` |
| LARA-TENSOR-024 | `LARA-TENSOR-024` | Audio VAE decoder configuration or rank-4 latent layout is invalid | Use the pinned causal decoder configuration and an 8-channel latent | `lara_ltx.audio_vae.decoder` |
| LARA-TENSOR-025 | `LARA-TENSOR-025` | Vocoder mel input is not rank-4 stereo with 64 mel bins | Supply `[batch, 2, frames, 64]` | `lara_ltx.audio_vae.vocoder` |
| LARA-TENSOR-026 | `LARA-TENSOR-026` | BWE input is not a rank-3 stereo waveform | Supply `[batch, 2, samples]` | `lara_ltx.audio_vae.vocoder` |
| LARA-TENSOR-027 | `LARA-TENSOR-027` | BWE residual and sinc-resampled skip lengths differ | Restore pinned sample rates, hop length and checkpoint config | `lara_ltx.audio_vae.vocoder` |
| LARA-TENSOR-028 | `LARA-TENSOR-028` | A transformer block received no active modality | Enable a valid video and/or audio stream before denoising | `lara_ltx.transformer.runtime` |
