# Parity plan

## Comparison contract

CUDA and MLX receive identical serialized inputs. RNG parity and computation
parity are reported separately. Every tensor report records:

- shape and dtype;
- finite, NaN and Inf counts;
- minimum, maximum, mean and standard deviation;
- maximum and mean absolute error;
- relative error and RMSE;
- reference RMS and normalized RMSE;
- cosine similarity;
- content checksum.

## Checkpoints

Compare primitive operators, mapped BF16 weights, text embeddings, initial
latents, a rotating representative block set and every block around the first
detected divergence, stage-one sampler steps,
upscaled latents, stage-two sampler steps, decoder inputs/outputs, decoded frames
and decoded audio. Store deterministic inputs independently from backend RNGs.
Component-specific tolerances remain unset until the first CUDA baseline is
captured; they must not be weakened merely to make a failing test pass.

For checkpoint-backed cross-backend BF16 components whose output scales differ
materially, a single absolute-error threshold is not a valid acceptance test.
The frozen component gate requires finite output, normalized RMSE at most
`2e-2`, and cosine similarity at least `0.9999`; maximum absolute error remains
reported for diagnosis. These criteria were selected after confirming the MLX
operation order and mappings against the pinned upstream CUDA source, and apply
uniformly to both video and audio streams.

The primitive baseline is now captured on the pinned Blackwell CUDA
environment in `golden/cuda/primitive_bf16_cuda.npz`. It covers split RoPE,
RMSNorm, tanh GELU and upstream eager 3D neighborhood attention. The primitive
BF16 tolerance is `2e-2` absolute and relative; checkpoint-backed component
tolerances still require measured evidence.

The first full official reference run is preserved as
`golden/cuda_hq_fixed_seed.mp4`, with its invoked command and output checksum
in `golden/cuda_hq_fixed_seed.json` and independently probed H.264/AAC stream
metadata in `golden/cuda_hq_fixed_seed_media.json`. It is a small fixed-seed
P0 smoke baseline (512x320, 17 frames), not yet an end-to-end MLX parity claim.

`golden/cuda_hq_boundary_trace.npz` and its repeated fixed-seed trace record
the CUDA boundary contract: prompt contexts, sigma schedules, both stage
latents, upscaled latent, video decoder input and decoded video chunk are
bitwise equal. The audio vocoder has a measured small BF16-level variation, so
`golden/manifests/cuda_hq_determinism.json` freezes its measured tolerance and
explicitly rejects final-MP4 byte equality as a quality or parity criterion.

`golden/cuda_audio_decode_reference.npz` isolates that final audio boundary
using the fixed P0 audio latent. It preserves the Audio VAE decoded
spectrogram, primary-vocoder waveform, BWE mel input, BWE residual and final
waveform. Its final waveform differs from the original P0 trace by at most
0.0009765625, exactly the documented CUDA BF16 vocoder tolerance. The
production MLX path must compare these intermediate boundaries rather than
only the final waveform.

`golden/cuda_spatial_upscaler_reference.npz` isolates Video VAE latent
statistics from the x2 spatial-upscaler. It preserves the normalized input,
unnormalized input, unnormalized x2 output and re-normalized output. The final
array is bitwise equal to the matching P0 upscaled-latent boundary.

`golden/cuda_gemma_feature_reference.npz` captures the Gemma V2 LTX feature
path without unnecessarily loading the Gemma language model: fixed hidden
states, attention mask, per-token RMS flattened states, and checkpoint-backed
video/audio projections. The masked token's flattened state is exactly zero.

`golden/mlx_lora_fusion_report.json` records target-Mac execution of the
one-weight-at-a-time distilled-LoRA fusion path. Stage 1 is bitwise equal to
the CUDA BF16 reference; stage 2 differs by at most 3.814697265625e-06, and
the measured fusion peak is 1,050,624 bytes. This closes the local fusion
operation gate without constructing a second transformer state dictionary.

The hash-verified official pack was then exercised on the target M5 Max through
the real selected-range BF16 loader. `golden/mlx_checkpoint_block_0_report.json`
passes the exact 84-key AV block mapping with normalized RMSE
`0.008268845969529879` (video) and `0.005530713899623725` (audio), cosine
similarity above `0.99996`, and a 773,563,888-byte component-load peak.
`golden/mlx_gemma_feature_report.json` passes both official feature projections
with normalized RMSE `0.008588210066130777` and `0.004598081459726736`, with a
2,312,122,440-byte peak. `golden/mlx_transformer_output_report.json` passes the
video/audio final modulation and projection heads with normalized RMSE below
`0.0051` and a 1,622,608-byte peak. This closes the P1 selected-weight value and
bounded-load gate without loading full checkpoints into NumPy or creating a
second full transformer copy.

The v6 block-0 Attention trace captures 408 named CUDA boundaries across all
six video/audio/self/text/cross modules and three guidance passes. Payload
deduplication stores 118 unique tensors in four report-hashed shards and
reconstructs 290 aliases. Direct compact-checkpoint replay is bit-exact with
all six full-pipeline block outputs. On MLX, all 176 exact-input operations pass
the frozen component gate. Computing learned Q/K RMSNorm and RoPE arithmetic in
FP32 before restoring BF16 reduces normalized Q/K RMSNorm error to at most
`1.66e-5` and RoPE-ready error to at most `2.04e-3`. Applying the same source
semantics to global RMSNorm reproduces the captured AdaLN boundary exactly. The
combined full stochastic replay ends at video NRMSE `0.1916` and audio NRMSE
`0.1110`. This remains diagnosis, not final-latent acceptance.

The v7 deep Attention trace restarts from the verified block-23 CUDA output and
replays only blocks 24-39 from compact, hash-verified official BF16 subsets.
All 12 video/audio source outputs at blocks 31 and 39 reproduce exactly. The 11
shards contain 409 unique tensors plus 559 aliases for 968 named boundaries.
On MLX, both block-specific reports pass all 176 exact-input operations: worst
NRMSE is `0.0039552` at block 31 and `0.0039529` at block 39, with cosine above
`0.9999`. The observed full-trajectory divergence is therefore not explained
by a discrete mapping or Attention sub-operation defect at either deep probe.
It remains consistent with accumulated cross-backend BF16 rounding.

The same archive also contains a small, stable-scale BF16 video-only upstream
transformer block (self-attention, text cross-attention and AdaLN-gated FFN),
including every block weight, input and output. Its MLX counterpart passes the
same primitive tolerance. This is an operator-path gate only; real checkpoint
block parity remains required after gated weights are available.

It additionally contains a simultaneous audio/video block case with both
directions of cross-attention, separate scale/shift and gate modulation
timesteps, and every source weight. The MLX non-cross-AdaLN AV block passes the
same `2e-2` BF16 tolerance for both stream outputs. This validates operation
order (self/text attention → simultaneous AV exchange → FFN) but is not a
substitute for checkpoint-backed, cross-AdaLN or conditioning parity.

The archive also captures video text cross-AdaLN: nine video modulation values,
static prompt modulation and a dynamic prompt-timestep input. The MLX path
passes the same provisional BF16 tolerance. The remaining gap is not the
operator sequence; it is loading the official checkpoint weights and producing
the upstream conditioning tensors.

The separate `vae_resnet_bf16_cuda.npz` and
`vae_resnet_group_bf16_cuda.npz` archives capture causal 3D Diffusion VAE
ResNet blocks with PixelNorm and GroupNorm, first-frame causal padding, and
channel-changing shortcut normalization and projection. The MLX implementation
matches both at the provisional BF16 tolerance. Decoder assembly and
checkpoint-backed VAE parity remain required.

The separate `vae_depth_to_space_bf16_cuda.npz` and
`vae_space_to_depth_bf16_cuda.npz` archives capture the causal Diffusion VAE
upsampling and downsampling blocks, including the upstream temporal first-frame
handling and residual paths. The MLX implementation matches both at the
provisional BF16 tolerance.

The `conv_vae_decoder_bf16_cuda.npz` archive covers the assembled non-attention
convolutional decoder subset: latent statistics, reflected spatial causal
convolution, residual-channel reduction, depth-to-space upsampling, PixelNorm,
and final spatial unpatchify. The MLX assembly matches the CUDA BF16 result.

The `vae_attention_bf16_cuda.npz` and
`vae_timestep_resnet_bf16_cuda.npz` archives cover per-frame single-head spatial
attention and the four-way timestep scale/shift ResNet modulation path. Both
MLX paths match their CUDA BF16 references.

The `timestep_embedding_bf16_cuda.npz` and `vae_mid_block_bf16_cuda.npz`
archives extend this through the PixArt sinusoidal/MLP timestep embedder and a
two-layer conditioned VAE mid-block. Both assembled paths match CUDA BF16.

The `diffvae_layers_bf16_cuda.npz`, `diffvae_na_block_bf16_cuda.npz`,
`diffvae_det_stage_bf16_cuda.npz`, and
`diffvae_combined_block_bf16_cuda.npz` archives cover the production Diffusion
VAE channel-last stack: LinearPixelShuffle, AdaLN-Zero, SwiGLU, absolute 3D
RoPE, deterministic NABlocks/stage assembly, and the stage-5 combined context
block. CUDA references use the pinned upstream eager-SDPA NA fallback when the
optional NATTEN wheel is unavailable; all MLX paths match at the recorded BF16
tolerances.

The `diffvae_decoder_bf16_cuda.npz` archive extends that evidence through the
complete untiled decoder orchestration: four deterministic context stages,
spatial patchify/unpatchify, PixArt timestep conditioning, two stage-5
diffusion passes, and both reverse Euler updates. The memory-bounded MLX stage-5
tile path uses receptive-field halos and matches both this CUDA output and the
untiled MLX path. Stage-4 uses the same disjoint-core/halo rule before its
pixel-shuffle upsample. Small latent volumes are edge-expanded to the upstream
minimum stage sizes, receive the trailing-frame NATTEN workaround, and are
cropped back to their requested pixel extent.

The `diffvae_padding_bf16_cuda.npz` archive separately exercises that boundary
path at the production 8x temporal / 32x spatial scale: a one-frame latent is
expanded to the minimum legal NA volume, decoded on its padded working canvas,
and cropped back to the requested 1x32x32 pixels. The MLX production entrypoint
matches the pinned CUDA BF16 output.

The routine corpus stores shape/dtype/statistics/checksums and selected tensors.
It does not permanently duplicate all 48 full-resolution block outputs. On a
failure, bisect tooling captures the smallest full-tensor window needed to find
the first divergent operation.

## Release gate

Numerical parity must be strong enough that end-to-end differences remain
perceptually immaterial across multiple prompts, seeds, durations and
resolutions. Evaluation combines frame metrics, temporal motion metrics, audio
synchronization and blind side-by-side review. Bitwise identity between CUDA and
Metal is not required. A visually plausible output alone never passes the gate.

Parity gates apply to the single in-process MLX pipeline. CLI and ComfyUI tests
verify adapter equivalence to the Python API; they do not define separate model
baselines.
