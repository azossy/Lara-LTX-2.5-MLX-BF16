# Lara-LTX 0.0.4 deep parity hardening preview

Lara-LTX 0.0.4 completes the bounded block-31/block-39 CUDA diagnostic that
remained open in 0.0.3. It does not weaken the frozen final-latent threshold or
claim bit-identical CUDA/MLX generation.

## Verified in this release

- Replayed CUDA transformer blocks 24-39 from the hash-verified block-23
  boundary using compact official BF16 transformer and distilled-LoRA subsets.
- Reproduced all 12 captured CUDA video/audio outputs at blocks 31 and 39
  exactly across the three guidance passes.
- Captured 968 named Attention boundaries as 409 unique tensors plus 559
  aliases in 11 SHA-256-verified shards totaling 377,930,707 bytes.
- Passed 176 exact-input MLX Attention comparisons at block 31 and another 176
  at block 39. Worst NRMSE was `0.00396` and `0.00395`, below the frozen `0.02`
  maximum; minimum cosine remained above `0.9999`.
- Added an exclusive downloader lock so concurrent resumable subset writers
  fail clearly instead of corrupting shared range parts.
- Re-ran the complete local test suite: 267 tests pass.

## Interpretation

The new deep traces rule out a discrete checkpoint mapping or Attention
sub-operation defect at the two observed trajectory divergence points. The
remaining same-seed difference is consistent with accumulated CUDA-versus-MLX
BF16 backend rounding. A broad FP32 sigmoid/gate change remains rejected because
it regressed the full trajectory. The engineering-preview limitation and the
independent blind-review gate therefore remain explicit.
