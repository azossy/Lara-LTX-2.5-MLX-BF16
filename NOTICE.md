# Attribution and Modification Notice

LTX-2.5 and the original LTX-2 software were developed by Lightricks Ltd.
Lara-LTX-2.5-MLX-BF16 is an independent, unofficial derivative and is not
endorsed by or affiliated with Lightricks.

This distribution is subject in its entirety to the
[LTX-2.x Community License Agreement](LICENSE.md), including Section 4 and
Attachment A. Recipients must review and comply with all use restrictions and
the acceptable-use policy incorporated by that agreement. A Commercial Entity,
as defined in Section 2, must obtain the required paid license before using this
derivative commercially. Transfer does not grant rights beyond the agreement.

## Modified implementation

The files in this repository were created or modified for this port. The main
changes from the pinned official implementation are:

- replacement of the PyTorch/CUDA inference path with an Apple Silicon
  MLX/Metal BF16 runtime;
- reviewed checkpoint-key mapping and bounded component loading;
- a native Gemma 4 conditioning path with the official LTX prompt connectors;
- two-stage Res2S sampling, spatial latent refinement, video/audio decoding and
  local MP4 muxing without a serving process;
- direct Python and CLI entry points plus an optional thin ComfyUI adapter; and
- CUDA-versus-MLX parity, quality and memory evidence tooling.

The pinned upstream source commit and official model revision are recorded in
`configs/project.toml`. No Lightricks trademark or endorsement is claimed.
