"""Video VAE operations required by the BF16 MLX port."""

from .attention import AttnBlock3D
from .conv_decoder import ConvVideoDecoder, ConvVideoDecoderConfig, DecoderBlockConfig
from .diffusion_blocks import (
    CombinedDiffusionNABlock,
    CombinedDiffusionNABlockConfig,
    DeterministicStage,
    DeterministicStageConfig,
    NABlock,
    NABlockConfig,
    NeighborhoodAttention3D,
    NeighborhoodAttention3DConfig,
)
from .diffusion_decoder import (
    DiffusionVideoDecoder,
    DiffusionVideoDecoderConfig,
    RecommendedDecoderTiling,
    Stage4TilingConfig,
    Stage5TilingConfig,
)
from .diffusion_layers import AdaLNZero, LinearPixelShuffleUpsample, LinearPixelShuffleUpsampleConfig, SwiGLU
from .neighborhood_attention import neighborhood_attention_3d
from .ops import PerChannelStatistics, patchify, unpatchify
from .resnet import CausalConv3d, ResnetBlock3D, ResnetBlock3DConfig, UNetMidBlock3D, UNetMidBlock3DConfig
from .sampling import (
    DepthToSpaceUpsample,
    DepthToSpaceUpsampleConfig,
    SpaceToDepthDownsample,
    SpaceToDepthDownsampleConfig,
)

__all__ = [
    "AdaLNZero",
    "AttnBlock3D",
    "CausalConv3d",
    "CombinedDiffusionNABlock",
    "CombinedDiffusionNABlockConfig",
    "ConvVideoDecoder",
    "ConvVideoDecoderConfig",
    "DecoderBlockConfig",
    "DepthToSpaceUpsample",
    "DepthToSpaceUpsampleConfig",
    "DeterministicStage",
    "DeterministicStageConfig",
    "DiffusionVideoDecoder",
    "DiffusionVideoDecoderConfig",
    "LinearPixelShuffleUpsample",
    "LinearPixelShuffleUpsampleConfig",
    "NABlock",
    "NABlockConfig",
    "NeighborhoodAttention3D",
    "NeighborhoodAttention3DConfig",
    "PerChannelStatistics",
    "RecommendedDecoderTiling",
    "ResnetBlock3D",
    "ResnetBlock3DConfig",
    "SpaceToDepthDownsample",
    "SpaceToDepthDownsampleConfig",
    "Stage4TilingConfig",
    "Stage5TilingConfig",
    "SwiGLU",
    "UNetMidBlock3D",
    "UNetMidBlock3DConfig",
    "neighborhood_attention_3d",
    "patchify",
    "unpatchify",
]
