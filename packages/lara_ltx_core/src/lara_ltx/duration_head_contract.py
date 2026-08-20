"""Pure-Python contract for the standalone LTX duration-head component.

This module deliberately has no MLX import. Checkpoint tooling can validate
the component contract on a non-Apple host before the MLX runtime is available.
"""

from __future__ import annotations

from dataclasses import dataclass

from lara_ltx.errors import LaraError

DEFAULT_VIDEO_CROSS_ATTENTION_DIM = 4096
DEFAULT_AUDIO_CROSS_ATTENTION_DIM = 2048
DEFAULT_POOLER_HIDDEN_DIM = 256
DEFAULT_NUM_QUERIES = 1
DEFAULT_NUM_POOLER_HEADS = 4
DEFAULT_MLP_HIDDEN_DIM = 256


@dataclass(frozen=True)
class DurationHeadConfig:
    """Checkpoint-compatible dimensions for the MLX DurationHead."""

    video_cross_attention_dim: int = DEFAULT_VIDEO_CROSS_ATTENTION_DIM
    audio_cross_attention_dim: int = DEFAULT_AUDIO_CROSS_ATTENTION_DIM
    pooler_hidden_dim: int = DEFAULT_POOLER_HIDDEN_DIM
    num_queries: int = DEFAULT_NUM_QUERIES
    num_pooler_heads: int = DEFAULT_NUM_POOLER_HEADS
    mlp_hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM

    def __post_init__(self) -> None:
        values = {
            "video_cross_attention_dim": self.video_cross_attention_dim,
            "audio_cross_attention_dim": self.audio_cross_attention_dim,
            "pooler_hidden_dim": self.pooler_hidden_dim,
            "num_queries": self.num_queries,
            "num_pooler_heads": self.num_pooler_heads,
            "mlp_hidden_dim": self.mlp_hidden_dim,
        }
        invalid = next((name for name, value in values.items() if value <= 0), None)
        if invalid is not None or self.pooler_hidden_dim % self.num_pooler_heads:
            raise LaraError(
                "LARA-TENSOR-017",
                details={"field": invalid or "pooler_hidden_dim/num_pooler_heads"},
            )


def duration_head_target_keys() -> frozenset[str]:
    """Return the exact MLX ``load_weights`` keys for the official head."""

    return frozenset(
        {
            "video_input_proj.weight",
            "video_input_proj.bias",
            "video_modality_emb",
            "audio_input_proj.weight",
            "audio_input_proj.bias",
            "audio_modality_emb",
            "attention_pooler.query_tokens",
            "attention_pooler.cross_attn.to_q.weight",
            "attention_pooler.cross_attn.to_q.bias",
            "attention_pooler.cross_attn.to_k.weight",
            "attention_pooler.cross_attn.to_k.bias",
            "attention_pooler.cross_attn.to_v.weight",
            "attention_pooler.cross_attn.to_v.bias",
            "attention_pooler.cross_attn.out_proj.weight",
            "attention_pooler.cross_attn.out_proj.bias",
            "mlp_hidden.weight",
            "mlp_hidden.bias",
            "mlp_out.weight",
            "mlp_out.bias",
        }
    )


def duration_head_target_shapes(config: DurationHeadConfig | None = None) -> dict[str, tuple[int, ...]]:
    """Return expected MLX parameter shapes for a DurationHead configuration."""

    selected = config or DurationHeadConfig()
    hidden = selected.pooler_hidden_dim
    return {
        "video_input_proj.weight": (hidden, selected.video_cross_attention_dim),
        "video_input_proj.bias": (hidden,),
        "video_modality_emb": (hidden,),
        "audio_input_proj.weight": (hidden, selected.audio_cross_attention_dim),
        "audio_input_proj.bias": (hidden,),
        "audio_modality_emb": (hidden,),
        "attention_pooler.query_tokens": (selected.num_queries, hidden),
        "attention_pooler.cross_attn.to_q.weight": (hidden, hidden),
        "attention_pooler.cross_attn.to_q.bias": (hidden,),
        "attention_pooler.cross_attn.to_k.weight": (hidden, hidden),
        "attention_pooler.cross_attn.to_k.bias": (hidden,),
        "attention_pooler.cross_attn.to_v.weight": (hidden, hidden),
        "attention_pooler.cross_attn.to_v.bias": (hidden,),
        "attention_pooler.cross_attn.out_proj.weight": (hidden, hidden),
        "attention_pooler.cross_attn.out_proj.bias": (hidden,),
        "mlp_hidden.weight": (selected.mlp_hidden_dim, hidden * selected.num_queries),
        "mlp_hidden.bias": (selected.mlp_hidden_dim,),
        "mlp_out.weight": (1, selected.mlp_hidden_dim),
        "mlp_out.bias": (1,),
    }
