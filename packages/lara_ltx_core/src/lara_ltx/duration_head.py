"""MLX implementation of the standalone LTX-2.5 duration regression head."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.duration_head_contract import DurationHeadConfig
from lara_ltx.errors import LaraError
from lara_ltx.transformer.attention import scaled_dot_product_attention
from lara_ltx.transformer.layers import gelu_approx


class CrossAttentionPooler(nn.Module):
    """Separate-QKV MLX counterpart to PyTorch ``MultiheadAttention``."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, queries: mx.array, tokens: mx.array) -> mx.array:
        attended = scaled_dot_product_attention(
            self.to_q(queries),
            self.to_k(tokens),
            self.to_v(tokens),
            self.heads,
        )
        return self.out_proj(attended)


class AttentionPooler(nn.Module):
    """Cross-attend learnable query tokens against a variable token sequence."""

    def __init__(self, config: DurationHeadConfig) -> None:
        super().__init__()
        self.hidden_dim = config.pooler_hidden_dim
        self.num_queries = config.num_queries
        self.query_tokens = mx.random.normal((self.num_queries, self.hidden_dim)) * 0.02
        self.cross_attn = CrossAttentionPooler(self.hidden_dim, config.num_pooler_heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        batch_size = tokens.shape[0]
        queries = mx.broadcast_to(
            mx.expand_dims(self.query_tokens, axis=0),
            (batch_size, self.num_queries, self.hidden_dim),
        )
        return self.cross_attn(queries, tokens)


class DurationHead(nn.Module):
    """Predict shot duration in seconds from video and/or audio connector tokens."""

    def __init__(self, config: DurationHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or DurationHeadConfig()
        self.video_input_proj = nn.Linear(
            self.config.video_cross_attention_dim,
            self.config.pooler_hidden_dim,
            bias=True,
        )
        self.video_modality_emb = mx.random.normal((self.config.pooler_hidden_dim,)) * 0.02
        self.audio_input_proj = nn.Linear(
            self.config.audio_cross_attention_dim,
            self.config.pooler_hidden_dim,
            bias=True,
        )
        self.audio_modality_emb = mx.random.normal((self.config.pooler_hidden_dim,)) * 0.02
        self.attention_pooler = AttentionPooler(self.config)
        self.mlp_hidden = nn.Linear(
            self.config.pooler_hidden_dim * self.config.num_queries,
            self.config.mlp_hidden_dim,
            bias=True,
        )
        self.mlp_out = nn.Linear(self.config.mlp_hidden_dim, 1, bias=True)

    def __call__(self, video_tokens: mx.array | None = None, audio_tokens: mx.array | None = None) -> mx.array:
        """Return per-batch duration in seconds, matching the upstream ``exp`` output."""

        if video_tokens is None and audio_tokens is None:
            raise LaraError("LARA-TENSOR-018", details={"reason": "no_modalities"})
        token_groups: list[mx.array] = []
        batch_size: int | None = None
        if video_tokens is not None:
            self._require_tokens(video_tokens, self.config.video_cross_attention_dim, "video", batch_size)
            batch_size = video_tokens.shape[0]
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            self._require_tokens(audio_tokens, self.config.audio_cross_attention_dim, "audio", batch_size)
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)

        tokens = mx.concatenate(token_groups, axis=1)
        pooled = self.attention_pooler(tokens)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        log_duration = self.mlp_out(gelu_approx(self.mlp_hidden(pooled_flat))).squeeze(-1)
        return mx.exp(log_duration)

    @staticmethod
    def _require_tokens(tokens: mx.array, expected_features: int, modality: str, batch_size: int | None) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != expected_features:
            raise LaraError(
                "LARA-TENSOR-018",
                details={"reason": f"{modality}_shape={tuple(tokens.shape)}"},
            )
        if batch_size is not None and tokens.shape[0] != batch_size:
            raise LaraError(
                "LARA-TENSOR-018",
                details={"reason": f"batch_mismatch={batch_size}/{tokens.shape[0]}"},
            )
