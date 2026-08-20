"""Checkpoint-compatible video transformer blocks for the MLX LTX runtime."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError

from .adaln import ada_zero, get_ada_values, post_self_attention
from .attention import Attention
from .layers import DEFAULT_NORM_EPSILON, FeedForward
from .rope import LTXRopeType

ADALN_SELF_ATTENTION = slice(0, 3)
ADALN_FEED_FORWARD = slice(3, 6)
VIDEO_ADALN_PARAMETER_COUNT = 6
VIDEO_CROSS_ADALN_PARAMETER_COUNT = 9
VIDEO_CROSS_ADALN_SLICE = slice(6, 9)
PROMPT_ADALN_PARAMETER_COUNT = 2
PROMPT_SHIFT_INDEX = 0
PROMPT_SCALE_INDEX = 1
AV_CROSS_SCALE_PARAMETER_COUNT = 4
AV_CROSS_GATE_PARAMETER_COUNT = 1
AV_CROSS_TABLE_PARAMETER_COUNT = AV_CROSS_SCALE_PARAMETER_COUNT + AV_CROSS_GATE_PARAMETER_COUNT


@dataclass(frozen=True)
class VideoTransformerConfig:
    """Dimensions for the video-only subset of one upstream AV transformer block."""

    dim: int
    heads: int
    head_dim: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False
    ff_bias: bool = True


class VideoTransformerBlock(nn.Module):
    """Self-attention, text cross-attention and AdaLN-gated FFN for video tokens."""

    def __init__(
        self,
        config: VideoTransformerConfig,
        *,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = DEFAULT_NORM_EPSILON,
    ) -> None:
        super().__init__()
        self.attn1 = Attention(
            query_dim=config.dim,
            heads=config.heads,
            dim_head=config.head_dim,
            rope_type=rope_type,
            norm_eps=norm_eps,
            apply_gated_attention=config.apply_gated_attention,
        )
        self.attn2 = Attention(
            query_dim=config.dim,
            context_dim=config.context_dim,
            heads=config.heads,
            dim_head=config.head_dim,
            rope_type=rope_type,
            norm_eps=norm_eps,
            apply_gated_attention=config.apply_gated_attention,
        )
        self.ff = FeedForward(config.dim, dim_out=config.dim, bias=config.ff_bias)
        self.cross_attention_adaln = config.cross_attention_adaln
        adaln_parameter_count = (
            VIDEO_CROSS_ADALN_PARAMETER_COUNT if config.cross_attention_adaln else VIDEO_ADALN_PARAMETER_COUNT
        )
        self.scale_shift_table = mx.zeros((adaln_parameter_count, config.dim))
        if config.cross_attention_adaln:
            self.prompt_scale_shift_table = mx.zeros((PROMPT_ADALN_PARAMETER_COUNT, config.dim))
        self.norm_eps = norm_eps

    def __call__(
        self,
        x: mx.array,
        *,
        context: mx.array,
        timesteps: mx.array,
        positional_embeddings: tuple[mx.array, mx.array] | None = None,
        context_mask: mx.array | None = None,
        prompt_timestep: mx.array | None = None,
        self_attention_mask: mx.array | None = None,
        self_attn_perturbation_mask: mx.array | None = None,
        self_attn_all_perturbed: bool = False,
    ) -> mx.array:
        """Apply the upstream video block, optionally with text cross-AdaLN."""

        shift_msa, scale_msa, gate_msa = get_ada_values(
            self.scale_shift_table,
            timesteps,
            ADALN_SELF_ATTENTION,
        )
        normalized = ada_zero(x, scale_msa, shift_msa, eps=self.norm_eps)
        attention_output = self.attn1(
            normalized,
            pe=positional_embeddings,
            mask=self_attention_mask,
            perturbation_mask=self_attn_perturbation_mask,
            all_perturbed=self_attn_all_perturbed,
        )
        updated, post_attention_normalized = post_self_attention(
            x,
            attention_output,
            gate_msa,
            eps=self.norm_eps,
        )
        if self.cross_attention_adaln:
            shift_query, scale_query, gate = get_ada_values(
                self.scale_shift_table,
                timesteps,
                VIDEO_CROSS_ADALN_SLICE,
            )
            prompt_modulation = self.prompt_scale_shift_table.astype(context.dtype)[None, None, :, :]
            if prompt_timestep is not None:
                prompt_modulation = prompt_modulation.astype(prompt_timestep.dtype)
                batch_size = prompt_timestep.shape[0]
                prompt_modulation = prompt_modulation + prompt_timestep.reshape(
                    batch_size,
                    prompt_timestep.shape[1],
                    PROMPT_ADALN_PARAMETER_COUNT,
                    -1,
                )
            prompt_shift = prompt_modulation[:, :, PROMPT_SHIFT_INDEX, :]
            prompt_scale = prompt_modulation[:, :, PROMPT_SCALE_INDEX, :]
            text_attention = self.attn2(
                post_attention_normalized * (1 + scale_query) + shift_query,
                context=context * (1 + prompt_scale) + prompt_shift,
                mask=context_mask,
            )
            updated = updated + text_attention * gate
        else:
            updated = updated + self.attn2(post_attention_normalized, context=context, mask=context_mask)

        shift_mlp, scale_mlp, gate_mlp = get_ada_values(
            self.scale_shift_table,
            timesteps,
            ADALN_FEED_FORWARD,
        )
        ff_input = ada_zero(updated, scale_mlp, shift_mlp, eps=self.norm_eps)
        return updated + self.ff(ff_input) * gate_mlp


@dataclass(frozen=True)
class TransformerStream:
    """Inputs for one modality in an AV transformer block.

    The fields use the checkpoint architecture's terms so a sampler can pass
    preprocessed video and audio streams without reinterpreting dimensions.
    """

    x: mx.array
    context: mx.array
    timesteps: mx.array
    prompt_timestep: mx.array | None = None
    cross_scale_shift_timestep: mx.array | None = None
    cross_gate_timestep: mx.array | None = None
    positional_embeddings: tuple[mx.array, mx.array] | None = None
    cross_positional_embeddings: tuple[mx.array, mx.array] | None = None
    context_mask: mx.array | None = None
    self_attention_mask: mx.array | None = None
    self_attn_perturbation_mask: mx.array | None = None
    self_attn_all_perturbed: bool = False
    cross_attn_perturbation_mask: mx.array | None = None
    cross_attn_skip_all: bool = False
    enabled: bool = True


class AVTransformerBlock(nn.Module):
    """Audio/video transformer block with bidirectional cross-attention.

    This implements the non-cross-AdaLN architecture. The constructor accepts
    either stream independently; bidirectional attention is active only when
    both corresponding inputs are enabled and nonempty.
    """

    def __init__(
        self,
        *,
        video: VideoTransformerConfig | None = None,
        audio: VideoTransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = DEFAULT_NORM_EPSILON,
    ) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.video_enabled = video is not None
        self.audio_enabled = audio is not None
        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim, bias=video.ff_bias)
            video_adaln_count = (
                VIDEO_CROSS_ADALN_PARAMETER_COUNT if video.cross_attention_adaln else VIDEO_ADALN_PARAMETER_COUNT
            )
            self.scale_shift_table = mx.zeros((video_adaln_count, video.dim))
            if self.cross_attention_adaln:
                self.prompt_scale_shift_table = mx.zeros((PROMPT_ADALN_PARAMETER_COUNT, video.dim))
        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim, bias=audio.ff_bias)
            audio_adaln_count = (
                VIDEO_CROSS_ADALN_PARAMETER_COUNT if audio.cross_attention_adaln else VIDEO_ADALN_PARAMETER_COUNT
            )
            self.audio_scale_shift_table = mx.zeros((audio_adaln_count, audio.dim))
            if self.cross_attention_adaln:
                self.audio_prompt_scale_shift_table = mx.zeros((PROMPT_ADALN_PARAMETER_COUNT, audio.dim))
        if audio is not None and video is not None:
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.head_dim,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.scale_shift_table_a2v_ca_audio = mx.zeros((AV_CROSS_TABLE_PARAMETER_COUNT, audio.dim))
            self.scale_shift_table_a2v_ca_video = mx.zeros((AV_CROSS_TABLE_PARAMETER_COUNT, video.dim))

    def _run_stream_before_ff(
        self,
        stream: TransformerStream,
        *,
        attn1: Attention,
        attn2: Attention,
        scale_shift_table: mx.array,
        prompt_scale_shift_table: mx.array | None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa = get_ada_values(scale_shift_table, stream.timesteps, ADALN_SELF_ATTENTION)
        normalized = ada_zero(stream.x, scale_msa, shift_msa, eps=self.norm_eps)
        attention_output = attn1(
            normalized,
            pe=stream.positional_embeddings,
            mask=stream.self_attention_mask,
            perturbation_mask=stream.self_attn_perturbation_mask,
            all_perturbed=stream.self_attn_all_perturbed,
        )
        updated, post_attention_normalized = post_self_attention(
            stream.x,
            attention_output,
            gate_msa,
            eps=self.norm_eps,
        )
        if self.cross_attention_adaln:
            if prompt_scale_shift_table is None:
                raise LaraError("LARA-TENSOR-008", details={"stream": "prompt"})
            shift_query, scale_query, gate = get_ada_values(
                scale_shift_table,
                stream.timesteps,
                VIDEO_CROSS_ADALN_SLICE,
            )
            prompt_modulation = prompt_scale_shift_table.astype(stream.context.dtype)[None, None, :, :]
            if stream.prompt_timestep is not None:
                prompt_modulation = prompt_modulation.astype(stream.prompt_timestep.dtype)
                batch_size = stream.prompt_timestep.shape[0]
                prompt_modulation = prompt_modulation + stream.prompt_timestep.reshape(
                    batch_size,
                    stream.prompt_timestep.shape[1],
                    PROMPT_ADALN_PARAMETER_COUNT,
                    -1,
                )
            prompt_shift = prompt_modulation[:, :, PROMPT_SHIFT_INDEX, :]
            prompt_scale = prompt_modulation[:, :, PROMPT_SCALE_INDEX, :]
            text_attention = attn2(
                post_attention_normalized * (1 + scale_query) + shift_query,
                context=stream.context * (1 + prompt_scale) + prompt_shift,
                mask=stream.context_mask,
            )
            updated = updated + text_attention * gate
        else:
            updated = updated + attn2(post_attention_normalized, context=stream.context, mask=stream.context_mask)
        return updated

    def _run_feed_forward(
        self,
        x: mx.array,
        stream: TransformerStream,
        *,
        ff: FeedForward,
        scale_shift_table: mx.array,
    ) -> mx.array:
        shift_mlp, scale_mlp, gate_mlp = get_ada_values(scale_shift_table, stream.timesteps, ADALN_FEED_FORWARD)
        return x + ff(ada_zero(x, scale_mlp, shift_mlp, eps=self.norm_eps)) * gate_mlp

    def _cross_ada_values(
        self,
        table: mx.array,
        scale_shift_timestep: mx.array | None,
        gate_timestep: mx.array | None,
        indices: slice,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if scale_shift_timestep is None or gate_timestep is None:
            raise LaraError("LARA-TENSOR-008", details={"stream": "audio/video"})
        scale, shift = get_ada_values(table[:AV_CROSS_SCALE_PARAMETER_COUNT], scale_shift_timestep, indices)
        (gate,) = get_ada_values(table[AV_CROSS_SCALE_PARAMETER_COUNT:], gate_timestep, slice(None))
        return scale, shift, gate

    @staticmethod
    def _cross_mask(stream: TransformerStream, output: mx.array) -> mx.array:
        return output if stream.cross_attn_perturbation_mask is None else output * stream.cross_attn_perturbation_mask

    def __call__(
        self,
        video: TransformerStream | None,
        audio: TransformerStream | None,
    ) -> tuple[mx.array | None, mx.array | None]:
        """Run independently enabled streams, then their simultaneous AV exchanges."""

        run_video = video is not None and video.enabled and video.x.shape[1] > 0
        run_audio = audio is not None and audio.enabled and audio.x.shape[1] > 0
        video_output = (
            self._run_stream_before_ff(
                video,
                attn1=self.attn1,
                attn2=self.attn2,
                scale_shift_table=self.scale_shift_table,
                prompt_scale_shift_table=getattr(self, "prompt_scale_shift_table", None),
            )
            if run_video
            else (video.x if video is not None else None)
        )
        audio_output = (
            self._run_stream_before_ff(
                audio,
                attn1=self.audio_attn1,
                attn2=self.audio_attn2,
                scale_shift_table=self.audio_scale_shift_table,
                prompt_scale_shift_table=getattr(self, "audio_prompt_scale_shift_table", None),
            )
            if run_audio
            else (audio.x if audio is not None else None)
        )
        if not (run_video and run_audio):
            return (
                self._run_feed_forward(
                    video_output,
                    video,
                    ff=self.ff,
                    scale_shift_table=self.scale_shift_table,
                )
                if run_video
                else video_output,
                self._run_feed_forward(
                    audio_output,
                    audio,
                    ff=self.audio_ff,
                    scale_shift_table=self.audio_scale_shift_table,
                )
                if run_audio
                else audio_output,
            )

        assert video is not None and audio is not None and video_output is not None and audio_output is not None
        video_before_cross, audio_before_cross = video_output, audio_output
        if not video.cross_attn_skip_all:
            video_scale, video_shift, video_gate = self._cross_ada_values(
                self.scale_shift_table_a2v_ca_video,
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
                slice(0, 2),
            )
            audio_scale, audio_shift, _ = self._cross_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(0, 2),
            )
            a2v = self.audio_to_video_attn(
                ada_zero(video_before_cross, video_scale, video_shift, eps=self.norm_eps),
                context=ada_zero(audio_before_cross, audio_scale, audio_shift, eps=self.norm_eps),
                pe=video.cross_positional_embeddings,
                k_pe=audio.cross_positional_embeddings,
            )
            video_output = video_before_cross + self._cross_mask(video, a2v * video_gate)
        if not audio.cross_attn_skip_all:
            audio_scale, audio_shift, audio_gate = self._cross_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(2, 4),
            )
            video_scale, video_shift, _ = self._cross_ada_values(
                self.scale_shift_table_a2v_ca_video,
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
                slice(2, 4),
            )
            v2a = self.video_to_audio_attn(
                ada_zero(audio_before_cross, audio_scale, audio_shift, eps=self.norm_eps),
                context=ada_zero(video_before_cross, video_scale, video_shift, eps=self.norm_eps),
                pe=audio.cross_positional_embeddings,
                k_pe=video.cross_positional_embeddings,
            )
            audio_output = audio_before_cross + self._cross_mask(audio, v2a * audio_gate)
        return (
            self._run_feed_forward(
                video_output,
                video,
                ff=self.ff,
                scale_shift_table=self.scale_shift_table,
            ),
            self._run_feed_forward(
                audio_output,
                audio,
                ff=self.audio_ff,
                scale_shift_table=self.audio_scale_shift_table,
            ),
        )
