"""Checkpoint-compatible transformer input projection and conditioning preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError

from .blocks import TransformerStream
from .rope import DEFAULT_ROPE_THETA, precompute_freqs_cis
from .timestep import PixArtAlphaCombinedTimestepSizeEmbeddings

DEFAULT_TIMESTEP_SCALE_MULTIPLIER = 1_000.0
DEFAULT_AV_CROSS_TIMESTEP_SCALE_MULTIPLIER = 1_000.0
DEFAULT_CROSS_POSITION_MAXIMUM = 20


@dataclass(frozen=True)
class TransformerInputConfig:
    input_channels: int
    hidden_dimension: int
    adaln_coefficient: int
    attention_heads: int
    max_positions: tuple[int, ...]
    timestep_scale_multiplier: float = DEFAULT_TIMESTEP_SCALE_MULTIPLIER
    positional_embedding_theta: float = DEFAULT_ROPE_THETA
    use_middle_indices_grid: bool = True
    prompt_adaln_coefficient: int | None = None
    use_keyframes_absolute_embedding: bool = False


@dataclass(frozen=True)
class PreparedTransformerInput:
    stream: TransformerStream
    embedded_timestep: mx.array


@dataclass(frozen=True)
class TransformerModalityInput:
    latent: mx.array
    sigma: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: mx.array | None = None
    attention_mask: mx.array | None = None
    keyframes_mask: mx.array | None = None


class AdaLayerNormSingle(nn.Module):
    """Upstream PixArt timestep MLP followed by one AdaLN projection."""

    def __init__(self, hidden_dimension: int, coefficient: int) -> None:
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(hidden_dimension)
        self.linear = nn.Linear(hidden_dimension, coefficient * hidden_dimension, bias=True)

    def __call__(self, timestep: mx.array, *, hidden_dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        activated = embedded_timestep * mx.sigmoid(embedded_timestep)
        return self.linear(activated), embedded_timestep


class TransformerInputPreprocessor(nn.Module):
    """Prepare one already-patchified modality for the AV transformer blocks."""

    def __init__(self, config: TransformerInputConfig) -> None:
        super().__init__()
        self.config = config
        self.patchify_proj = nn.Linear(config.input_channels, config.hidden_dimension, bias=True)
        self.adaln_single = AdaLayerNormSingle(config.hidden_dimension, config.adaln_coefficient)
        if config.prompt_adaln_coefficient is not None:
            self.prompt_adaln_single = AdaLayerNormSingle(
                config.hidden_dimension,
                config.prompt_adaln_coefficient,
            )
        if config.use_keyframes_absolute_embedding:
            self.keyframes_abs_pos_embedding = mx.zeros((1, config.hidden_dimension))

    def _prepare_timestep(
        self,
        timestep: mx.array,
        adaln: AdaLayerNormSingle,
        *,
        batch_size: int,
        hidden_dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        projected, embedded = adaln(
            (timestep * self.config.timestep_scale_multiplier).reshape(-1),
            hidden_dtype=hidden_dtype,
        )
        return (
            projected.reshape(batch_size, -1, projected.shape[-1]),
            embedded.reshape(batch_size, -1, embedded.shape[-1]),
        )

    @staticmethod
    def _prepare_context_mask(mask: mx.array | None, hidden_dtype: mx.Dtype) -> mx.array | None:
        if mask is None or mx.issubdtype(mask.dtype, mx.floating):
            return mask
        return ((mask - 1).astype(hidden_dtype) * mx.finfo(hidden_dtype).max).reshape(
            mask.shape[0],
            1,
            -1,
            mask.shape[-1],
        )

    @staticmethod
    def _prepare_self_attention_mask(mask: mx.array | None, hidden_dtype: mx.Dtype) -> mx.array | None:
        if mask is None:
            return None
        limits = mx.finfo(hidden_dtype)
        positive = mask > 0
        logarithm = mx.log(mx.maximum(mask.astype(mx.float32), limits.smallest_normal)).astype(hidden_dtype)
        bias = mx.where(positive, logarithm, mx.array(limits.min, dtype=hidden_dtype))
        return mx.expand_dims(bias, 1)

    def _apply_keyframe_embedding(self, hidden: mx.array, keyframes_mask: mx.array | None) -> mx.array:
        embedding = getattr(self, "keyframes_abs_pos_embedding", None)
        if embedding is None or keyframes_mask is None:
            return hidden
        if keyframes_mask.shape != (*hidden.shape[:2], 1):
            raise LaraError("LARA-TENSOR-021", details={"reason": "invalid_keyframes_mask"})
        return hidden + (keyframes_mask > 0).astype(hidden.dtype) * embedding.astype(hidden.dtype)

    def prepare(
        self,
        *,
        latent: mx.array,
        timesteps: mx.array,
        positions: mx.array,
        context: mx.array,
        sigma: mx.array | None = None,
        context_mask: mx.array | None = None,
        attention_mask: mx.array | None = None,
        keyframes_mask: mx.array | None = None,
        enabled: bool = True,
    ) -> PreparedTransformerInput:
        if (
            latent.ndim != 3
            or latent.shape[-1] != self.config.input_channels
            or timesteps.ndim != 2
            or timesteps.shape != latent.shape[:2]
            or context.ndim != 3
            or context.shape[0] != latent.shape[0]
            or context.shape[-1] != self.config.hidden_dimension
        ):
            raise LaraError("LARA-TENSOR-021", details={"reason": "invalid_modality_layout"})
        hidden = self._apply_keyframe_embedding(self.patchify_proj(latent), keyframes_mask)
        batch_size = hidden.shape[0]
        timestep, embedded_timestep = self._prepare_timestep(
            timesteps,
            self.adaln_single,
            batch_size=batch_size,
            hidden_dtype=latent.dtype,
        )
        prompt_timestep = None
        prompt_adaln = getattr(self, "prompt_adaln_single", None)
        if prompt_adaln is not None:
            if sigma is None or sigma.ndim != 1 or sigma.shape[0] != batch_size:
                raise LaraError("LARA-TENSOR-021", details={"reason": "invalid_sigma"})
            prompt_timestep, _ = self._prepare_timestep(
                sigma,
                prompt_adaln,
                batch_size=batch_size,
                hidden_dtype=latent.dtype,
            )
        positional_embeddings = precompute_freqs_cis(
            positions,
            self.config.hidden_dimension,
            theta=self.config.positional_embedding_theta,
            max_positions=self.config.max_positions,
            use_middle_indices_grid=self.config.use_middle_indices_grid,
            attention_heads=self.config.attention_heads,
        )
        stream = TransformerStream(
            x=hidden,
            context=context.reshape(batch_size, -1, self.config.hidden_dimension),
            context_mask=self._prepare_context_mask(context_mask, latent.dtype),
            timesteps=timestep,
            prompt_timestep=prompt_timestep,
            positional_embeddings=positional_embeddings,
            self_attention_mask=self._prepare_self_attention_mask(attention_mask, latent.dtype),
            enabled=enabled,
        )
        return PreparedTransformerInput(stream=stream, embedded_timestep=embedded_timestep)

    def prepare_modality(self, modality: TransformerModalityInput) -> PreparedTransformerInput:
        """Prepare the dataclass form used by the multimodal orchestrator."""

        return self.prepare(
            latent=modality.latent,
            sigma=modality.sigma,
            timesteps=modality.timesteps,
            positions=modality.positions,
            context=modality.context,
            enabled=modality.enabled,
            context_mask=modality.context_mask,
            attention_mask=modality.attention_mask,
            keyframes_mask=modality.keyframes_mask,
        )


class AVTransformerInputPreprocessor(nn.Module):
    """Prepare both modalities and their simultaneous AV cross-attention inputs."""

    def __init__(
        self,
        *,
        video: TransformerInputConfig,
        audio: TransformerInputConfig,
        cross_attention_dimension: int,
        cross_position_maximum: int = DEFAULT_CROSS_POSITION_MAXIMUM,
        av_cross_timestep_scale_multiplier: float = DEFAULT_AV_CROSS_TIMESTEP_SCALE_MULTIPLIER,
    ) -> None:
        super().__init__()
        self.video = TransformerInputPreprocessor(video)
        self.audio = TransformerInputPreprocessor(audio)
        self.video_cross_scale_shift = AdaLayerNormSingle(video.hidden_dimension, coefficient=4)
        self.audio_cross_scale_shift = AdaLayerNormSingle(audio.hidden_dimension, coefficient=4)
        self.video_cross_gate = AdaLayerNormSingle(video.hidden_dimension, coefficient=1)
        self.audio_cross_gate = AdaLayerNormSingle(audio.hidden_dimension, coefficient=1)
        self.cross_attention_dimension = cross_attention_dimension
        self.cross_position_maximum = cross_position_maximum
        self.av_cross_timestep_scale_multiplier = av_cross_timestep_scale_multiplier

    @staticmethod
    def _project_cross_timestep(
        adaln: AdaLayerNormSingle,
        timestep: mx.array,
        *,
        multiplier: float,
        batch_size: int,
        hidden_dtype: mx.Dtype,
    ) -> mx.array:
        projected, _ = adaln((timestep * multiplier).reshape(-1), hidden_dtype=hidden_dtype)
        return projected.reshape(batch_size, -1, projected.shape[-1])

    def _cross_positional_embeddings(
        self,
        modality: TransformerModalityInput,
        config: TransformerInputConfig,
    ) -> tuple[mx.array, mx.array]:
        temporal_positions = modality.positions[:, 0:1]
        return precompute_freqs_cis(
            temporal_positions,
            self.cross_attention_dimension,
            theta=config.positional_embedding_theta,
            max_positions=(self.cross_position_maximum,),
            use_middle_indices_grid=True,
            attention_heads=config.attention_heads,
        )

    def _attach_cross_inputs(
        self,
        prepared: PreparedTransformerInput,
        modality: TransformerModalityInput,
        cross_modality: TransformerModalityInput,
        *,
        config: TransformerInputConfig,
        scale_shift_adaln: AdaLayerNormSingle,
        gate_adaln: AdaLayerNormSingle,
    ) -> PreparedTransformerInput:
        batch_size = modality.latent.shape[0]
        if (
            cross_modality.sigma.ndim != 1
            or cross_modality.sigma.shape[0] != batch_size
            or cross_modality.latent.shape[0] != batch_size
        ):
            raise LaraError("LARA-TENSOR-021", details={"reason": "invalid_cross_modality_sigma"})
        scale_shift = self._project_cross_timestep(
            scale_shift_adaln,
            modality.timesteps,
            multiplier=config.timestep_scale_multiplier,
            batch_size=batch_size,
            hidden_dtype=modality.latent.dtype,
        )
        gate = self._project_cross_timestep(
            gate_adaln,
            cross_modality.sigma,
            multiplier=self.av_cross_timestep_scale_multiplier,
            batch_size=batch_size,
            hidden_dtype=modality.latent.dtype,
        )
        stream = replace(
            prepared.stream,
            cross_positional_embeddings=self._cross_positional_embeddings(modality, config),
            cross_scale_shift_timestep=scale_shift,
            cross_gate_timestep=gate,
        )
        return replace(prepared, stream=stream)

    def prepare(
        self,
        video: TransformerModalityInput | None,
        audio: TransformerModalityInput | None,
    ) -> tuple[PreparedTransformerInput | None, PreparedTransformerInput | None]:
        if video is None and audio is None:
            raise LaraError("LARA-TENSOR-021", details={"reason": "missing_modalities"})
        prepared_video = self.video.prepare_modality(video) if video is not None else None
        prepared_audio = self.audio.prepare_modality(audio) if audio is not None else None
        if video is not None and audio is not None and prepared_video is not None and prepared_audio is not None:
            prepared_video = self._attach_cross_inputs(
                prepared_video,
                video,
                audio,
                config=self.video.config,
                scale_shift_adaln=self.video_cross_scale_shift,
                gate_adaln=self.video_cross_gate,
            )
            prepared_audio = self._attach_cross_inputs(
                prepared_audio,
                audio,
                video,
                config=self.audio.config,
                scale_shift_adaln=self.audio_cross_scale_shift,
                gate_adaln=self.audio_cross_gate,
            )
        return prepared_video, prepared_audio
