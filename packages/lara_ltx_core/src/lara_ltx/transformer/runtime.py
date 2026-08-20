"""Memory-bounded sequential runtime for the 48-block AV transformer."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from lara_ltx.errors import LaraError
from lara_ltx.models.loading import iter_component_weight_batches, load_safetensors_shard
from lara_ltx.models.lora import LoraPair, build_lora_pairs
from lara_ltx.models.lora_fusion import fuse_lora_weight
from lara_ltx.models.transformer_block import (
    AUDIO_DIMENSION,
    AUDIO_HEAD_COUNT,
    PRODUCTION_TRANSFORMER_BLOCK_COUNT,
    TRANSFORMER_BLOCK_SOURCE_PREFIX,
    VIDEO_DIMENSION,
    VIDEO_HEAD_COUNT,
    build_transformer_block_mappings,
    transformer_block_target_shapes,
)

from .blocks import AVTransformerBlock, TransformerStream, VideoTransformerConfig

VIDEO_HEAD_DIMENSION = VIDEO_DIMENSION // VIDEO_HEAD_COUNT
AUDIO_HEAD_DIMENSION = AUDIO_DIMENSION // AUDIO_HEAD_COUNT


@dataclass(frozen=True)
class TransformerBlockLoadRecord:
    block_index: int
    loaded_tensor_count: int
    fused_lora_pair_count: int
    active_after_load_bytes: int
    peak_after_execution_bytes: int


@dataclass(frozen=True)
class TransformerSequenceResult:
    video: TransformerStream | None
    audio: TransformerStream | None
    completed_block_count: int


def production_transformer_block() -> AVTransformerBlock:
    """Construct one empty production block with checkpoint-compatible names."""

    return AVTransformerBlock(
        video=VideoTransformerConfig(
            dim=VIDEO_DIMENSION,
            heads=VIDEO_HEAD_COUNT,
            head_dim=VIDEO_HEAD_DIMENSION,
            context_dim=VIDEO_DIMENSION,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            ff_bias=False,
        ),
        audio=VideoTransformerConfig(
            dim=AUDIO_DIMENSION,
            heads=AUDIO_HEAD_COUNT,
            head_dim=AUDIO_HEAD_DIMENSION,
            context_dim=AUDIO_DIMENSION,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            ff_bias=True,
        ),
    )


def run_transformer_block_sequence(
    video: TransformerStream | None,
    audio: TransformerStream | None,
    blocks: Iterable[tuple[int, AVTransformerBlock]],
) -> TransformerSequenceResult:
    """Execute and materialize one block at a time to cut the MLX lazy graph."""

    completed = 0
    for block_index, block in blocks:
        video_output, audio_output = block(video, audio)
        materialized = tuple(value for value in (video_output, audio_output) if value is not None)
        if not materialized:
            raise LaraError("LARA-TENSOR-028", details={"block_index": block_index})
        mx.eval(*materialized)
        if video is not None and video_output is not None:
            video = replace(video, x=video_output)
        if audio is not None and audio_output is not None:
            audio = replace(audio, x=audio_output)
        completed += 1
        # Drop the consumer's reference before requesting the next generator
        # item; otherwise the next block is loaded while this one is resident.
        del block
    return TransformerSequenceResult(video=video, audio=audio, completed_block_count=completed)


class CheckpointTransformerBlocks:
    """Yield one strict-loaded production block and release it before the next."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        block_count: int = PRODUCTION_TRANSFORMER_BLOCK_COUNT,
        lora_checkpoint: Path | None = None,
        lora_strength: float = 0.0,
    ) -> None:
        if block_count <= 0:
            raise LaraError("LARA-MODEL-028", details={"key": f"block_count={block_count}"})
        if not math.isfinite(lora_strength) or lora_strength < 0:
            raise LaraError("LARA-MODEL-027", details={"value": lora_strength})
        self.checkpoint = checkpoint
        self.block_count = block_count
        self.lora_checkpoint = lora_checkpoint
        self.lora_strength = lora_strength
        self.mappings = build_transformer_block_mappings(checkpoint, block_count=block_count)
        self.lora_pairs = (
            build_lora_pairs(lora_checkpoint, checkpoint) if lora_checkpoint is not None and lora_strength > 0 else ()
        )
        self.records: list[TransformerBlockLoadRecord] = []

    def _pairs_for_block(self, block_index: int) -> tuple[LoraPair, ...]:
        prefix = f"{TRANSFORMER_BLOCK_SOURCE_PREFIX}{block_index}."
        return tuple(pair for pair in self.lora_pairs if pair.target_key.startswith(prefix))

    def _fuse_block_lora(self, block: AVTransformerBlock, block_index: int) -> int:
        if self.lora_checkpoint is None:
            return 0
        prefix = f"{TRANSFORMER_BLOCK_SOURCE_PREFIX}{block_index}."
        fused_count = 0
        for pair in self._pairs_for_block(block_index):
            runtime_key = pair.target_key.removeprefix(prefix)
            parameters = dict(tree_flatten(block.parameters()))
            base = parameters.get(runtime_key)
            if base is None or tuple(base.shape) != pair.base_shape:
                raise LaraError("LARA-MODEL-026", details={"key": pair.target_key})
            loaded = load_safetensors_shard(
                self.lora_checkpoint,
                required_names=(pair.a_key, pair.b_key),
            )
            fused = fuse_lora_weight(
                base,
                loaded.tensors[pair.a_key],
                loaded.tensors[pair.b_key],
                self.lora_strength,
            )
            mx.eval(fused)
            block.load_weights(((runtime_key, fused),), strict=False)
            fused_count += 1
        return fused_count

    def __iter__(self) -> Iterator[tuple[int, AVTransformerBlock]]:
        target_shapes = transformer_block_target_shapes()
        for block_index, mapping in enumerate(self.mappings):
            rules = mapping.get("rules")
            if not isinstance(rules, list):
                raise LaraError("LARA-MODEL-014", details={"source_key": f"block_{block_index}_rules"})
            mx.reset_peak_memory()
            block = production_transformer_block()
            loaded_tensor_count = 0
            for weight_batch in iter_component_weight_batches(
                (self.checkpoint,),
                rules,
                expected_target_shapes=target_shapes,
            ):
                block.load_weights(weight_batch)
                mx.eval(*[value for _, value in weight_batch])
                loaded_tensor_count += len(weight_batch)
            weight_batch = ()
            fused_count = self._fuse_block_lora(block, block_index)
            active_after_load = int(mx.get_active_memory())
            yield block_index, block
            self.records.append(
                TransformerBlockLoadRecord(
                    block_index=block_index,
                    loaded_tensor_count=loaded_tensor_count,
                    fused_lora_pair_count=fused_count,
                    active_after_load_bytes=active_after_load,
                    peak_after_execution_bytes=int(mx.get_peak_memory()),
                )
            )
            del block
