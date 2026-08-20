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
from lara_ltx.models.transformer_input import (
    transformer_input_target_shapes,
    validate_transformer_input_mapping,
)
from lara_ltx.models.transformer_output import (
    transformer_output_target_shapes,
    validate_transformer_output_mapping,
)

from .blocks import AVTransformerBlock, TransformerStream, VideoTransformerConfig
from .input import AVTransformerInputPreprocessor, TransformerInputConfig
from .output import AVTransformerOutput, TransformerOutputConfig

VIDEO_HEAD_DIMENSION = VIDEO_DIMENSION // VIDEO_HEAD_COUNT
AUDIO_HEAD_DIMENSION = AUDIO_DIMENSION // AUDIO_HEAD_COUNT
PRODUCTION_INPUT_CHANNELS = 128
PRODUCTION_OUTPUT_CHANNELS = 128
VIDEO_POSITION_MAXIMUMS = (20, 2_048, 2_048)
AUDIO_POSITION_MAXIMUMS = (20,)
ADALN_COEFFICIENT = 9
PROMPT_ADALN_COEFFICIENT = 2


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


def production_transformer_input() -> AVTransformerInputPreprocessor:
    def config(hidden_dimension: int, max_positions: tuple[int, ...], *, keyframes: bool) -> TransformerInputConfig:
        return TransformerInputConfig(
            input_channels=PRODUCTION_INPUT_CHANNELS,
            hidden_dimension=hidden_dimension,
            adaln_coefficient=ADALN_COEFFICIENT,
            prompt_adaln_coefficient=PROMPT_ADALN_COEFFICIENT,
            attention_heads=VIDEO_HEAD_COUNT,
            max_positions=max_positions,
            use_keyframes_absolute_embedding=keyframes,
        )

    return AVTransformerInputPreprocessor(
        video=config(VIDEO_DIMENSION, VIDEO_POSITION_MAXIMUMS, keyframes=True),
        audio=config(AUDIO_DIMENSION, AUDIO_POSITION_MAXIMUMS, keyframes=False),
        cross_attention_dimension=AUDIO_DIMENSION,
    )


def production_transformer_output() -> AVTransformerOutput:
    return AVTransformerOutput(
        video=TransformerOutputConfig(VIDEO_DIMENSION, PRODUCTION_OUTPUT_CHANNELS),
        audio=TransformerOutputConfig(AUDIO_DIMENSION, PRODUCTION_OUTPUT_CHANNELS),
    )


def _load_and_fuse_component(
    module: object,
    *,
    checkpoint: Path,
    rules: tuple[dict[str, object], ...],
    target_shapes: dict[str, tuple[int, ...]],
    lora_checkpoint: Path | None,
    lora_pairs: tuple[LoraPair, ...],
    lora_strength: float,
) -> int:
    for weight_batch in iter_component_weight_batches(
        (checkpoint,),
        rules,
        expected_target_shapes=target_shapes,
    ):
        module.load_weights(weight_batch)
        mx.eval(*[value for _, value in weight_batch])
    weight_batch = ()
    if lora_checkpoint is None or lora_strength == 0:
        return 0
    source_to_target = {str(rule["source_key"]): str(rule["target_key"]) for rule in rules}
    relevant = tuple(pair for pair in lora_pairs if pair.target_key in source_to_target)
    parameters = dict(tree_flatten(module.parameters()))
    for pair in relevant:
        runtime_key = source_to_target[pair.target_key]
        base = parameters.get(runtime_key)
        if base is None or tuple(base.shape) != pair.base_shape:
            raise LaraError("LARA-MODEL-026", details={"key": pair.target_key})
        loaded = load_safetensors_shard(lora_checkpoint, required_names=(pair.a_key, pair.b_key))
        fused = fuse_lora_weight(
            base,
            loaded.tensors[pair.a_key],
            loaded.tensors[pair.b_key],
            lora_strength,
        )
        mx.eval(fused)
        module.load_weights(((runtime_key, fused),), strict=False)
    return len(relevant)


def load_production_transformer_input(
    *,
    checkpoint: Path,
    mapping: dict[str, object],
    lora_checkpoint: Path | None = None,
    lora_strength: float = 0.0,
    lora_pairs: tuple[LoraPair, ...] | None = None,
) -> tuple[AVTransformerInputPreprocessor, int]:
    _validate_component_lora_request(lora_checkpoint, lora_strength)
    rules = validate_transformer_input_mapping(mapping)
    pairs = lora_pairs or (
        build_lora_pairs(lora_checkpoint, checkpoint) if lora_checkpoint is not None and lora_strength > 0 else ()
    )
    module = production_transformer_input()
    count = _load_and_fuse_component(
        module,
        checkpoint=checkpoint,
        rules=rules,
        target_shapes=transformer_input_target_shapes(),
        lora_checkpoint=lora_checkpoint,
        lora_pairs=pairs,
        lora_strength=lora_strength,
    )
    return module, count


def load_production_transformer_output(
    *,
    checkpoint: Path,
    mapping: dict[str, object],
    lora_checkpoint: Path | None = None,
    lora_strength: float = 0.0,
    lora_pairs: tuple[LoraPair, ...] | None = None,
) -> tuple[AVTransformerOutput, int]:
    _validate_component_lora_request(lora_checkpoint, lora_strength)
    rules = validate_transformer_output_mapping(mapping)
    pairs = lora_pairs or (
        build_lora_pairs(lora_checkpoint, checkpoint) if lora_checkpoint is not None and lora_strength > 0 else ()
    )
    module = production_transformer_output()
    count = _load_and_fuse_component(
        module,
        checkpoint=checkpoint,
        rules=rules,
        target_shapes=transformer_output_target_shapes(),
        lora_checkpoint=lora_checkpoint,
        lora_pairs=pairs,
        lora_strength=lora_strength,
    )
    return module, count


def _validate_component_lora_request(lora_checkpoint: Path | None, strength: float) -> None:
    if not math.isfinite(strength) or strength < 0:
        raise LaraError("LARA-MODEL-027", details={"value": strength})
    if strength > 0 and lora_checkpoint is None:
        raise LaraError("LARA-MODEL-026", details={"key": "missing_lora_checkpoint"})


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
    """Low-memory validation provider that reloads blocks for each iteration."""

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
        pairs = self._pairs_for_block(block_index)
        if not pairs:
            return 0
        required_names = tuple(name for pair in pairs for name in (pair.a_key, pair.b_key))
        loaded = load_safetensors_shard(self.lora_checkpoint, required_names=required_names)
        parameters = dict(tree_flatten(block.parameters()))
        fused_count = 0
        for pair in pairs:
            runtime_key = pair.target_key.removeprefix(prefix)
            base = parameters.get(runtime_key)
            if base is None or tuple(base.shape) != pair.base_shape:
                raise LaraError("LARA-MODEL-026", details={"key": pair.target_key})
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


class ResidentCheckpointTransformerBlocks:
    """Production provider that loads all BF16 blocks once and reuses them."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        lora_checkpoint: Path,
        lora_strength: float,
        block_count: int = PRODUCTION_TRANSFORMER_BLOCK_COUNT,
    ) -> None:
        provider = CheckpointTransformerBlocks(
            checkpoint=checkpoint,
            block_count=block_count,
            lora_checkpoint=lora_checkpoint,
            lora_strength=lora_strength,
        )
        self.checkpoint = checkpoint
        self.lora_checkpoint = lora_checkpoint
        self.lora_pairs = provider.lora_pairs
        self.blocks: list[AVTransformerBlock] = []
        for _, block in provider:
            self.blocks.append(block)
        self.load_records = tuple(provider.records)
        self.lora_strength = lora_strength

    def __iter__(self) -> Iterator[tuple[int, AVTransformerBlock]]:
        return iter(enumerate(self.blocks))

    def _pairs_for_block(self, block_index: int) -> tuple[LoraPair, ...]:
        prefix = f"{TRANSFORMER_BLOCK_SOURCE_PREFIX}{block_index}."
        return tuple(pair for pair in self.lora_pairs if pair.target_key.startswith(prefix))

    def set_lora_strength(self, strength: float) -> None:
        """Replace block-local weights from base+LoRA without a second model state."""

        if not math.isfinite(strength) or strength < 0:
            raise LaraError("LARA-MODEL-027", details={"value": strength})
        if strength == self.lora_strength:
            return
        for block_index, block in enumerate(self.blocks):
            prefix = f"{TRANSFORMER_BLOCK_SOURCE_PREFIX}{block_index}."
            pairs = self._pairs_for_block(block_index)
            base_weights = load_safetensors_shard(
                self.checkpoint,
                required_names=tuple(pair.target_key for pair in pairs),
            )
            lora_weights = load_safetensors_shard(
                self.lora_checkpoint,
                required_names=tuple(name for pair in pairs for name in (pair.a_key, pair.b_key)),
            )
            for pair in pairs:
                runtime_key = pair.target_key.removeprefix(prefix)
                fused = fuse_lora_weight(
                    base_weights.tensors[pair.target_key],
                    lora_weights.tensors[pair.a_key],
                    lora_weights.tensors[pair.b_key],
                    strength,
                )
                mx.eval(fused)
                block.load_weights(((runtime_key, fused),), strict=False)
            del fused, base_weights, lora_weights
        self.lora_strength = strength
