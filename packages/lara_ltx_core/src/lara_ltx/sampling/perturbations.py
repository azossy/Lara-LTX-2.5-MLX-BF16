"""Batch-stable STG and AV isolation masks for transformer blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import mlx.core as mx
import numpy as np

from lara_ltx.errors import LaraError
from lara_ltx.transformer.blocks import TransformerStream


class PerturbationType(IntEnum):
    SKIP_VIDEO_SELF_ATTN = 0
    SKIP_AUDIO_SELF_ATTN = 1
    SKIP_A2V_CROSS_ATTN = 2
    SKIP_V2A_CROSS_ATTN = 3


@dataclass(frozen=True)
class Perturbation:
    type: PerturbationType
    blocks: tuple[int, ...] | None

    def applies(self, perturbation_type: PerturbationType, block: int) -> bool:
        return self.type == perturbation_type and (self.blocks is None or block in self.blocks)


@dataclass(frozen=True)
class PerturbationConfig:
    perturbations: tuple[Perturbation, ...] = ()

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        return any(item.applies(perturbation_type, block) for item in self.perturbations)


class BatchedPerturbationConfig:
    def __init__(self, perturbations: tuple[PerturbationConfig, ...], *, num_blocks: int, dtype: mx.Dtype) -> None:
        if not perturbations or num_blocks <= 0:
            raise LaraError("LARA-SAMPLING-002", details={"reason": "invalid_perturbation_batch"})
        keep = np.array(
            [
                [
                    [not config.is_perturbed(PerturbationType(direction), block) for config in perturbations]
                    for block in range(num_blocks)
                ]
                for direction in range(len(PerturbationType))
            ],
            dtype=np.bool_,
        )
        self._keep_cpu = keep
        self._keep = mx.array(keep, dtype=dtype)

    def mask(self, perturbation_type: PerturbationType, block: int) -> mx.array:
        return self._keep[int(perturbation_type), block].reshape(-1, 1, 1)

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return bool(np.any(~self._keep_cpu[int(perturbation_type), block]))

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return bool(np.all(~self._keep_cpu[int(perturbation_type), block]))


def _attach(
    stream: TransformerStream | None,
    config: BatchedPerturbationConfig,
    *,
    block: int,
    self_type: PerturbationType,
    cross_type: PerturbationType,
) -> TransformerStream | None:
    if stream is None:
        return None
    self_all = config.all_in_batch(self_type, block)
    self_partial = config.any_in_batch(self_type, block) and not self_all
    cross_all = config.all_in_batch(cross_type, block)
    return replace(
        stream,
        self_attn_perturbation_mask=config.mask(self_type, block) if self_partial else None,
        self_attn_all_perturbed=self_all,
        cross_attn_perturbation_mask=None if cross_all else config.mask(cross_type, block),
        cross_attn_skip_all=cross_all,
    )


def attach_block_perturbations(
    video: TransformerStream | None,
    audio: TransformerStream | None,
    config: BatchedPerturbationConfig,
    *,
    block: int,
) -> tuple[TransformerStream | None, TransformerStream | None]:
    return (
        _attach(
            video,
            config,
            block=block,
            self_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
            cross_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
        ),
        _attach(
            audio,
            config,
            block=block,
            self_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
            cross_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
        ),
    )
