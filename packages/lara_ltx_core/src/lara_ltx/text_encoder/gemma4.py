"""Logits-free Gemma 4 hidden-state collection on the native MLX-LM core."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lara_ltx.errors import LaraError
from lara_ltx.models.gemma_feature_runtime import GemmaFeatureExtractorV2
from lara_ltx.models.gemma_text import gemma_text_target_shapes

try:
    from mlx_lm.models.base import create_causal_mask
    from mlx_lm.models.gemma4_text import Gemma4TextModel, ModelArgs
except ImportError as exc:  # pragma: no cover - the dependency is platform-gated
    raise LaraError("LARA-RUNTIME-007") from exc


def build_gemma4_text_args(packed_config: dict[str, Any]) -> ModelArgs:
    """Build only the reviewed text core; vision/audio towers and LM head stay absent."""

    gemma_text_target_shapes(packed_config)
    text_config = packed_config["text_config"]
    return ModelArgs.from_dict(text_config)


def _left_padding_counts(input_ids: mx.array, attention_mask: mx.array) -> mx.array:
    if input_ids.ndim != 2 or attention_mask.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise LaraError("LARA-TENSOR-022", details={"reason": "invalid_token_layout"})
    mask = np.asarray(attention_mask)
    if np.any((mask != 0) & (mask != 1)) or np.any(np.diff(mask, axis=1) < 0):
        raise LaraError("LARA-TENSOR-022", details={"reason": "attention_mask_must_be_left_padded"})
    return mx.array(np.sum(mask == 0, axis=1), dtype=mx.int32)


class Gemma4HiddenStateEncoder(nn.Module):
    """MLX-LM Gemma 4 text core that returns HF-compatible 49 hidden states."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        if args.hidden_size_per_layer_input or args.num_kv_shared_layers or args.enable_moe_block:
            raise LaraError("LARA-MODEL-035", details={"reason": "unsupported_text_architecture"})
        self.model = Gemma4TextModel(args)

    def _masks(self, hidden: mx.array, left_padding: mx.array) -> list[mx.array]:
        by_type: dict[str, mx.array] = {}
        masks: list[mx.array] = []
        for layer in self.model.layers:
            if layer.layer_type not in by_type:
                window = self.model.window_size if layer.layer_type == "sliding_attention" else None
                by_type[layer.layer_type] = create_causal_mask(
                    hidden.shape[1],
                    window_size=window,
                    left_padding=left_padding,
                )
            masks.append(by_type[layer.layer_type])
        return masks

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        left_padding = _left_padding_counts(input_ids, attention_mask)
        hidden = self.model.embed_tokens(input_ids) * self.model.embed_scale
        mx.eval(hidden)
        hidden_states = [hidden]
        masks = self._masks(hidden, left_padding)
        final_layer_index = len(self.model.layers) - 1
        for index, (layer, mask) in enumerate(zip(self.model.layers, masks, strict=True)):
            hidden, _, _ = layer(hidden, mask, None, per_layer_input=None, shared_kv=None, offset=None)
            if index != final_layer_index:
                mx.eval(hidden)
                hidden_states.append(hidden)
        hidden = self.model.norm(hidden)
        mx.eval(hidden)
        hidden_states.append(hidden)
        return mx.stack(hidden_states, axis=-1)


class LTXGemma4TextEncoder(Gemma4HiddenStateEncoder):
    """Complete LTX text conditioner without a language-model logits head."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        self.feature_extractor = GemmaFeatureExtractorV2()

    def encode_features(self, input_ids: mx.array, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        hidden_states = self(input_ids, attention_mask)
        return self.feature_extractor(hidden_states, attention_mask)
