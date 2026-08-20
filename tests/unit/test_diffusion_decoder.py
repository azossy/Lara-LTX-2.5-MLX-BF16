from pathlib import Path

import numpy as np
import pytest
from mlx.utils import tree_flatten

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.diffusion_decoder import (
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_STAGE5_KERNEL,
    DEFAULT_STAGE_CHANNELS,
    DEFAULT_STAGE_KERNELS,
    DEFAULT_TIMESTEP_SCALE_MULTIPLIER,
    MODEL_OUTPUT_X0,
    DiffusionVideoDecoder,
    DiffusionVideoDecoderConfig,
    load_diffusion_video_decoder,
)


def _tiny_config(**overrides: object) -> DiffusionVideoDecoderConfig:
    values: dict[str, object] = {
        "in_channels": 4,
        "out_channels": 3,
        "patch_size": 2,
        "head_dim": 16,
        "stage_channels": (16, 16, 16, 16, 16),
        "stage_depths": (1, 1, 1, 1, 1),
        "stage_kernels": ((3, 3, 3),) * 5,
        "upsamples": (((1, 1, 1), 1),) * 4,
        "stage5_kernel": (3, 3, 3),
        "stage5_channels": 16,
        "timestep_embedding_dim": 16,
    }
    values.update(overrides)
    return DiffusionVideoDecoderConfig(**values)


def test_official_bf16_decoder_defaults_match_checkpoint_metadata() -> None:
    """Prevent synthetic test settings from drifting from the released VAE."""

    config = DiffusionVideoDecoderConfig()

    assert config.stage_channels == DEFAULT_STAGE_CHANNELS == (2048, 1024, 512, 512, 256)
    assert (
        config.stage_kernels
        == DEFAULT_STAGE_KERNELS
        == (
            (3, 7, 7),
            (3, 7, 7),
            (3, 5, 5),
            (3, 5, 5),
            (11, 11, 11),
        )
    )
    assert config.stage5_kernel == DEFAULT_STAGE5_KERNEL == (11, 11, 11)
    assert config.default_num_inference_steps == DEFAULT_INFERENCE_STEPS == 1
    assert config.timestep_scale_multiplier == DEFAULT_TIMESTEP_SCALE_MULTIPLIER == 1000.0
    assert config.model_output_type == MODEL_OUTPUT_X0


def test_default_timestep_schedule_is_batched() -> None:
    decoder = DiffusionVideoDecoder(_tiny_config(default_num_inference_steps=2))
    schedule = decoder.default_timesteps(batch_size=2)
    np.testing.assert_allclose(np.asarray(schedule), np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32))


def test_x0_euler_conversion_matches_velocity_definition() -> None:
    decoder = DiffusionVideoDecoder(_tiny_config(model_output_type="x0"))
    sample = mx.array([[[[[2.0]]]]], dtype=mx.float32)
    denoised = mx.array([[[[[1.0]]]]], dtype=mx.float32)
    result = decoder.euler_step(sample, denoised, mx.array([0.5]), mx.array([0.25]))
    np.testing.assert_allclose(np.asarray(result), np.array([[[[[1.5]]]]], dtype=np.float32))


def test_invalid_stage_reduction_is_rejected() -> None:
    with pytest.raises(LaraError, match="LARA-TENSOR-016"):
        DiffusionVideoDecoder(_tiny_config(stage_channels=(16, 32, 16, 16, 16)))


def test_production_decode_pads_and_crops_small_latent() -> None:
    decoder = DiffusionVideoDecoder(_tiny_config(default_num_inference_steps=1))
    assert decoder.minimum_latent_shape == (3, 3, 3)
    latent = mx.zeros((1, 4, 1, 1, 1), dtype=mx.bfloat16)
    noise = mx.zeros((1, 3, 1, 2, 2), dtype=mx.bfloat16)
    output = decoder.decode(latent, noise)
    assert output.shape == noise.shape
    assert bool(mx.all(mx.isfinite(output)).item())


def test_tiling_recommendation_respects_activation_budget() -> None:
    decoder = DiffusionVideoDecoder(_tiny_config())
    full = decoder.recommend_tiling((3, 16, 16), activation_budget_bytes=3_000_000)
    constrained = decoder.recommend_tiling((3, 16, 16), activation_budget_bytes=200_000)

    assert full.stage4.tile_shape == (5, 16, 16)
    assert full.stage5.tile_shape == (5, 16, 16)
    assert np.prod(constrained.stage4.tile_shape) < np.prod(full.stage4.tile_shape)
    assert np.prod(constrained.stage5.tile_shape) < np.prod(full.stage5.tile_shape)
    assert constrained.resident_stage3_bytes == 40_960


def test_tiling_recommendation_rejects_budget_below_resident_context() -> None:
    decoder = DiffusionVideoDecoder(_tiny_config())
    with pytest.raises(LaraError, match="LARA-RUNTIME-001"):
        decoder.recommend_tiling((3, 16, 16), activation_budget_bytes=10_000)


def _tiny_checkpoint_mapping(tmp_path: Path) -> tuple[Path, dict[str, object], DiffusionVideoDecoderConfig]:
    config = _tiny_config()
    template = DiffusionVideoDecoder(config)
    weights = {
        name: mx.full(value.shape, 0.25, dtype=mx.bfloat16) for name, value in tree_flatten(template.parameters())
    }
    checkpoint = tmp_path / "tiny-video-vae.safetensors"
    mx.save_safetensors(str(checkpoint), weights)
    rules = [
        {
            "source_file": checkpoint.name,
            "source_key": name,
            "target_key": name,
            "transform": "identity",
            "dtype": "BF16",
            "shape": list(value.shape),
        }
        for name, value in sorted(weights.items())
    ]
    mapping: dict[str, object] = {
        "schema_version": 2,
        "component": "diffusion_video_decoder",
        "ignored_sources": [{"source_key": "decoder.type_emb", "reason": "upstream_load_artifact"}],
        "rules": rules,
    }
    return checkpoint, mapping, config


def test_diffusion_decoder_loader_strict_loads_all_reviewed_targets(tmp_path: Path) -> None:
    checkpoint, mapping, config = _tiny_checkpoint_mapping(tmp_path)

    decoder = load_diffusion_video_decoder(checkpoint=checkpoint, mapping=mapping, config=config)

    parameters = dict(tree_flatten(decoder.parameters()))
    assert len(parameters) == len(mapping["rules"])
    for value in parameters.values():
        np.testing.assert_array_equal(
            np.asarray(value.astype(mx.float32)), np.full(value.shape, 0.25, dtype=np.float32)
        )


def test_diffusion_decoder_loader_rejects_manifest_metadata(tmp_path: Path) -> None:
    checkpoint, mapping, config = _tiny_checkpoint_mapping(tmp_path)
    mapping["component"] = "wrong_component"

    with pytest.raises(LaraError) as error:
        load_diffusion_video_decoder(checkpoint=checkpoint, mapping=mapping, config=config)

    assert error.value.code == "LARA-MODEL-037"
