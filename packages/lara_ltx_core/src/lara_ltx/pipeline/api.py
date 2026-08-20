"""Small public prompt-to-MP4 API backed exclusively by MLX/Metal."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import mlx.core as mx

from lara_ltx.audio_vae import load_audio_vae_decoder, load_vocoder_with_bwe
from lara_ltx.errors import LaraError
from lara_ltx.media import DecodedMediaResult, MediaEncodingConfig
from lara_ltx.models import (
    build_audio_vae_mapping,
    build_diffusion_vae_decoder_mapping,
    build_gemma_feature_mapping,
    build_gemma_text_mapping,
    build_packed_gemma_tokenizer,
    build_prompt_connector_mapping,
    build_spatial_upscaler_mapping,
    build_transformer_input_mapping,
    build_transformer_output_mapping,
    build_vocoder_mapping,
    gemma_feature_target_shapes,
    gemma_text_target_shapes,
    iter_component_weight_batches,
    load_packed_gemma_assets,
    load_packed_gemma_config,
    load_prompt_connector_config,
    tokenize_prompts,
    validate_gemma_feature_mapping,
    validate_gemma_text_mapping,
)
from lara_ltx.sampling import (
    AudioLatentLayout,
    GaussianNoiser,
    LTX2Scheduler,
    MultiModalGuider,
    Res2sSampler,
    VideoLatentLayout,
    create_audio_state,
    create_video_state,
)
from lara_ltx.text_encoder import LTXGemma4TextEncoder, build_gemma4_text_args, load_prompt_connector_processor
from lara_ltx.transformer import ResidentCheckpointTransformerBlocks
from lara_ltx.video_vae import (
    load_diffusion_video_decoder,
    load_spatial_video_upscaler,
)

from .configuration import (
    DISTRIBUTION_MANIFEST_FILENAME,
    CheckpointPaths,
    DistributionSource,
    GenerationProfile,
    PipelineProfile,
    load_distribution_source,
    load_pipeline_profile,
)
from .decode import CheckpointDecodeRuntime, DecodeRuntimeConfig
from .generation import LocalGenerationRuntime
from .two_stage import (
    CheckpointStageModuleLoader,
    TwoStageContexts,
    TwoStageSamplingConfig,
    TwoStageSamplingRequest,
    TwoStageSamplingRuntime,
)


@dataclass(frozen=True)
class LTXVideo:
    """Decoded synchronized video/audio with its configured local encoder."""

    media: DecodedMediaResult
    encoding: MediaEncodingConfig

    @property
    def video(self):  # type intentionally follows the NumPy payload
        return self.media.video

    @property
    def audio(self):  # type intentionally follows the NumPy payload
        return self.media.audio

    def save(self, output_path: Path | str) -> Path:
        return self.media.save(output_path, config=self.encoding)


def _load_text_contexts(
    text_checkpoint: Path,
    connector_checkpoint: Path,
    *,
    prompt: str,
    negative_prompt: str,
    max_length: int,
) -> TwoStageContexts:
    packed_config = load_packed_gemma_config(text_checkpoint)
    text_rules = validate_gemma_text_mapping(build_gemma_text_mapping(text_checkpoint), packed_config)
    feature_rules = validate_gemma_feature_mapping(build_gemma_feature_mapping(text_checkpoint))
    encoder = LTXGemma4TextEncoder(build_gemma4_text_args(packed_config))
    for rules, shapes in (
        (text_rules, gemma_text_target_shapes(packed_config)),
        (feature_rules, gemma_feature_target_shapes()),
    ):
        for batch in iter_component_weight_batches((text_checkpoint,), rules, expected_target_shapes=shapes):
            encoder.load_weights(batch, strict=False)
            mx.eval(*[value for _, value in batch])
    assets = load_packed_gemma_assets(text_checkpoint)
    tokenizer = build_packed_gemma_tokenizer(assets, max_length=max_length)
    token_ids, attention_mask = tokenize_prompts(
        tokenizer,
        [prompt, negative_prompt],
        max_length=max_length,
    )
    attention = mx.array(attention_mask, dtype=mx.int32)
    video_features, audio_features = encoder.encode_features(
        mx.array(token_ids, dtype=mx.int32),
        attention,
    )
    mx.eval(video_features, audio_features)
    del encoder, tokenizer, assets
    mx.clear_cache()

    connector_config = load_prompt_connector_config(connector_checkpoint)
    connector = load_prompt_connector_processor(
        checkpoint=connector_checkpoint,
        mapping=build_prompt_connector_mapping(connector_checkpoint),
        config=connector_config,
    )
    video, audio, _ = connector(video_features, audio_features, attention)
    contexts = TwoStageContexts(
        video_positive=video[0:1],
        video_negative=video[1:2],
        audio_positive=audio[0:1],
        audio_negative=audio[1:2],
    )
    mx.eval(
        contexts.video_positive,
        contexts.video_negative,
        contexts.audio_positive,
        contexts.audio_negative,
    )
    del connector, video_features, audio_features, attention, video, audio
    mx.clear_cache()
    return contexts


def _build_request(profile: GenerationProfile, contexts: TwoStageContexts) -> TwoStageSamplingRequest:
    latent_frames = (profile.num_frames - 1) // 8 + 1
    stage_one_video_layout = VideoLatentLayout(
        batch=1,
        channels=profile.video_latent_channels,
        frames=latent_frames,
        height=profile.height // 64,
        width=profile.width // 64,
        frame_rate=profile.frame_rate,
    )
    stage_two_video_layout = VideoLatentLayout(
        batch=1,
        channels=profile.video_latent_channels,
        frames=latent_frames,
        height=profile.height // 32,
        width=profile.width // 32,
        frame_rate=profile.frame_rate,
    )
    audio_frames = round(
        profile.num_frames
        / profile.frame_rate
        * profile.audio_model_sample_rate
        / profile.audio_hop_length
        / profile.audio_latent_downsample_factor
    )
    audio_layout = AudioLatentLayout(
        batch=1,
        channels=profile.audio_latent_channels,
        frames=audio_frames,
        mel_bins=profile.audio_mel_bins,
        sample_rate=profile.audio_model_sample_rate,
        hop_length=profile.audio_hop_length,
        latent_downsample_factor=profile.audio_latent_downsample_factor,
    )
    scheduler = LTX2Scheduler()
    scheduler_profile = profile.scheduler
    stage_one_sigmas = scheduler.execute(
        steps=profile.num_inference_steps,
        token_count=stage_one_video_layout.token_count,
        max_shift=scheduler_profile.max_shift,
        base_shift=scheduler_profile.base_shift,
        stretch=scheduler_profile.stretch,
        terminal=scheduler_profile.terminal,
    )
    return TwoStageSamplingRequest(
        stage_one_video=create_video_state(stage_one_video_layout),
        stage_one_audio=create_audio_state(audio_layout),
        stage_one_video_layout=stage_one_video_layout,
        stage_two_video_layout=stage_two_video_layout,
        audio_layout=audio_layout,
        contexts=contexts,
        video_guider=MultiModalGuider(profile.video_guidance),
        audio_guider=MultiModalGuider(profile.audio_guidance),
        stage_one_sigmas=stage_one_sigmas,
        stage_two_sigmas=mx.array(profile.stage_two_sigmas, dtype=mx.float32),
    )


class LTXPipeline:
    """Native Apple Silicon LTX-2.5 BF16 pipeline with no serving process."""

    def __init__(self, *, checkpoints: CheckpointPaths, profile: PipelineProfile, model_root: Path) -> None:
        self.checkpoints = checkpoints
        self.profile = profile
        self.model_root = model_root

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        *,
        revision: str | None = None,
        cache_dir: Path | str | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        profile_path: Path | None = None,
    ) -> "LTXPipeline":
        """Resolve a local directory or an authenticated Hugging Face snapshot."""

        profile = load_pipeline_profile(profile_path)
        candidate = Path(model).expanduser()
        cache_value = cache_dir or os.getenv(profile.download.cache_environment_variable)
        token_value = token or os.getenv(profile.download.token_environment_variable)
        source: DistributionSource | None = None
        if candidate.is_dir():
            manifest_path = candidate / DISTRIBUTION_MANIFEST_FILENAME
            if manifest_path.is_file():
                source = load_distribution_source(manifest_path)
            else:
                root = candidate.resolve()
        else:
            requested_repository = str(model)
            if requested_repository != profile.model.repository_id:
                try:
                    from huggingface_hub import hf_hub_download
                    from huggingface_hub.errors import EntryNotFoundError

                    manifest_path = Path(
                        hf_hub_download(
                            repo_id=requested_repository,
                            filename=DISTRIBUTION_MANIFEST_FILENAME,
                            revision=revision,
                            cache_dir=cache_value,
                            token=token_value,
                            local_files_only=local_files_only,
                        )
                    )
                    source = load_distribution_source(manifest_path)
                except EntryNotFoundError:
                    source = None
                except LaraError:
                    raise
                except Exception as error:
                    raise LaraError("LARA-PIPELINE-002", details={"path": requested_repository}) from error
            if source is None:
                source = DistributionSource(
                    repository_id=requested_repository,
                    revision=revision or profile.model.revision,
                )
        if source is not None:
            try:
                from huggingface_hub import snapshot_download

                root = Path(
                    snapshot_download(
                        repo_id=source.repository_id,
                        revision=source.revision,
                        cache_dir=cache_value,
                        token=token_value,
                        allow_patterns=profile.model.allow_patterns,
                        local_files_only=local_files_only,
                    )
                ).resolve()
            except Exception as error:
                raise LaraError("LARA-PIPELINE-002", details={"path": source.repository_id}) from error
        return cls(checkpoints=profile.model.resolve(root), profile=profile, model_root=root)

    def _sampling_runtime(self, generation: GenerationProfile) -> TwoStageSamplingRuntime:
        paths = self.checkpoints
        blocks = ResidentCheckpointTransformerBlocks(
            checkpoint=paths.transformer,
            lora_checkpoint=paths.distilled_lora,
            lora_strength=generation.stage_one_lora_strength,
            block_count=generation.block_count,
        )
        upscaler = load_spatial_video_upscaler(
            upscaler_checkpoint=paths.spatial_upscaler,
            video_vae_checkpoint=paths.video_vae,
            mapping=build_spatial_upscaler_mapping(paths.spatial_upscaler),
        )
        module_loader = CheckpointStageModuleLoader(
            transformer_checkpoint=paths.transformer,
            lora_checkpoint=paths.distilled_lora,
            input_mapping=build_transformer_input_mapping(paths.transformer),
            output_mapping=build_transformer_output_mapping(paths.transformer),
            lora_pairs=blocks.lora_pairs,
        )
        sampler = generation.sampler
        sampler_keywords = {
            "eta": sampler.eta,
            "bongmath": sampler.bongmath,
            "bongmath_max_iterations": sampler.bongmath_max_iterations,
            "noise_seed": generation.seed,
            "noise_seed_substep": generation.seed + sampler.substep_seed_offset,
        }
        return TwoStageSamplingRuntime(
            blocks=blocks,
            expected_block_count=generation.block_count,
            stage_module_loader=module_loader,
            set_lora_strength=blocks.set_lora_strength,
            upscaler=upscaler,
            initial_noiser=GaussianNoiser(generation.seed),
            stage_one_sampler=Res2sSampler(**sampler_keywords),
            stage_two_sampler=Res2sSampler(**sampler_keywords),
            config=TwoStageSamplingConfig(
                stage_one_lora_strength=generation.stage_one_lora_strength,
                stage_two_lora_strength=generation.stage_two_lora_strength,
            ),
        )

    def _decode_runtime(self, generation: GenerationProfile) -> CheckpointDecodeRuntime:
        paths = self.checkpoints
        decode = self.profile.decode
        return CheckpointDecodeRuntime(
            video_decoder_loader=partial(
                load_diffusion_video_decoder,
                checkpoint=paths.video_vae,
                mapping=build_diffusion_vae_decoder_mapping(paths.video_vae),
            ),
            audio_decoder_loader=partial(
                load_audio_vae_decoder,
                checkpoint=paths.audio_vae,
                mapping=build_audio_vae_mapping(paths.audio_vae),
            ),
            vocoder_loader=partial(
                load_vocoder_with_bwe,
                checkpoint=paths.audio_vae,
                mapping=build_vocoder_mapping(paths.audio_vae),
            ),
            config=DecodeRuntimeConfig(
                video_noise_seed=generation.seed + decode.video_noise_seed_offset,
                video_timesteps=decode.video_timesteps,
                video_activation_budget_bytes=decode.video_activation_budget_bytes,
            ),
        )

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
    ) -> LTXVideo:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LaraError("LARA-PIPELINE-003", details={"reason": "empty_prompt"})
        generation = self.profile.generation.with_overrides(
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
        )
        contexts = _load_text_contexts(
            self.checkpoints.text_encoder,
            self.checkpoints.transformer,
            prompt=prompt.strip(),
            negative_prompt=generation.negative_prompt,
            max_length=generation.text_max_length,
        )
        request = _build_request(generation, contexts)
        runtime = LocalGenerationRuntime(
            sampling_runtime_factory=partial(self._sampling_runtime, generation),
            decode_runtime_factory=partial(self._decode_runtime, generation),
        )
        media = runtime.generate(request)
        encoding = replace(self.profile.media, frame_rate=generation.frame_rate)
        return LTXVideo(media=media, encoding=encoding)
