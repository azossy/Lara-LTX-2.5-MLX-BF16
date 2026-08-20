from argparse import Namespace
from pathlib import Path

from tools.parity.run_cuda_hq_golden import build_command


def test_build_command_uses_split_bf16_components_without_quantization() -> None:
    root = Path("/reference")
    arguments = Namespace(
        python_executable="python",
        transformer_path=root / "transformer.safetensors",
        text_encoder_path=root / "text_encoder.safetensors",
        video_vae_path=root / "video_vae.safetensors",
        audio_vae_path=root / "audio_vae.safetensors",
        duration_head_path=root / "duration_head.safetensors",
        distilled_lora_path=root / "distilled_lora.safetensors",
        spatial_upsampler_path=root / "upscaler.safetensors",
        prompt="test prompt",
        output_video=root / "output.mp4",
        seed=7,
        width=512,
        height=320,
        num_frames=17,
        frame_rate=24.0,
        num_inference_steps=15,
        diffvae_optimization="chunked_eager",
    )

    command = build_command(arguments)

    assert command[:3] == ["python", "-m", "ltx_pipelines.ti2vid_two_stages_hq"]
    assert "--quantization" not in command
    assert command[command.index("--transformer-path") + 1] == str(arguments.transformer_path)
    assert command[command.index("--video-vae-path") + 1] == str(arguments.video_vae_path)
    assert command[command.index("--num-frames") + 1] == "17"
