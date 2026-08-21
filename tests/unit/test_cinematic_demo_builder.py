from __future__ import annotations

from pathlib import Path

import pytest

from tools.quality.build_cinematic_demo import (
    DemoBuildError,
    _individual_command,
    _scale_filter,
    _side_by_side_command,
    _validate_source,
)

PRESENTATION = {
    "target_width": 3840,
    "target_height": 2160,
    "scaler": "lanczos",
    "video_codec": "libx264",
    "crf": 16,
    "pixel_format": "yuv420p",
    "audio_codec": "aac",
    "audio_bitrate": "320k",
}
CASE = {"width": 512, "height": 320, "num_frames": 241}


def test_individual_export_uses_aspect_preserving_lanczos_and_faststart() -> None:
    command = _individual_command(
        executable="ffmpeg",
        source=Path("native.mp4"),
        output=Path("presentation.mp4"),
        presentation=PRESENTATION,
    )

    assert _scale_filter(3840, 2160, "lanczos") in command
    assert "+faststart" in command
    assert command[-1] == "presentation.mp4"


def test_side_by_side_export_preserves_two_audio_tracks() -> None:
    command = _side_by_side_command(
        executable="ffmpeg",
        cuda_video=Path("cuda.mp4"),
        mlx_video=Path("mlx.mp4"),
        output=Path("comparison.mp4"),
        presentation=PRESENTATION,
    )

    filter_graph = command[command.index("-filter_complex") + 1]
    assert "hstack=inputs=2:shortest=1" in filter_graph
    assert "[0:a]apad[cuda_audio]" in filter_graph
    assert "[1:a]apad[mlx_audio]" in filter_graph
    assert command.count("-map") == 3
    assert "title=CUDA audio" in command
    assert "title=MLX Metal audio" in command


def test_source_validation_requires_exact_native_grid_and_audio() -> None:
    probe = {
        "streams": [
            {"codec_type": "video", "width": 512, "height": 320, "nb_read_frames": "241"},
            {"codec_type": "audio"},
        ]
    }
    _validate_source(Path("valid.mp4"), probe, CASE)

    probe["streams"][0]["nb_read_frames"] = "240"
    with pytest.raises(DemoBuildError, match="source_grid_mismatch"):
        _validate_source(Path("invalid.mp4"), probe, CASE)


def test_side_by_side_rejects_odd_target_width() -> None:
    presentation = dict(PRESENTATION, target_width=3839)

    with pytest.raises(DemoBuildError, match="side_by_side_width_must_be_even"):
        _side_by_side_command(
            executable="ffmpeg",
            cuda_video=Path("cuda.mp4"),
            mlx_video=Path("mlx.mp4"),
            output=Path("comparison.mp4"),
            presentation=presentation,
        )
