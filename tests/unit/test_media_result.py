from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pytest
from lara_ltx.errors import LaraError
from lara_ltx.media import DecodedMediaResult, MediaEncodingConfig


def _result() -> DecodedMediaResult:
    return DecodedMediaResult(
        video=np.linspace(-1.0, 1.0, 1 * 3 * 2 * 4 * 6, dtype=np.float32).reshape(1, 3, 2, 4, 6),
        audio=np.zeros((1, 2, 800), dtype=np.float32),
    )


def test_decoded_media_result_encodes_atomically_with_configured_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"mp4")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("lara_ltx.media.result._resolve_binary", lambda value: value)
    monkeypatch.setattr("lara_ltx.media.result.subprocess.run", run)
    output = tmp_path / "result.mp4"

    saved = _result().save(
        output,
        config=MediaEncodingConfig(frame_rate=24.0, audio_sample_rate=48_000, ffmpeg_binary="configured-ffmpeg"),
    )

    assert saved == output.resolve()
    assert output.read_bytes() == b"mp4"
    assert commands[0][0] == "configured-ffmpeg"
    assert commands[0][commands[0].index("-framerate") + 1] == "24.0"
    assert commands[0][commands[0].index("-ar") + 1] == "48000"


@pytest.mark.parametrize(
    ("video", "audio", "reason"),
    [
        (np.zeros((1, 2, 2, 2, 2)), np.zeros((1, 2, 10)), "invalid_video_shape"),
        (np.zeros((1, 3, 2, 2, 2)), np.zeros((1, 1, 10)), "invalid_audio_shape"),
        (np.full((1, 3, 2, 2, 2), np.nan), np.zeros((1, 2, 10)), "non_finite_video"),
    ],
)
def test_decoded_media_result_rejects_invalid_tensors(
    tmp_path: Path,
    video: np.ndarray,
    audio: np.ndarray,
    reason: str,
) -> None:
    with pytest.raises(LaraError) as error:
        DecodedMediaResult(video=video, audio=audio).save(
            tmp_path / "result.mp4",
            config=MediaEncodingConfig(frame_rate=24.0),
        )

    assert error.value.code == "LARA-MEDIA-001"
    assert error.value.details["reason"] == reason


def test_media_encoding_config_rejects_invalid_values() -> None:
    with pytest.raises(LaraError) as error:
        MediaEncodingConfig(frame_rate=0.0)

    assert error.value.code == "LARA-MEDIA-001"
