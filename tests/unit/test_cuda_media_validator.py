import pytest

from tools.parity.validate_cuda_media import _validated_probe_payload


def test_validated_probe_payload_requires_positive_audio_video_media() -> None:
    payload = {
        "format": {"duration": "1.25"},
        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
    }

    duration, stream_types = _validated_probe_payload(payload)

    assert duration == 1.25
    assert stream_types == {"audio", "video"}


def test_validated_probe_payload_rejects_missing_audio() -> None:
    payload = {"format": {"duration": "1.25"}, "streams": [{"codec_type": "video"}]}

    with pytest.raises(ValueError, match="LARA-RUNTIME-006"):
        _validated_probe_payload(payload)


def test_validated_probe_payload_rejects_zero_duration() -> None:
    payload = {
        "format": {"duration": "0"},
        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
    }

    with pytest.raises(ValueError, match="LARA-RUNTIME-006"):
        _validated_probe_payload(payload)
