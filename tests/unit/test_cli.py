from __future__ import annotations

from pathlib import Path

from lara_ltx import cli
from lara_ltx.errors import LaraError
from lara_ltx.pipeline import LTXPipeline


class _Video:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def save(self, output: Path) -> Path:
        self.events.append(("save", output))
        return output


class _Pipeline:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def __call__(self, **kwargs: object) -> _Video:
        self.events.append(("generate", kwargs))
        return _Video(self.events)


def test_cli_generate_is_a_thin_public_pipeline_wrapper(monkeypatch, tmp_path: Path, capsys) -> None:
    events: list[object] = []
    monkeypatch.setattr(
        LTXPipeline,
        "from_pretrained",
        lambda *args, **kwargs: events.append(("load", args, kwargs)) or _Pipeline(events),
    )
    output = tmp_path / "result.mp4"

    status = cli.main(
        [
            "generate",
            "--model",
            "local-model",
            "--prompt",
            "ocean sunrise",
            "--seed",
            "42",
            "--output",
            str(output),
        ]
    )

    assert status == cli.EXIT_SUCCESS
    assert [event[0] for event in events] == ["load", "generate", "save"]
    assert events[1][1]["prompt"] == "ocean sunrise"
    assert events[1][1]["seed"] == 42
    assert str(output) in capsys.readouterr().out


def test_cli_reports_localized_lara_error_without_traceback(monkeypatch, capsys) -> None:
    def fail(*args, **kwargs):
        raise LaraError("LARA-PIPELINE-002", details={"path": "missing"})

    monkeypatch.setattr(LTXPipeline, "from_pretrained", fail)

    status = cli.main(["generate", "--model", "missing", "--prompt", "test", "--output", "out.mp4"])

    captured = capsys.readouterr()
    assert status == cli.EXIT_USER_ERROR
    assert "LARA-PIPELINE-002" in captured.err
    assert "Traceback" not in captured.err
