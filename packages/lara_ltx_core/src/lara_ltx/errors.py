"""Stable, localized user-facing errors."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_LOCALE = "en"
LOCALE_ENVIRONMENT_VARIABLE = "LARA_LOCALE"
MESSAGE_DIRECTORY = Path(__file__).with_name("messages")


def _catalog(locale: str) -> dict[str, str]:
    candidate = MESSAGE_DIRECTORY / f"{locale}.json"
    fallback = MESSAGE_DIRECTORY / f"{DEFAULT_LOCALE}.json"
    selected = candidate if candidate.is_file() else fallback
    with selected.open(encoding="utf-8") as handle:
        return json.load(handle)


class LaraError(RuntimeError):
    """Error carrying a stable code and localized recovery guidance."""

    def __init__(self, code: str, *, details: dict[str, Any] | None = None, locale: str | None = None) -> None:
        selected_locale = locale or os.getenv(LOCALE_ENVIRONMENT_VARIABLE, DEFAULT_LOCALE)
        message = _catalog(selected_locale).get(code, _catalog(DEFAULT_LOCALE)["LARA-GENERAL-001"])
        self.code = code
        self.details = details or {}
        rendered = message.format(**self.details)
        super().__init__(f"[{code}] {rendered}")
