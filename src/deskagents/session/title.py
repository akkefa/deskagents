"""Log-backed deterministic fallback session titles.

Upstream: packages/session/session-title at ce7073d.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import Any

from deepseek_cordis import Context, Service

_ESCAPES = re.compile(
    r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|$)|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]|\x1b[@-_]"
)
_CONTROLS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200e\u200f\u202a-\u202e"
    r"\u2060-\u2064\u2066-\u206f\ufeff]"
)


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _clean(value: str) -> str:
    return " ".join(_CONTROLS.sub("", _ESCAPES.sub("", value)).split())


def truncate_title_utf8(value: str, max_bytes: int) -> str:
    _positive("max_bytes", max_bytes)
    encoded = value.encode()
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def normalize_session_title(value: str, max_bytes: int) -> str:
    return truncate_title_utf8(_clean(value), max_bytes).rstrip()


def fallback_session_title(value: str, max_words: int, max_bytes: int) -> str:
    _positive("max_words", max_words)
    return truncate_title_utf8(" ".join(_clean(value).split()[:max_words]), max_bytes).rstrip()


def fold_session_title(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for event in reversed(events):
        if event["type"] == "session/title":
            data = event["data"]
            return {
                "title": data["title"],
                "messageSeqs": data["messageSeqs"],
                "source": data["source"],
                "eventSeq": event["seq"],
                "updatedAt": event["time"],
            }
    return None


class SessionTitleInvalidError(ValueError):
    pass


class SessionTitleService(Service):
    name = "sessionTitle"

    def __init__(self, ctx: Context, config: Mapping[str, Any]) -> None:
        super().__init__(ctx, config)
        self.fallback_max_words = config["fallbackMaxWords"]
        self.fallback_max_bytes = config["fallbackMaxBytes"]
        self.max_title_bytes = config["maxTitleBytes"]
        for name, value in (
            ("fallbackMaxWords", self.fallback_max_words),
            ("fallbackMaxBytes", self.fallback_max_bytes),
            ("maxTitleBytes", self.max_title_bytes),
        ):
            _positive(name, value)
        if self.fallback_max_bytes > self.max_title_bytes:
            raise ValueError("fallbackMaxBytes must not exceed maxTitleBytes")
        ctx.on("session/event", self._on_event)

    def get(self, session: Any) -> Mapping[str, Any] | None:
        return fold_session_title(session.snapshot_events())

    def rename(self, session: Any, title: str) -> Mapping[str, Any]:
        normalized = normalize_session_title(title, self.max_title_bytes)
        if not normalized:
            raise SessionTitleInvalidError("session title must contain visible characters")
        session.append(
            "session/title",
            {"title": normalized, "messageSeqs": [], "source": {"kind": "user"}},
        )
        snapshot = self.get(session)
        assert snapshot is not None
        return snapshot

    def _on_event(self, session: Any, event: Mapping[str, Any]) -> None:
        if event["type"] != "user/message" or event["data"]["source"]["kind"] != "user":
            return
        if self.get(session) is not None:
            return
        text = "\n".join(
            block["text"] for block in event["data"]["content"] if block["type"] == "text"
        )
        title = fallback_session_title(text, self.fallback_max_words, self.fallback_max_bytes)
        if title:
            asyncio.get_running_loop().call_soon(
                self._append_fallback, session, title, event["seq"]
            )

    def _append_fallback(self, session: Any, title: str, seq: int) -> None:
        if self.get(session) is not None:
            return
        session.append(
            "session/title",
            {
                "title": title,
                "messageSeqs": [seq],
                "source": {"kind": "fallback"},
            },
        )


__all__ = [
    "SessionTitleInvalidError",
    "SessionTitleService",
    "fallback_session_title",
    "fold_session_title",
    "normalize_session_title",
    "truncate_title_utf8",
]
