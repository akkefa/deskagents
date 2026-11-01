"""Incremental session-log contribution for DeepSeek requests.

Upstream: packages/session/session-log-deepseek at ce7073d.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from deepseek_cordis import Context, Service

from deskagents.core.session import Session, SessionStore
from deskagents.llm.deepseek_llm_api_extensions import DeepSeekLlmApiExtensionRegistry


def accepted_through(session: Session) -> int:
    through = -1
    for event in session.snapshot_events():
        if event["type"] != "session-log-deepseek/delivery-accepted":
            continue
        data = event["data"]
        if (
            data.get("sessionId") == session.id
            and data.get("sessionFormatVersion") == session.header["version"]
            and isinstance(data.get("throughSeq"), int)
            and data["throughSeq"] < event["seq"]
        ):
            through = max(through, data["throughSeq"])
    return through


def _wire_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: event[key]
        for key in (
            "type",
            "seq",
            "time",
            "data",
            "surfaceOp",
            "sourceEventSeqs",
            "ignorable",
        )
        if key in event
    }


class _Provider:
    def __init__(self, service: SessionLogDeepSeekService) -> None:
        self.service = service

    def prepare(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if not self.service.enabled or "sessionId" not in request:
            return None
        session = self.service.sessions.get(str(request["sessionId"]))
        if session is None:
            return None
        after = accepted_through(session)
        pending = session.snapshot_events()[after + 1 :]
        header = session.header
        envelope = {
            "version": 1,
            "sessionFormatVersion": header["version"],
            "session": {
                "version": header["version"],
                "id": str(header["id"]),
                "createdAt": header["createdAt"],
                **({"cwd": header["cwd"]} if "cwd" in header else {}),
                **(
                    {"parentSession": str(header["parentSession"])}
                    if "parentSession" in header
                    else {}
                ),
                **(
                    {"seedLength": session.inherited_event_count}
                    if header.get("isSeeded") is True
                    else {}
                ),
            },
            "afterSeq": after,
        }
        events: list[dict[str, Any]] = []
        through = after
        for event in pending:
            candidate = [*events, _wire_event(event)]
            value = {**envelope, "throughSeq": event["seq"], "events": candidate}
            if len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) > (
                self.service.max_bytes
            ):
                break
            events = candidate
            through = event["seq"]
        if not events:
            return None
        value = {**envelope, "throughSeq": through, "events": events}

        def accept() -> None:
            session.append(
                "session-log-deepseek/delivery-accepted",
                {
                    "sessionId": session.id,
                    "sessionFormatVersion": session.header["version"],
                    "throughSeq": through,
                },
            )

        return {"value": value, "accept": accept}


class SessionLogDeepSeekService(Service):
    name = "sessionLogDeepSeek"

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, config)
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.max_bytes = int(config.get("maxBytes", 8 * 1024 * 1024))
        if self.max_bytes < 1:
            raise ValueError("maxBytes must be positive")
        sessions = ctx.get("sessions")
        extensions = ctx.get("deepseekLlmApiExtensions")
        if not isinstance(sessions, SessionStore):
            raise RuntimeError("session-log-deepseek requires SessionStore")
        if not isinstance(extensions, DeepSeekLlmApiExtensionRegistry):
            raise RuntimeError("session-log-deepseek requires DeepSeekLlmApiExtensionRegistry")
        self.sessions = sessions
        extensions.register("dsh_session_log", _Provider(self), ctx=ctx)


__all__ = ["SessionLogDeepSeekService", "accepted_through"]
