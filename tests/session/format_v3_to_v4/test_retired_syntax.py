"""Ported from packages/session/session-format-v3-to-v4/tests/retired-syntax.spec.ts."""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatEventCollector,
    SessionFormatRestoreOptions,
)
from deskagents.session.format_catalog import session_format_catalog
from deskagents.session.format_v3_to_v4 import (
    released_v4_session_format_codec,
    restore_released_v4_artifact,
)
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-format-v3-to-v4/tests/retired-syntax.spec.ts"
GROUP = "native retired syntax admission"

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "native-syntax",
    "createdAt": 1,
    "delegationDepth": 0,
    "isSeeded": False,
}
PHYSICAL: dict[str, Any] = {"type": "session", **HEADER}
TURN: SessionFormatEvent = {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
STRICT_CURRENT = SessionFormatRestoreOptions(recovery="strict", validation="current")


def request(data: Any) -> SessionFormatEvent:
    return {"type": "request/header", "seq": 1, "time": 1, "data": data}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects retired headers and PTC tags in strict reads, encoding, and "
    "recoverable suffixes"
)
def test_rejects_retired_headers_and_ptc_tags() -> None:
    invalid: list[SessionFormatEvent] = [
        request(None),
        request({"header": None}),
        request({"header": {"config": {"provider": "mock", "model": "mock"}, "system": ""}}),
        {"type": "tool/code-dispatch-start", "seq": 1, "time": 1, "data": {}},
        {"type": "tool/code-dispatch", "seq": 1, "time": 1, "data": {}},
    ]
    for event in invalid:
        with pytest.raises(SessionFormatError):
            session_format_catalog.encode_current_event(event)
        read = session_format_catalog.create_restore(PHYSICAL, STRICT_CURRENT)
        read.decode_row(TURN)
        with pytest.raises(SessionFormatError):
            read.decode_row(event)
        scanner = released_v4_session_format_codec.create_decoder(PHYSICAL, "recoverable")
        output = SessionFormatEventCollector()
        scanner.decode_row(None, output)
        with pytest.raises(SessionFormatError):
            scanner.decode_row(event, output)
        artifact = SessionFormatArtifact(
            header=HEADER, inherited_event_count=0, events=[TURN, event]
        )
        with pytest.raises(SessionFormatError):
            restore_released_v4_artifact(artifact, {str(TURN["type"]), str(event["type"])})


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps known request fields open and obsolete ignorable tags opaque"
)
def test_keeps_known_request_fields_open_and_obsolete_ignorable_tags_opaque() -> None:
    rows: list[SessionFormatEvent] = [
        TURN,
        request(
            {
                "reason": "initial",
                "header": {"config": {"provider": "mock", "model": "mock"}, "extra": True},
            }
        ),
        {
            "type": "tool/code-dispatch-start",
            "seq": 2,
            "time": 1,
            "data": {"unknown": True},
            "ignorable": True,
        },
        {"type": "tool/code-dispatch", "seq": 3, "time": 1, "data": None, "ignorable": True},
    ]
    read = session_format_catalog.create_restore(PHYSICAL, STRICT_CURRENT)
    for event in rows:
        read.decode_row(
            parse_json(stringify_json(session_format_catalog.encode_current_event(event)))
        )
    assert list(read.finish().events) == rows
