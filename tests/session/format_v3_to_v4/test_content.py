"""Ported from packages/session/session-format-v3-to-v4/tests/content.spec.ts.

Only the native v4 cases are ported. Every other "V3 content conversion" case and every
"unknown V3 event identities" case exercises the v3->v4 migration (``migrateV3Content``,
``migrateV3EventContent``, ``createSessionFormatV3ToV4``, ``namespaceV3OpaqueEvent``),
which is not in Python.
"""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatEventCollector,
)
from deskagents.session.format_v3_to_v4 import released_v4_session_format_codec

SPEC = "packages/session/session-format-v3-to-v4/tests/content.spec.ts"

IMAGE: dict[str, Any] = {
    "type": "image",
    "attachment": {
        "attachmentId": "image",
        "mediaType": "image/png",
        "bytes": 1,
        "width": 1,
        "height": 1,
    },
}
WRAPPER: dict[str, Any] = {
    "type": "future",
    "content": [{"type": "tool-result", "toolCallId": "opaque", "content": [IMAGE]}],
}
CONVERTED_BLOCK: dict[str, Any] = {**WRAPPER, "type": "plugin:future"}
USER: dict[str, Any] = {
    "id": "user",
    "role": "user",
    "source": {"kind": "user"},
    "content": [WRAPPER],
}
NATIVE_HEADER: dict[str, Any] = {
    "type": "session",
    "version": 4,
    "id": "native",
    "createdAt": 0,
    "isSeeded": False,
    "delegationDepth": 0,
}


def event(type_: str, data: Any) -> SessionFormatEvent:
    return {"type": type_, "seq": 0, "time": 1, "data": data}


@pytest.mark.upstream(
    f"{SPEC}::V3 content conversion > retains native V4 deferred-loading declaration validation"
)
def test_retains_native_deferred_loading_validation() -> None:
    tool: dict[str, Any] = {
        "name": "echo",
        "description": "Echo",
        "parameters": {},
        "deferLoading": True,
    }
    header: dict[str, Any] = {"config": {"provider": "mock", "model": "mock"}, "tools": [tool]}
    row = event("request/header", {"header": header})
    physical = released_v4_session_format_codec.encode_event(row)
    collector = SessionFormatEventCollector()
    released_v4_session_format_codec.create_decoder(NATIVE_HEADER, "strict").decode_row(
        physical, collector
    )
    assert collector.values == [row]
    with pytest.raises(SessionFormatError, match=r"deferLoading must be true"):
        released_v4_session_format_codec.encode_event(
            event(
                "request/header", {"header": {**header, "tools": [{**tool, "deferLoading": False}]}}
            )
        )


@pytest.mark.upstream(f"{SPEC}::native V4 opaque content > round trips %s extension content")
@pytest.mark.parametrize("role", ["system", "user"])
def test_round_trips_extension_content(role: str) -> None:
    message = {
        **USER,
        "role": role,
        "source": {"kind": "system-prompt" if role == "system" else "user"},
        "content": [CONVERTED_BLOCK, {"type": "plugin:future", "content": [WRAPPER]}],
    }
    row = event(
        f"{role}/message",
        message if role == "user" else {"turn": 1, "step": 1, "message": message},
    )
    physical = released_v4_session_format_codec.encode_event(row)
    collector = SessionFormatEventCollector()
    released_v4_session_format_codec.create_decoder(NATIVE_HEADER, "strict").decode_row(
        physical, collector
    )
    assert collector.values == [row]
