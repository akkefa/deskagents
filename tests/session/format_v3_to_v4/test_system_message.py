"""Ported from packages/session/session-format-v3-to-v4/tests/system-message.spec.ts.

Upstream derives the system-message source fixture by migrating a v3
``{kind: 'plugin', plugin: '@deepseek-ai/dsh-system-prompt'}`` source through
``createSessionFormatCatalogWithChildren([])``. Python does not read v3, so the fixture
is the migration's documented result (``producerKind`` in ``sources.ts``):
``{kind: 'system-prompt'}``. Both cases themselves only exercise v4 input.
"""

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
from deskagents.session.format_v3_to_v4.codec import released_v4_session_format_codec
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-format-v3-to-v4/tests/system-message.spec.ts"
GROUP = "native system-message fields"

HEADER: dict[str, Any] = {
    "type": "session",
    "version": 4,
    "id": "system-fields",
    "createdAt": 1,
    "delegationDepth": 0,
    "isSeeded": False,
}
SOURCE: dict[str, Any] = {"kind": "system-prompt"}
ATTACHMENT: dict[str, Any] = {
    "attachmentId": "sha256:image",
    "mediaType": "image/png",
    "bytes": 1,
    "width": 1,
    "height": 1,
}
STRICT_CURRENT = SessionFormatRestoreOptions(recovery="strict", validation="current")


def message(content: Any) -> dict[str, Any]:
    return {"id": "system", "role": "system", "source": SOURCE, "content": content}


def row(data: Any) -> SessionFormatEvent:
    return {"type": "system/message", "seq": 2, "time": 1, "surfaceOp": "append", "data": data}


def restored(data: Any) -> SessionFormatArtifact:
    read = session_format_catalog.create_restore(HEADER, STRICT_CURRENT)
    events: list[SessionFormatEvent] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 1, "data": {"turn": 1, "step": 1}},
        row(data),
    ]
    for event in events:
        read.decode_row(
            parse_json(stringify_json(session_format_catalog.encode_current_event(event)))
        )
    return read.finish()


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > preserves required fields and extra JSON on known and extensible block "
    "types"
)
def test_preserves_required_fields_and_extra_json() -> None:
    content = [
        {"type": "text", "text": "", "extension": {"keep": True}},
        {"type": "reasoning", "text": "reasoning"},
        {"type": "tool-call", "id": "call", "name": "read", "arguments": "{}"},
        {"type": "file", "extension": True},
        {"type": "image", "attachment": ATTACHMENT},
        {
            "type": "image",
            "attachment": {
                **ATTACHMENT,
                "name": "name",
                "originalDimensions": {"width": 1, "height": 2},
                "extension": True,
            },
        },
    ]
    data = {"turn": 1, "step": 1, "message": message(content), "extra": True}
    assert restored(data).events[2]["data"] == data


def _invalid_data() -> list[Any]:
    attachments: list[Any] = [
        None,
        {**ATTACHMENT, "mediaType": "text/plain"},
        {**ATTACHMENT, "attachmentId": ""},
        {**ATTACHMENT, "bytes": -1},
        {**ATTACHMENT, "width": 0},
        {**ATTACHMENT, "height": 0},
        {**ATTACHMENT, "name": 1},
        {**ATTACHMENT, "originalDimensions": None},
        {**ATTACHMENT, "originalDimensions": {"width": 0, "height": 1}},
        {**ATTACHMENT, "originalDimensions": {"width": 1, "height": 0}},
    ]
    contents: list[Any] = [
        None,
        [None],
        [{}],
        [{"type": ""}],
        [{"type": "text", "text": 1}],
        [{"type": "tool-call", "id": ""}],
        [{"type": "tool-call", "id": "id", "name": ""}],
        [{"type": "tool-call", "id": "id", "name": "name", "arguments": None}],
        [{"type": "tool-result", "toolCallId": "call", "content": []}],
        *([{"type": "image", "attachment": value}] for value in attachments),
    ]
    messages: list[Any] = [
        {**message([]), "id": ""},
        {**message([]), "role": "user"},
        *(message(content) for content in contents),
    ]
    return [
        None,
        {"turn": 0, "step": 1},
        {"turn": 1, "step": 0},
        {"turn": 1, "step": 1, "message": None},
        *({"turn": 1, "step": 1, "message": value} for value in messages),
    ]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects malformed fields before a recoverable scanner can discard the row"
)
def test_rejects_malformed_fields_before_a_recoverable_scanner_can_discard_the_row() -> None:
    for data in _invalid_data():
        with pytest.raises(SessionFormatError):
            restored(data)
        read = released_v4_session_format_codec.create_decoder(HEADER, "recoverable")
        output = SessionFormatEventCollector()
        read.decode_row(None, output)
        with pytest.raises(SessionFormatError):
            read.decode_row(row(data), output)
