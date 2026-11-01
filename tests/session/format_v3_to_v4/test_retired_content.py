"""Ported from packages/session/session-format-v3-to-v4/tests/retired-content.spec.ts.

Not ported (need v3 input through ``historicalSessionFormatCatalog`` or the v3->v4
migration of ``createSessionFormatCatalogWithChildren``): "refuses a V3 user wrapper at
the V4 target while leaving the released reader and source intact" and "keeps tool
arguments, replay state, schemas, nested metadata, and unknown ignorable payloads
opaque" (its final assertion migrates a v3 opaque row).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatEventCollector,
)
from deskagents.session.format_v3_to_v4 import (
    assert_v4_row_admission,
    released_v4_session_format_codec,
    restore_released_v4_artifact,
)

SPEC = "packages/session/session-format-v3-to-v4/tests/retired-content.spec.ts"
GROUP = "retired tool-result content admission"

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "retired-content",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}
WRAPPER: dict[str, Any] = {
    "type": "tool-result",
    "toolCallId": "retired-call",
    "content": [{"type": "text", "text": "saved result"}],
}
CONTENT = [WRAPPER]
USER: dict[str, Any] = {
    "id": "user",
    "role": "user",
    "source": {"kind": "user"},
    "content": CONTENT,
}
DEFAULT_ERROR = r"released tool-result wrapper"


def message(role: str) -> dict[str, Any]:
    source = (
        {"kind": "model", "provider": "test", "model": "test"}
        if role == "assistant"
        else {"kind": "tool", "callId": "call"}
        if role == "tool"
        else {"kind": f"{role}-prompt"}
    )
    return {
        "id": role,
        "role": role,
        "content": CONTENT,
        "source": source,
        **({"toolCallId": "call"} if role == "tool" else {}),
    }


def stream_case(type_: str, kind: str) -> tuple[str, str, dict[str, Any], str]:
    chunk = (
        {"type": kind, "index": 0, "blockType": "tool-result"}
        if kind == "block-start"
        else {"type": kind, "index": 0, "block": WRAPPER}
    )
    return (
        f"{type_} stream {kind}",
        type_,
        {
            "turn": 1,
            "step": 1,
            "message": {**message("assistant"), "content": []},
            "stream": [{"type": "chunk", "elapsedMs": 0, "chunk": chunk}],
        },
        DEFAULT_ERROR,
    )


CASES: list[tuple[str, str, dict[str, Any], str]] = [
    ("user message", "user/message", USER, DEFAULT_ERROR),
    *[
        (
            f"{role} message",
            f"{role}/message",
            {"turn": 1, "step": 1, "message": message(role)},
            r"rejects retired tool-result" if role == "system" else DEFAULT_ERROR,
        )
        for role in ["system", "developer", "assistant"]
    ],
    (
        "native tool message",
        "tool/result",
        {"turn": 1, "step": 1, "message": message("tool")},
        DEFAULT_ERROR,
    ),
    (
        "queued input",
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "inserted": [USER]},
        DEFAULT_ERROR,
    ),
    ("title request", "session/title-llm-request", {"messages": [USER]}, DEFAULT_ERROR),
    ("compaction summary", "compaction/summary", {"summary": CONTENT}, DEFAULT_ERROR),
    (
        "compaction raw output",
        "compaction/summary",
        {"summary": [], "rawOutput": CONTENT},
        DEFAULT_ERROR,
    ),
    ("PTC result", "tool/ptc-dispatch", {"content": CONTENT}, DEFAULT_ERROR),
    ("team message", "team/message/queued", {"message": USER}, DEFAULT_ERROR),
    *[
        stream_case(type_, kind)
        for type_ in ["assistant/message", "assistant/attempt"]
        for kind in ["block-start", "block-end"]
    ],
]


def row(type_: str, data: dict[str, Any]) -> SessionFormatEvent:
    return {"type": type_, "data": data, "seq": 0, "time": 1}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses $name in native encoding, restoration and recoverable suffixes"
)
@pytest.mark.parametrize(("name", "type_", "data", "error"), CASES, ids=[case[0] for case in CASES])
def test_refuses_retired_content(name: str, type_: str, data: dict[str, Any], error: str) -> None:
    event = row(type_, data)
    before = copy.deepcopy(event)
    with pytest.raises(SessionFormatError, match=error):
        released_v4_session_format_codec.encode_event(event)
    with pytest.raises(SessionFormatError, match=error):
        restore_released_v4_artifact(
            SessionFormatArtifact(header=HEADER, inherited_event_count=0, events=[event]), {type_}
        )
    for recovery in ("strict", "recoverable"):
        decoder = released_v4_session_format_codec.create_decoder(
            {"type": "session", **HEADER}, recovery
        )
        collector = SessionFormatEventCollector()
        if recovery == "recoverable":
            decoder.decode_row(None, collector)
        with pytest.raises(SessionFormatError, match=error):
            decoder.decode_row(event, collector)
    assert event == before


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > interprets ignorable developer content only for a reader that knows "
    "the event, or for encoding"
)
def test_interprets_ignorable_developer_content_only_when_known() -> None:
    event: SessionFormatEvent = {
        **row("developer/message", {"turn": 1, "step": 1, "message": message("developer")}),
        "ignorable": True,
    }
    collector = SessionFormatEventCollector()
    decoder = released_v4_session_format_codec.create_decoder(
        {"type": "session", **HEADER}, "strict"
    )
    decoder.decode_row(event, collector)
    assert collector.values == [event]
    assert_v4_row_admission(event, set())
    with pytest.raises(SessionFormatError, match=r"released tool-result wrapper"):
        assert_v4_row_admission(event, {"developer/message"})
    with pytest.raises(SessionFormatError, match=r"released tool-result wrapper"):
        released_v4_session_format_codec.encode_event(event)
    artifact = SessionFormatArtifact(header=HEADER, inherited_event_count=0, events=[event])
    assert restore_released_v4_artifact(artifact, set()) is artifact
