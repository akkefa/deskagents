"""Ported from packages/session/session-format-v3-to-v4/tests/message-sources.spec.ts.

Upstream restores through ``createSessionFormatCatalogWithChildren([])``; for v4 input
that is :data:`session_format_catalog`. Not ported (they migrate v3 input, which this
Python build does not read yet): "migrates external producer %s without interpreting
inherited object keys" and "retains direct source kind %s through migration and native
restore".
"""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEventCollector,
    SessionFormatRestoreOptions,
)
from deskagents.session.format_catalog import session_format_catalog
from deskagents.session.format_v3_to_v4 import released_v4_session_format_codec
from deskagents.session.format_v3_to_v4.message_sources import (
    assert_v4_message_sources,
    assert_v4_source_row_admission,
)
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-format-v3-to-v4/tests/message-sources.spec.ts"
GROUP = "V4 source admission"

HEADER: dict[str, Any] = {
    "type": "session",
    "version": 4,
    "id": "sources",
    "createdAt": 1,
    "delegationDepth": 0,
    "isSeeded": False,
}


def event(source: Any) -> dict[str, Any]:
    return {
        "type": "user/message",
        "seq": 0,
        "time": 1,
        "surfaceOp": "append",
        "data": {
            "id": "u",
            "role": "user",
            "source": source,
            "content": [{"type": "text", "text": "hello"}],
        },
    }


def restore(source: Any) -> SessionFormatArtifact:
    reader = session_format_catalog.create_restore(
        HEADER, SessionFormatRestoreOptions(recovery="strict", validation="current")
    )
    reader.decode_row(parse_json(stringify_json(event(source))))
    return reader.finish()


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects missing, malformed, empty, and retired source kinds"
)
def test_rejects_missing_malformed_empty_and_retired_source_kinds() -> None:
    sources: list[Any] = [
        None,
        [],
        {},
        {"kind": ""},
        {"kind": 1},
        {"kind": "plugin", "plugin": "compact"},
    ]
    for source in sources:
        with pytest.raises(SessionFormatError, match=r"producer-owned source kind"):
            restore(source)
        with pytest.raises(SessionFormatError, match=r"producer-owned source kind"):
            assert_v4_message_sources(event(source))


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects retired wrappers in every declared message slot before recovery"
    " can discard them"
)
def test_rejects_retired_wrappers_in_every_message_slot() -> None:
    message = event({"kind": "plugin", "plugin": "compact"})["data"]
    rows: list[dict[str, Any]] = [
        event({"kind": "plugin", "plugin": "compact"}),
        *(
            {"type": type_, "seq": 0, "time": 1, "data": {"message": message}}
            for type_ in ["system/message", "assistant/message", "tool/result"]
        ),
        {"type": "agent/inbox/spliced", "seq": 0, "time": 1, "data": {"inserted": [message]}},
        {"type": "session/title-llm-request", "seq": 0, "time": 1, "data": {"messages": [message]}},
    ]
    for row in rows:
        decoder = released_v4_session_format_codec.create_decoder(HEADER, "recoverable")
        output = SessionFormatEventCollector()
        decoder.decode_row(None, output)
        with pytest.raises(SessionFormatError, match=r"producer-owned source kind"):
            decoder.decode_row(row, output)
        with pytest.raises(SessionFormatError, match=r"producer-owned source kind"):
            session_format_catalog.encode_current_event(row)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > leaves malformed incomplete rows for normal scanner corruption handling"
)
def test_leaves_malformed_incomplete_rows_for_scanner() -> None:
    for row in [
        None,
        {"type": "user/message", "data": None},
        {"type": "system/message", "data": {}},
        {"type": "agent/inbox/spliced", "data": {"inserted": False}},
        {"type": "agent/inbox/spliced", "data": {"inserted": [None]}},
        {
            "type": "session/title-llm-request",
            "data": {"messages": [event({"kind": "user"})["data"]]},
        },
        {"type": "external/event", "data": {"source": {"kind": "plugin"}}},
    ]:
        assert_v4_source_row_admission(row)
