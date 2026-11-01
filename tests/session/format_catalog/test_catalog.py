"""Ported from packages/session/session-format-catalog/tests/catalog.spec.ts.

Only the v4 case is applicable. Not ported: "supplies every required current message
interpreter and validates its durable references" (needs the image-offload
projection, not ported yet) and every case that restores v0-v3 input ("statically owns
every adjacent edge through the current writer", "restores v%i empty and non-empty
inherited prefixes through every adjacent edge", "inserts the v3 system head and
remaps the inherited cut (seeded=%s)", "refuses unclassified ignorable v2 events
(%s)", "migrates frozen v%i PTC records ...", "rejects obsolete required tags in V3
input ...", "refuses V3 dispatch tag collisions ...", "renames V2-owned ignorable
dispatch tags ...", "refuses a v3 delivery marker in v2 input (%s)", "finishes a
streaming migration with the target header and dense rows" and both "historical child
prerequisite catalog" cases). The extra tests below cover the Python-only refusal of
v0-v3 and a byte-level v4 round trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from deskagents.core.session import SESSION_FORMAT_VERSION
from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatRestoreOptions,
    SessionFormatUnsupportedMigrationError,
)
from deskagents.session.format_catalog import session_format_catalog
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-format-catalog/tests/catalog.spec.ts"
GROUP = "first-party Session format catalog"
STRICT_CURRENT = SessionFormatRestoreOptions(recovery="strict", validation="current")


def restore(header: object, rows: Sequence[object]) -> SessionFormatArtifact:
    current = session_format_catalog.create_restore(header, STRICT_CURRENT)
    for row in rows:
        current.decode_row(row)
    return current.finish()


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > restores the installed current vocabulary without freezing ordinary "
    "payload additions"
)
def test_restores_installed_vocabulary_with_payload_additions() -> None:
    header = {
        "type": "session",
        "version": SESSION_FORMAT_VERSION,
        "id": "current-growth",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    extended = restore(
        header,
        [
            {
                "type": "turn/start",
                "seq": 0,
                "time": 1,
                "data": {"turn": 1, "postReleaseMember": True},
            }
        ],
    )
    assert extended.events == [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1, "postReleaseMember": True}}
    ]
    with pytest.raises(SessionFormatUnsupportedMigrationError, match="unknown event type"):
        restore(header, [{"type": "ordinary/not-installed", "seq": 0, "time": 1, "data": "future"}])
    extension = restore(
        header,
        [{"type": "ordinary/external", "seq": 0, "time": 1, "data": None, "ignorable": True}],
    )
    assert extension.events == [
        {"type": "ordinary/external", "seq": 0, "time": 1, "data": None, "ignorable": True}
    ]


@pytest.mark.parametrize("version", [0, 1, 2, 3])
def test_unported_formats_are_unsupported_not_malformed(version: int) -> None:
    """Python-only: v0-v3 are refused until their codecs and migrations are ported."""
    header = {
        "type": "session",
        "version": version,
        "id": "old",
        "createdAt": 1,
        "delegationDepth": 0,
    }
    result = session_format_catalog.read_header(header)
    assert (result.status, result.stored_version, result.target_version) == (
        "unsupported",
        version,
        4,
    )
    assert result.reason is not None and "not supported by this Python build yet" in result.reason
    with pytest.raises(
        SessionFormatUnsupportedMigrationError, match="not supported by this Python build"
    ):
        session_format_catalog.create_restore(header, STRICT_CURRENT)


def test_current_records_round_trip_through_json_text() -> None:
    """Encode with the v4 encoder, serialize like dsh, parse and restore unchanged."""
    header: dict[str, Any] = {
        "version": 4,
        "id": "round-trip",
        "createdAt": 1700000000000,
        "cwd": "/work",
        "isSeeded": False,
        "delegationDepth": 0,
    }
    events: list[dict[str, Any]] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 2, "data": {"turn": 1, "step": 1}},
        {
            "type": "user/message",
            "seq": 2,
            "time": 3,
            "data": {
                "id": "user-1",
                "role": "user",
                "content": [{"type": "text", "text": "héllo   \U0001f600"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {"type": "feedback/record", "seq": 3, "time": 4, "data": {"text": "x"}, "ignorable": True},
        {"type": "step/end", "seq": 4, "time": 5, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 5,
            "time": 6,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    header_line = stringify_json(session_format_catalog.encode_current_header(header, 0))
    assert header_line.startswith('{"type":"session","version":4,')
    lines = [stringify_json(session_format_catalog.encode_current_event(event)) for event in events]
    restored = restore(parse_json(header_line), [parse_json(line) for line in lines])
    assert restored == SessionFormatArtifact(header, 0, events)
    assert list(restored.header) == list(header)
