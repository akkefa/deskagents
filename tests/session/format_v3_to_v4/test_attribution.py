"""Ported from packages/session/session-format-v3-to-v4/tests/attribution.spec.ts.

Unknown producer attribution survives the native codec and detached Session reader.
"""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.core.session import Session
from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatEventCollector,
)
from deskagents.session.format_v3_to_v4 import released_v4_session_format_codec as codec
from deskagents.session.format_v3_to_v4 import restore_released_v4_artifact
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-format-v3-to-v4/tests/attribution.spec.ts"
GROUP = "uninstalled producer attribution"

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "unknown-attribution",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > preserves %s source metadata through decoding, adoption and derivation"
)
@pytest.mark.parametrize("role", ["user", "developer"])
def test_preserves_source_metadata_through_decoding_adoption_and_derivation(role: str) -> None:
    message: dict[str, Any] = {
        "id": "attributed-message",
        "role": role,
        "source": {
            "kind": "external-attribution",
            "location": {"file": "notes.txt", "lines": [2, 5]},
            "enabled": False,
        },
        "content": [{"type": "text", "text": "Retain this context."}],
    }
    input_: list[dict[str, Any]] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 2, "data": {"turn": 1, "step": 1}},
        {
            "type": f"{role}/message",
            "seq": 2,
            "time": 3,
            "surfaceOp": "append",
            "data": message if role == "user" else {"turn": 1, "step": 1, "message": message},
        },
        {"type": "step/end", "seq": 3, "time": 4, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 4,
            "time": 5,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    physical = [stringify_json(codec.encode_event(event)) for event in input_]
    output = SessionFormatEventCollector()
    decoder = codec.create_decoder(codec.encode_header(HEADER, 0), "strict")
    for row in physical:
        decoder.decode_row(parse_json(row), output)
    inherited_event_count = decoder.finish(output)
    artifact = SessionFormatArtifact(
        header=HEADER, events=output.values, inherited_event_count=inherited_event_count
    )
    restored = restore_released_v4_artifact(artifact, frozenset(event["type"] for event in input_))
    assert list(restored.events) == input_
    # The format reader validates stored JSON; adoption validates the current Session fields.
    session = Session.from_restore(
        HEADER["id"],
        restored.events,
        restored.header,
        restored.inherited_event_count,
        "detached",
    )
    assert session.derive_messages() == [message]
    assert [stringify_json(codec.encode_event(event)) for event in restored.events] == physical
