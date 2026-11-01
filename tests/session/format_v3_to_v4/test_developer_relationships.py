"""Ported from packages/session/session-format-v3-to-v4/tests/developer-relationships.spec.ts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from deskagents.core.session import Session
from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEventCollector,
)
from deskagents.session.format_v3_to_v4 import (
    assert_v4_row_admission,
    restore_released_v4_artifact,
)
from deskagents.session.format_v3_to_v4 import (
    released_v4_session_format_codec as codec,
)

SPEC = "packages/session/session-format-v3-to-v4/tests/developer-relationships.spec.ts"
GROUP = "V4 developer relationship admission"

Row = dict[str, Any]

COORDINATES: dict[str, Any] = {"turn": 1, "step": 1}
BEGIN: list[Row] = [
    {"type": "turn/start", "data": {"turn": 1}},
    {"type": "step/start", "data": COORDINATES},
]
SYSTEM: Row = {
    "type": "system/message",
    "surfaceOp": "append",
    "data": {
        **COORDINATES,
        "message": {
            "id": "system",
            "role": "system",
            "source": {"kind": "system-prompt"},
            "content": [{"type": "text", "text": "System"}],
        },
    },
}
EMPTY_DEVELOPER: Row = {
    "type": "developer/message",
    "surfaceOp": "append",
    "data": {
        **COORDINATES,
        "message": {
            "id": "developer",
            "role": "developer",
            "source": {"kind": "tool-registry"},
            "content": [],
        },
    },
}


def artifact(rows: Sequence[Row]) -> SessionFormatArtifact:
    return SessionFormatArtifact(
        header={
            "version": 4,
            "id": "developer-relations",
            "createdAt": 1,
            "isSeeded": False,
            "delegationDepth": 0,
        },
        events=[{**row, "seq": seq, "time": seq + 1} for seq, row in enumerate(rows)],
        inherited_event_count=0,
    )


def restore(rows: Sequence[Row]) -> SessionFormatArtifact:
    input_ = artifact(rows)
    return restore_released_v4_artifact(input_, {str(event["type"]) for event in input_.events})


def _plain(value: SessionFormatArtifact) -> Any:
    return json.dumps(
        {
            "header": value.header,
            "events": list(value.events),
            "inheritedEventCount": value.inherited_event_count,
        }
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > retains an empty developer position while replacing the protected system "
    "head"
)
def test_retains_an_empty_developer_position_while_replacing_the_protected_system_head() -> None:
    replacement: Row = {
        **SYSTEM,
        "surfaceOp": {"op": "replace", "startSeq": 2, "endSeq": 2},
        "sourceEventSeqs": [2],
        "data": {
            **COORDINATES,
            "message": {
                "id": "replacement",
                "role": "system",
                "source": {"kind": "system-prompt"},
                "content": [{"type": "text", "text": "Updated"}],
            },
        },
    }
    input_ = restore([*BEGIN, SYSTEM, EMPTY_DEVELOPER, replacement])
    session = Session.from_restore(
        str(input_.header["id"]), list(input_.events), input_.header, 0, "detached"
    )
    assert list(session.surface.nodes) == [4, 3]
    assert [message["id"] for message in session.derive_messages()] == ["replacement"]
    assert input_.events[3]["data"] == EMPTY_DEVELOPER["data"]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects an empty developer replacement of the protected system head"
)
def test_rejects_an_empty_developer_replacement_of_the_protected_system_head() -> None:
    replacement = {
        **EMPTY_DEVELOPER,
        "surfaceOp": {"op": "replace", "startSeq": 2, "endSeq": 2},
        "sourceEventSeqs": [2],
    }
    with pytest.raises(SessionFormatError, match=r"protected system head"):
        restore([*BEGIN, SYSTEM, replacement])


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > includes empty developer nodes in compaction spans and protects the "
    "system head"
)
def test_includes_empty_developer_nodes_in_compaction_spans() -> None:
    prune: Row = {
        "type": "compaction/prune",
        "data": {
            "shadowedRange": {"start": 3, "end": 3},
            "shadowedSeqs": [3],
            "shadowedTokenCount": 0,
        },
    }
    restore([*BEGIN, SYSTEM, EMPTY_DEVELOPER, prune])
    with pytest.raises(SessionFormatError, match=r"protected system head"):
        restore(
            [
                *BEGIN,
                SYSTEM,
                EMPTY_DEVELOPER,
                {
                    **prune,
                    "data": {
                        "shadowedRange": {"start": 2, "end": 3},
                        "shadowedSeqs": [2, 3],
                        "shadowedTokenCount": 0,
                    },
                },
            ]
        )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > requires known developer messages to match the current open step %#"
)
@pytest.mark.parametrize(
    "rows",
    [
        [BEGIN[0], EMPTY_DEVELOPER],
        [
            *BEGIN,
            {
                **EMPTY_DEVELOPER,
                "data": {
                    **COORDINATES,
                    "step": 2,
                    "message": {
                        "id": "developer",
                        "role": "developer",
                        "source": {"kind": "tool-registry"},
                        "content": [],
                    },
                },
            },
        ],
    ],
)
def test_requires_known_developer_messages_to_match_the_current_open_step(
    rows: list[Row],
) -> None:
    with pytest.raises(SessionFormatError, match=r"open.*step"):
        restore(rows)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > defers ignorable developer payloads during physical decoding while "
    "keeping encoder validation"
)
def test_defers_ignorable_developer_payloads_during_physical_decoding() -> None:
    opaque: Row = {
        "type": "developer/message",
        "ignorable": True,
        "data": {"message": {"role": "developer", "source": {"kind": "plugin"}}, "future": True},
        "surfaceOp": {"future": True},
    }
    input_ = artifact([opaque, *BEGIN, SYSTEM])
    physical_header = codec.encode_header(input_.header, 0)
    decoded = SessionFormatEventCollector()
    decoder = codec.create_decoder(physical_header, "strict")
    for event in input_.events:
        decoder.decode_row(event, decoded)
    decoder.finish(decoded)
    assert decoded.values == list(input_.events)
    with pytest.raises(SessionFormatError):
        codec.encode_event(input_.events[0])
    assert_v4_row_admission(input_.events[0], set())
    with pytest.raises(SessionFormatError):
        assert_v4_row_admission(input_.events[0], {"developer/message"})

    known = {**input_.events[0], "data": EMPTY_DEVELOPER["data"]}
    encoded = codec.encode_event(known)
    assert encoded["type"] == "developer/message"
    assert encoded["ignorable"] is True


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps unknown ignorable developer fields and coordinates opaque before an "
    "ordinary step"
)
def test_keeps_unknown_ignorable_developer_fields_and_coordinates_opaque() -> None:
    opaque: Row = {
        "type": "developer/message",
        "ignorable": True,
        "data": {"message": {"role": "developer", "source": {"kind": "plugin"}}, "future": True},
        "surfaceOp": {"future": True},
        "sourceEventSeqs": {"future": True},
    }
    input_ = artifact([opaque, *BEGIN, SYSTEM])
    before = _plain(input_)
    known = {str(event["type"]) for event in input_.events}
    known.discard("developer/message")
    assert restore_released_v4_artifact(input_, known) is input_
    assert _plain(input_) == before
    assert [event["seq"] for event in input_.events] == [0, 1, 2, 3]
    known.add("developer/message")
    with pytest.raises(SessionFormatError):
        restore_released_v4_artifact(input_, known)
