"""Ported from packages/session/session-format-v3-to-v4/tests/canonical.spec.ts.

Not ported (they restore a v2 artifact through the v2->v3->v4 migrations, which this
Python build does not read yet): "composes seeded systems, source ownership, PTC,
canonical replacements and native round-trip" and "rejects noncanonical native
metadata %j". "keeps earlier-generation delivery markers ignored and validates native
current ownership" is ported without its final assertion, which restores a v3 header
through ``createSessionFormatCatalogWithChildren``.
"""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEventCollector,
)
from deskagents.session.format_v3_to_v4 import (
    released_v4_session_format_codec,
    restore_released_v4_artifact,
)

SPEC = "packages/session/session-format-v3-to-v4/tests/canonical.spec.ts"
GROUP = "canonical V4 integration"

HEADER: dict[str, Any] = {
    "type": "session",
    "version": 2,
    "id": "canonical-v4",
    "createdAt": 1,
    "isSeeded": True,
    "parentSession": "parent",
    "delegationDepth": 0,
}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > leaves common event and message validation to current Session restoration"
)
def test_leaves_common_validation_to_session_restoration() -> None:
    decoder = released_v4_session_format_codec.create_decoder(
        {**HEADER, "version": 4, "isSeeded": False}, "recoverable"
    )
    output = SessionFormatEventCollector()
    decoder.decode_row(None, output)
    assert decoder.finish(output) == 0


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps earlier-generation delivery markers ignored and validates native"
    " current ownership"
)
def test_keeps_earlier_generation_delivery_markers_ignored() -> None:
    def marker(version: int) -> dict[str, Any]:
        return {
            "type": "session-log-deepseek/delivery-accepted",
            "seq": 0,
            "time": 1,
            "data": {"sessionId": "other", "throughSeq": 0, "sessionFormatVersion": version},
        }

    logical = {key: value for key, value in HEADER.items() if key != "type"}
    artifact = SessionFormatArtifact(
        header={**logical, "version": 4, "isSeeded": False},
        inherited_event_count=0,
        events=[marker(3)],
    )
    assert restore_released_v4_artifact(artifact, frozenset([marker(3)["type"]])) is artifact
    with pytest.raises(SessionFormatError, match=r"wrong Session"):
        restore_released_v4_artifact(
            SessionFormatArtifact(
                header=artifact.header,
                inherited_event_count=artifact.inherited_event_count,
                events=[
                    {"type": "feedback/record", "seq": 0, "time": 1, "ignorable": True, "data": {}},
                    {**marker(4), "seq": 1},
                ],
            ),
            frozenset(["feedback/record", marker(4)["type"]]),
        )
    # Upstream then restores a v3 header through createSessionFormatCatalogWithChildren([]);
    # v3 input is not readable by this Python build.


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > retains native compaction checkpoint sources during restoration"
)
def test_retains_native_compaction_checkpoint_sources() -> None:
    events: list[dict[str, Any]] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "user/message",
            "seq": 1,
            "time": 2,
            "data": {
                "id": "input",
                "role": "user",
                "source": {"kind": "user"},
                "content": [{"type": "text", "text": "input"}],
            },
            "surfaceOp": "append",
        },
        {
            "type": "compaction/start",
            "seq": 2,
            "time": 3,
            "data": {"compactionId": "compact-1", "turn": 1},
        },
        {
            "type": "compaction/summary",
            "seq": 3,
            "time": 4,
            "data": {
                "compactionId": "compact-1",
                "summary": [{"type": "text", "text": "summary"}],
                "shadowedRange": {"start": 1, "end": 1},
                "shadowedSeqs": [1],
                "shadowedTokenCount": 1,
                "provider": "mock",
                "model": "mock",
            },
        },
        {
            "type": "user/message",
            "seq": 4,
            "time": 5,
            "data": {
                "id": "checkpoint",
                "role": "user",
                "source": {"kind": "compact-checkpoint", "compactionId": "compact-1"},
                "content": [{"type": "text", "text": "summary"}],
            },
            "surfaceOp": {"op": "replace", "startSeq": 1, "endSeq": 1},
            "sourceEventSeqs": [2, 3, 1],
        },
        {
            "type": "compaction/end",
            "seq": 5,
            "time": 6,
            "data": {"compactionId": "compact-1", "turn": 1},
        },
        {
            "type": "turn/end",
            "seq": 6,
            "time": 7,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    artifact = SessionFormatArtifact(
        header={
            "version": 4,
            "id": "compact-v4",
            "createdAt": 1,
            "isSeeded": False,
            "delegationDepth": 0,
        },
        inherited_event_count=0,
        events=events,
    )
    known = frozenset(event["type"] for event in events)
    assert restore_released_v4_artifact(artifact, known) is artifact
    assert events[4]["data"]["source"] == {
        "kind": "compact-checkpoint",
        "compactionId": "compact-1",
    }
