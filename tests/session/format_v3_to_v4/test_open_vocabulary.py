"""Ported from packages/session/session-format-v3-to-v4/tests/open-vocabulary.spec.ts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatHeader,
)
from deskagents.session.format_v3_to_v4 import restore_released_v4_artifact
from deskagents.util.values import stringify_json

SPEC = "packages/session/session-format-v3-to-v4/tests/open-vocabulary.spec.ts"
GROUP = "known V4 event interpretation"

HEADER: SessionFormatHeader = {
    "version": 4,
    "id": "open-vocabulary",
    "createdAt": 1,
    "delegationDepth": 0,
    "isSeeded": False,
}


def artifact(events: Sequence[dict[str, Any]]) -> SessionFormatArtifact:
    return SessionFormatArtifact(
        header=HEADER,
        inherited_event_count=0,
        events=[{**event, "seq": seq, "time": seq} for seq, event in enumerate(events)],
    )


def stringify_artifact(value: SessionFormatArtifact) -> str:
    """``JSON.stringify(artifact)``."""
    return stringify_json(
        {
            "header": value.header,
            "inheritedEventCount": value.inherited_event_count,
            "events": list(value.events),
        }
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps unknown ignorable payloads and coordinates opaque to every native"
    " semantic pass"
)
def test_keeps_unknown_ignorable_payloads_opaque() -> None:
    unknown = [
        {"type": type_, "data": None, "ignorable": True}
        for type_ in [
            "tool/result",
            "system/message",
            "request/header",
            "session/title-llm-request",
            "subagent/catalog",
            "session-log-deepseek/delivery-accepted",
            "compaction/start",
            "compaction/end",
            "session/end-seed",
        ]
    ]
    input_ = artifact(
        [
            *unknown,
            {"type": "turn/start", "data": {"turn": 1}},
            {"type": "turn/end", "data": {"turn": 1}},
        ]
    )
    before = stringify_artifact(input_)
    assert restore_released_v4_artifact(input_, frozenset(["turn/start", "turn/end"])) is input_
    assert stringify_artifact(input_) == before
    assert input_.events[-1]["seq"] == len(unknown) + 1


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > does not let an opaque end-seed clear an owned compaction"
)
def test_opaque_end_seed_does_not_clear_owned_compaction() -> None:
    input_ = artifact(
        [
            {"type": "compaction/start", "data": {"compactionId": "owner", "turn": None}},
            {"type": "session/end-seed", "data": {}, "ignorable": True},
            {"type": "turn/start", "data": {"turn": 1}},
        ]
    )
    with pytest.raises(SessionFormatError, match=r"crosses an open compaction"):
        restore_released_v4_artifact(input_, frozenset(["compaction/start", "turn/start"]))


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > does not reinterpret an unknown ignorable user row through a known title"
    " citation"
)
def test_does_not_reinterpret_unknown_ignorable_user_row() -> None:
    input_ = artifact(
        [
            {"type": "user/message", "data": None, "ignorable": True},
            {
                "type": "session/title",
                "data": {"title": "title", "source": {"kind": "generated"}, "messageSeqs": [0]},
            },
        ]
    )
    with pytest.raises(SessionFormatError, match=r"earlier human user/message"):
        restore_released_v4_artifact(input_, frozenset(["session/title"]))


@pytest.mark.upstream(f"{SPEC}::{GROUP} > still validates known events marked ignorable")
def test_still_validates_known_events_marked_ignorable() -> None:
    input_ = artifact([{"type": "tool/result", "data": None, "ignorable": True}])
    with pytest.raises(SessionFormatError, match=r"data must be an object"):
        restore_released_v4_artifact(input_, frozenset(["tool/result"]))


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > still validates unknown envelopes and refuses unknown required types"
)
def test_still_validates_unknown_envelopes() -> None:
    required = artifact([{"type": "external/event", "data": None}])
    with pytest.raises(SessionFormatError, match=r"unknown event type"):
        restore_released_v4_artifact(required, frozenset())
    input_ = artifact([{"type": "external/event", "data": None, "ignorable": True}])
    with pytest.raises(SessionFormatError, match=r"not dense"):
        restore_released_v4_artifact(
            replace(input_, events=[{**input_.events[0], "seq": 3}]), frozenset()
        )
