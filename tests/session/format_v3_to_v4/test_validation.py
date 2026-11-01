"""Ported from packages/session/session-format-v3-to-v4/tests/validation.spec.ts.

``assertReleasedV4Artifact`` (upstream ``src/testing/validation.ts``) is re-implemented
locally as :func:`assert_released_v4_physical_artifact`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEventCollector,
    SessionFormatHeader,
)
from deskagents.session.format_v3_to_v4 import (
    assert_released_v4_header,
    released_v4_session_format_codec,
    restore_released_v4_artifact,
)

SPEC = "packages/session/session-format-v3-to-v4/tests/validation.spec.ts"

KNOWN = frozenset(
    [
        "turn/start",
        "step/start",
        "assistant/message",
        "tool/call",
        "tool/result",
        "step/end",
        "turn/end",
        "user/message",
        "session/end-seed",
    ]
)


def header(**overrides: Any) -> SessionFormatHeader:
    return {
        "version": 4,
        "id": "v4-session",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
        **overrides,
    }


def artifact(
    events: Sequence[Any], h: SessionFormatHeader | None = None, inherited: int = 0
) -> SessionFormatArtifact:
    return SessionFormatArtifact(
        header=header() if h is None else h, inherited_event_count=inherited, events=events
    )


def assert_released_v4_physical_artifact(value: SessionFormatArtifact) -> None:
    """Validate an artifact through the native codec without vocabulary-aware restoration."""
    codec = released_v4_session_format_codec
    decoder = codec.create_decoder(
        codec.encode_header(value.header, value.inherited_event_count), "strict"
    )
    output = SessionFormatEventCollector()
    for event in value.events:
        decoder.decode_row(codec.encode_event(event), output)
    decoder.finish(output)


def message(id_: str, role: str, content: list[Any], source: dict[str, Any]) -> dict[str, Any]:
    return {"id": id_, "role": role, "content": content, "source": source}


def conversation() -> list[dict[str, Any]]:
    """One canonical released-v4 turn: call a tool and receive its result."""
    return [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 2, "data": {"turn": 1, "step": 1}},
        {
            "type": "assistant/message",
            "seq": 2,
            "time": 3,
            "surfaceOp": "append",
            "data": {
                "turn": 1,
                "step": 1,
                "message": message(
                    "assistant-1",
                    "assistant",
                    [
                        {"type": "text", "text": "calling"},
                        {"type": "tool-call", "id": "call-1", "name": "bash", "arguments": "{}"},
                    ],
                    {"kind": "model", "provider": "mock", "model": "mock"},
                ),
                "stream": [
                    {
                        "type": "chunk",
                        "time": 3,
                        "chunk": {
                            "type": "block-end",
                            "index": 1,
                            "block": {
                                "type": "tool-call",
                                "id": "call-1",
                                "name": "bash",
                                "arguments": "{}",
                            },
                        },
                    }
                ],
            },
        },
        {
            "type": "tool/call",
            "seq": 3,
            "time": 4,
            "data": {
                "turn": 1,
                "step": 1,
                "callId": "call-1",
                "name": "bash",
                "arguments": "{}",
            },
        },
        {
            "type": "tool/result",
            "seq": 4,
            "time": 5,
            "surfaceOp": "append",
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    **message(
                        "tool-1",
                        "tool",
                        [{"type": "text", "text": "ok"}],
                        {"kind": "tool", "callId": "call-1"},
                    ),
                    "toolCallId": "call-1",
                    "isError": False,
                },
            },
        },
        {"type": "step/end", "seq": 5, "time": 6, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 6,
            "time": 7,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]


HEADER_GROUP = "assertReleasedV4Header"


@pytest.mark.upstream(f"{SPEC}::{HEADER_GROUP} > accepts the minimal and complete header images")
def test_accepts_minimal_and_complete_headers() -> None:
    assert_released_v4_header(header())
    assert_released_v4_header(
        header(
            cwd="/work",
            parentSession="parent",
            origin="subagent",
            agentPreset="headless",
            isSeeded=True,
        )
    )


@pytest.mark.upstream(f"{SPEC}::{HEADER_GROUP} > refuses malformed headers")
def test_refuses_malformed_headers() -> None:
    malformed: list[tuple[str, dict[str, Any], str]] = [
        ("wrong version", {**header(), "version": 2}, r"expected format v4"),
        ("non-string id", {**header(), "id": 3}, r"id must be a string"),
        ("non-count createdAt", {**header(), "createdAt": "x"}, r"createdAt"),
        ("non-count delegationDepth", {**header(), "delegationDepth": -1}, r"delegationDepth"),
        ("non-boolean isSeeded", {**header(), "isSeeded": 1}, r"isSeeded must be boolean"),
        ("relative cwd", {**header(), "cwd": "work"}, r"cwd must be absolute"),
        (
            "non-string parentSession",
            {**header(), "parentSession": 3},
            r"parentSession must be a string",
        ),
        ("non-string agentPreset", {**header(), "agentPreset": 3}, r"agentPreset must be a string"),
        ("invalid origin", {**header(), "origin": "other"}, r'origin must be "subagent"'),
    ]
    for name, value, pattern in malformed:
        with pytest.raises(SessionFormatError) as caught:
            assert_released_v4_header(value)
        assert re.search(pattern, str(caught.value)), name


RESTORE_GROUP = "restoreReleasedV4Artifact"


@pytest.mark.upstream(
    f"{SPEC}::{RESTORE_GROUP} > validates dense envelopes, cuts, and event vocabulary"
)
def test_validates_dense_envelopes_cuts_and_vocabulary() -> None:
    valid = artifact(conversation())
    assert restore_released_v4_artifact(valid, KNOWN) is valid

    with pytest.raises(SessionFormatError, match=r"inherited event count exceeds its events"):
        restore_released_v4_artifact(artifact(conversation(), header(isSeeded=True), 20), KNOWN)
    with pytest.raises(
        SessionFormatError,
        match=r"format v4 unseeded Session contains an inherited end-seed marker",
    ):
        restore_released_v4_artifact(
            artifact(
                [{"type": "session/end-seed", "seq": 0, "time": 1, "data": {"inherited": True}}],
                header(),
                0,
            ),
            KNOWN,
        )
    with pytest.raises(
        SessionFormatError, match=r"unseeded format v4 Session has inherited events"
    ):
        restore_released_v4_artifact(artifact(conversation(), header(), 1), KNOWN)


@pytest.mark.upstream(
    f"{SPEC}::{RESTORE_GROUP} > retains unknown child membership and rejects invalid identities"
    " or duplicate child ids"
)
def test_retains_unknown_child_membership() -> None:
    known = KNOWN | {"subagent/catalog"}
    data: dict[str, Any] = {
        "version": 1,
        "childId": "unreadable",
        "childCreatedAt": 2,
        "mode": "unknown",
    }
    event = {"type": "subagent/catalog", "seq": 0, "time": 1, "data": data}
    valid = artifact([event])
    assert restore_released_v4_artifact(valid, known) is valid
    restore_released_v4_artifact(
        artifact([{**event, "data": {**data, "label": "retained"}}]), known
    )
    for invalid in [
        None,
        {**data, "version": 0},
        {**data, "version": 2},
        {**data, "childId": 1},
        {**data, "childCreatedAt": -1},
        {**data, "mode": "invalid"},
        {**data, "label": None},
    ]:
        with pytest.raises(SessionFormatError):
            restore_released_v4_artifact(artifact([{**event, "data": invalid}]), known)
    with pytest.raises(SessionFormatError, match="duplicate catalog child"):
        restore_released_v4_artifact(artifact([event, {**event, "seq": 1}]), known)
    with pytest.raises(SessionFormatError, match="duplicate catalog child"):
        restore_released_v4_artifact(
            artifact(
                [
                    event,
                    {
                        "type": "subagent/catalog",
                        "seq": 1,
                        "time": 1,
                        "data": {**data, "mode": "one-shot"},
                    },
                ]
            ),
            known,
        )


@pytest.mark.upstream(f"{SPEC}::{RESTORE_GROUP} > validates only the envelope in physical mode")
def test_validates_only_envelope_in_physical_mode() -> None:
    unknown = conversation()
    unknown.insert(2, {"type": "campaign/unknown", "seq": 0, "time": 1, "data": {"x": 1}})
    for index, item in enumerate(unknown):
        item["seq"] = index
    assert_released_v4_physical_artifact(artifact(unknown))
    with pytest.raises(SessionFormatError, match=r"unknown event type"):
        restore_released_v4_artifact(artifact(unknown), KNOWN)


@pytest.mark.upstream(
    f"{SPEC}::{RESTORE_GROUP} > refuses unknown non-ignorable types and admits ignorable ones"
)
def test_refuses_unknown_required_and_admits_ignorable() -> None:
    unknown = conversation()
    unknown.insert(2, {"type": "campaign/unknown", "seq": 99, "time": 1, "data": {"x": 1}})
    with pytest.raises(SessionFormatError, match=r"unknown event type"):
        restore_released_v4_artifact(artifact(unknown), KNOWN)
    ignorable = conversation()
    ignorable.insert(
        0, {"type": "campaign/unknown", "seq": 0, "time": 1, "ignorable": True, "data": {"x": 1}}
    )
    for index, item in enumerate(ignorable):
        item["seq"] = index
    restore_released_v4_artifact(artifact(ignorable), KNOWN | {"campaign/unknown"})


@pytest.mark.upstream(f"{SPEC}::{RESTORE_GROUP} > enforces V4-owned event density")
def test_enforces_event_density() -> None:
    with pytest.raises(SessionFormatError, match=r"is not dense"):
        restore_released_v4_artifact(
            artifact([{"type": "turn/start", "time": 1, "data": {}}]), KNOWN
        )
    with pytest.raises(SessionFormatError, match=r"is not dense"):
        restore_released_v4_artifact(
            artifact([conversation()[0], {**conversation()[1], "seq": 3}]), KNOWN
        )


@pytest.mark.upstream(f"{SPEC}::{RESTORE_GROUP} > enforces end-seed marker agreement")
def test_enforces_end_seed_marker_agreement() -> None:
    def marker(seq: int, inherited: bool) -> dict[str, Any]:
        return {
            "type": "session/end-seed",
            "seq": seq,
            "time": 1,
            "data": {"inherited": True} if inherited else {},
        }

    known = KNOWN | {"session/end-seed"}
    restore_released_v4_artifact(artifact([marker(0, True)], header(isSeeded=True), 0), known)
    with pytest.raises(SessionFormatError, match=r"seeded header disagrees"):
        restore_released_v4_artifact(
            artifact([marker(0, False), marker(1, True)], header(isSeeded=True), 0), known
        )
    with pytest.raises(
        SessionFormatError, match=r"unseeded Session contains an inherited end-seed marker"
    ):
        restore_released_v4_artifact(artifact([marker(0, True)], header(), 0), known)


@pytest.mark.upstream(
    f"{SPEC}::{RESTORE_GROUP} > refuses non-object records and non-string event types"
)
def test_refuses_non_object_records_and_non_string_types() -> None:
    with pytest.raises(SessionFormatError, match=r"unknown event type"):
        restore_released_v4_artifact(
            artifact([{"type": 3, "seq": 0, "time": 1, "data": {}}]), KNOWN
        )
    # Upstream reads ``null.type`` and throws a JavaScript TypeError.
    with pytest.raises(TypeError):
        restore_released_v4_artifact(artifact([None]), KNOWN)
