"""Ported from packages/session/session-format-v3-to-v4/tests/fork.spec.ts.

Not ported: "round trips a pruned fork result while preserving its original message
identity" needs the cordis ``Context`` with the session projection registry, token
meter and tool-result pruner plugins, which Python lacks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from deskagents.core.session import Session, SessionSeq, build_fork_seed
from deskagents.llm import ToolCallId, create_message
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
from deskagents.session.format_v3_to_v4.fork_result import assert_v4_fork_result
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-format-v3-to-v4/tests/fork.spec.ts"
GROUP = "V4 fork results"


def fork() -> Session:
    source = Session.create("fork-parent")
    source.append("turn/start", {"turn": 1})
    source.append("step/start", {"turn": 1, "step": 1})
    source.append(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "stream": [],
            "message": create_message(
                role="assistant",
                source={"kind": "model", "provider": "mock", "model": "mock"},
                content=[
                    {
                        "type": "tool-call",
                        "id": ToolCallId("read-one"),
                        "name": "read",
                        "arguments": "{}",
                    }
                ],
            ),
        },
        surface_op="append",
    )
    seed = build_fork_seed(source.snapshot_events(), SessionSeq(2))
    return Session.create(
        "fork-child",
        seed,
        {
            "version": 4,
            "id": "fork-child",
            "createdAt": 1,
            "delegationDepth": 0,
            "isSeeded": True,
            "parentSession": source.id,
        },
        3,
    )


def artifact(session: Session | None = None) -> SessionFormatArtifact:
    session = fork() if session is None else session
    value: Any = parse_json(
        stringify_json(
            {
                "header": session.header,
                "events": session.snapshot_events(),
                "inheritedEventCount": session.inherited_event_count,
            }
        )
    )
    return SessionFormatArtifact(
        header=value["header"],
        events=value["events"],
        inherited_event_count=value["inheritedEventCount"],
    )


def restore(events: Sequence[Any]) -> SessionFormatArtifact:
    base = artifact()
    return restore_released_v4_artifact(
        SessionFormatArtifact(
            header=base.header, inherited_event_count=base.inherited_event_count, events=events
        ),
        {event["type"] for event in events},
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > ignores non-object fork-result data and validates non-object sources"
)
def test_ignores_non_object_fork_result_data_and_validates_non_object_sources() -> None:
    assert_v4_fork_result({"type": "tool/result", "data": None})
    with pytest.raises(SessionFormatError, match="invalid V4 not-started fork result"):
        assert_v4_fork_result(
            {
                "type": "tool/result",
                "seq": 0,
                "surfaceOp": "append",
                "data": {
                    "error": {"code": "TOOL_NOT_STARTED", "name": "ToolNotStartedError"},
                    "message": {
                        "id": "forked-tool-result-undefined-0",
                        "role": "tool",
                        "toolCallId": "call-1",
                        "isError": True,
                        "source": None,
                        "content": [{"type": "text", "text": "not started"}],
                    },
                },
            }
        )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > persists original fork IDs and wording, restores relationships and "
    "retains nested inherited cuts"
)
def test_persists_original_fork_ids_and_wording() -> None:
    child = fork()
    original: Any = artifact(child).events
    before = stringify_json(original)
    physical = codec.encode_header(artifact(child).header, child.inherited_event_count)
    decoder = codec.create_decoder(physical, "strict")
    output = SessionFormatEventCollector()
    for event in original:
        row = codec.encode_event(event)
        assert_v4_row_admission(row)
        decoder.decode_row(row, output)
    assert decoder.finish(output) == 3
    assert output.values == list(original)
    assert restore(output.values).events is output.values
    assert stringify_json(original) == before
    result = next(event for event in child.snapshot_events() if event["type"] == "tool/result")
    assert result["data"]["message"]["id"] == f"forked-tool-result-read-one-{result['seq']}"
    assert "The parent session may have executed it after the fork point." in stringify_json(result)
    nested = Session.create(
        "fork-grandchild",
        build_fork_seed(child.snapshot_events(), result["seq"]),
        {
            "version": 4,
            "id": "fork-grandchild",
            "createdAt": 1,
            "delegationDepth": 0,
            "isSeeded": True,
            "parentSession": child.id,
        },
        result["seq"] + 1,
    )
    assert (
        restore_released_v4_artifact(artifact(nested), {str(event["type"]) for event in original})
        is not None
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects incomplete persisted result fields through V4 restoration"
)
def test_rejects_incomplete_persisted_result_fields_through_v4_restoration() -> None:
    original: Any = artifact().events
    result = next(event for event in original if event["type"] == "tool/result")
    data = result["data"]
    message = data["message"]
    variants = [None, {**data, "message": {**message, "source": None}}]
    for invalid in variants:
        events = [{**event, "data": invalid} if event is result else event for event in original]
        with pytest.raises(SessionFormatError):
            restore(events)


def _id_sequence(row: Any) -> None:
    row["data"]["message"]["id"] = "forked-tool-result-read-one-0"


def _error_name(row: Any) -> None:
    row["data"]["error"]["name"] = "OtherError"


def _call_id(row: Any) -> None:
    row["data"]["message"]["toolCallId"] = "different"


def _error_result(row: Any) -> None:
    row["data"]["message"]["isError"] = False


def _source_references(row: Any) -> None:
    row["sourceEventSeqs"] = [2]


def _content(row: Any) -> None:
    row["data"]["message"]["content"] = []


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects malformed fork %s")
@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("ID sequence", _id_sequence),
        ("error name", _error_name),
        ("call ID", _call_id),
        ("error result", _error_result),
        ("source references", _source_references),
        ("content", _content),
    ],
)
def test_rejects_malformed_fork(name: str, mutate: Callable[[Any], None]) -> None:
    events: Any = parse_json(stringify_json(artifact().events))
    result = next(event for event in events if event["type"] == "tool/result")
    mutate(result)
    with pytest.raises(SessionFormatError):
        restore(events)
    with pytest.raises(SessionFormatError):
        assert_v4_row_admission(result)
