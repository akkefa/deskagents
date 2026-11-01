"""Ported from packages/core/session/tests/repair.spec.ts.

Python's ``open_turn_closers`` takes the cause kind (``"interrupted"`` or ``"forked"``)
rather than a ``{kind}`` record, so the local wrapper passes ``cause["kind"]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from deskagents.core.session import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    SessionEvent,
    ToolCallRecovery,
    session_seq,
)
from deskagents.core.session import interrupted_turn_closers as repair_interrupted_turn
from deskagents.core.session import open_turn_closers as close_open_turn
from deskagents.llm import ToolCallId, create_message, create_tool_result_message

SPEC = "packages/core/session/tests/repair.spec.ts"

Cause = dict[str, Any]
MOCK_SOURCE = {"kind": "model", "provider": "mock", "model": "mock"}


def interrupted_turn_closers(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    for event in events:
        session_seq(event["seq"])
    return repair_interrupted_turn(events)


def open_turn_closers(events: Sequence[SessionEvent], cause: Cause) -> list[SessionEvent]:
    for event in events:
        session_seq(event["seq"])
    return close_open_turn(events, cause["kind"])


def matches(actual: object, expected: object) -> bool:
    """``toMatchObject``: records match a subset of keys, arrays match element-wise."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and matches(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(matches(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return actual == expected


def user_turn_start(turn: int, seq: int) -> SessionEvent:
    return {"type": "turn/start", "seq": seq, "time": seq, "data": {"turn": turn}}


def assistant_calls(
    seq: int, turn: int, step: int, *call_ids: str, text: str | None = None, surface: bool = True
) -> SessionEvent:
    content: list[dict[str, Any]] = [] if text is None else [{"type": "text", "text": text}]
    content += [
        {"type": "tool-call", "id": ToolCallId(call_id), "name": "bash", "arguments": "{}"}
        for call_id in call_ids
    ]
    event: dict[str, Any] = {
        "type": "assistant/message",
        "seq": seq,
        "time": seq,
        "data": {
            "turn": turn,
            "step": step,
            "message": create_message(role="assistant", content=content, source=MOCK_SOURCE),
        },
    }
    if surface:
        event["surfaceOp"] = "append"
    return event


def tool_result(
    seq: int, turn: int, step: int, call_id: str, text: str | None = None
) -> SessionEvent:
    content = [] if text is None else [{"type": "text", "text": text}]
    return {
        "type": "tool/result",
        "seq": seq,
        "time": seq,
        "surfaceOp": "append",
        "data": {
            "turn": turn,
            "step": step,
            "message": create_tool_result_message(
                call_id=ToolCallId(call_id), content=content, is_error=False
            ),
        },
    }


def tool_call(seq: int, turn: int, step: int, call_id: str) -> SessionEvent:
    return {
        "type": "tool/call",
        "seq": seq,
        "time": seq,
        "data": {
            "turn": turn,
            "step": step,
            "callId": ToolCallId(call_id),
            "name": "bash",
            "arguments": "{}",
        },
    }


def step_start(seq: int, turn: int, step: int) -> SessionEvent:
    return {"type": "step/start", "seq": seq, "time": seq, "data": {"turn": turn, "step": step}}


def live_tool_requests() -> SessionEvent:
    return {
        "type": "assistant/message",
        "seq": session_seq(21),
        "time": 110,
        "surfaceOp": "append",
        "data": {
            "turn": 3,
            "step": 2,
            "stream": [],
            "message": create_message(
                role="assistant",
                source=MOCK_SOURCE,
                content=[
                    {
                        "type": "tool-call",
                        "id": ToolCallId("started"),
                        "name": "bash",
                        "arguments": "{}",
                    },
                    {
                        "type": "tool-call",
                        "id": ToolCallId("pending"),
                        "name": "bash",
                        "arguments": "{}",
                    },
                ],
            ),
        },
    }


def types(events: Sequence[SessionEvent]) -> list[str]:
    return [event["type"] for event in events]


@pytest.mark.upstream(
    f"{SPEC}::ToolCallRecovery > observes a live step without earlier history and retains "
    "repairs until their commits arrive"
)
def test_recovery_observes_live_step() -> None:
    recovery = ToolCallRecovery()
    assert recovery.results() == []
    recovery.observe(
        {"type": "step/start", "seq": session_seq(20), "time": 100, "data": {"turn": 3, "step": 2}}
    )
    recovery.observe(live_tool_requests())
    recovery.observe(
        {
            "type": "tool/call",
            "seq": session_seq(22),
            "time": 120,
            "data": {
                "turn": 3,
                "step": 2,
                "callId": ToolCallId("started"),
                "name": "bash",
                "arguments": "{}",
            },
        }
    )

    results = recovery.results()
    assert [
        {
            "seq": event["seq"],
            "time": event["time"],
            "turn": event["data"]["turn"],
            "step": event["data"]["step"],
            "callId": event["data"]["message"]["toolCallId"],
            "code": (event["data"].get("error") or {}).get("code"),
            "sourceEventSeqs": event.get("sourceEventSeqs"),
        }
        for event in results
    ] == [
        {
            "seq": 23,
            "time": 120,
            "turn": 3,
            "step": 2,
            "callId": "started",
            "code": TOOL_OUTCOME_UNKNOWN,
            "sourceEventSeqs": [22],
        },
        {
            "seq": 24,
            "time": 120,
            "turn": 3,
            "step": 2,
            "callId": "pending",
            "code": TOOL_NOT_STARTED,
            "sourceEventSeqs": None,
        },
    ]
    assert recovery.results() == results
    for index, event in enumerate(results):
        recovery.observe(event)
        assert recovery.results() == results[index + 1 :]


@pytest.mark.upstream(
    f"{SPEC}::ToolCallRecovery > does not acknowledge a pending call with $kind result reusing "
    "its id"
)
@pytest.mark.parametrize(
    ("kind", "turn", "step"),
    [("replacement", 3, 2), ("other turn", 2, 2), ("other step", 3, 1)],
)
def test_recovery_ignores_non_matching_results(kind: str, turn: int, step: int) -> None:
    recovery = ToolCallRecovery()
    recovery.observe(live_tool_requests())
    event: dict[str, Any] = {
        "type": "tool/result",
        "seq": session_seq(22),
        "time": 120,
        "surfaceOp": (
            {"op": "replace", "startSeq": session_seq(10), "endSeq": session_seq(10)}
            if kind == "replacement"
            else "append"
        ),
        "data": {
            "turn": turn,
            "step": step,
            "message": create_tool_result_message(
                call_id=ToolCallId("started"),
                content=[{"type": "text", "text": "old result"}],
                is_error=False,
            ),
        },
    }
    if kind == "replacement":
        event["sourceEventSeqs"] = [session_seq(10)]
    recovery.observe(event)

    assert [result["data"]["message"]["toolCallId"] for result in recovery.results()] == [
        ToolCallId("started"),
        ToolCallId("pending"),
    ]


CAUSES: list[Cause] = [{"kind": "interrupted"}, {"kind": "forked"}]
OPEN = "openTurnClosers (cause %o)"
FORKED_NOT_STARTED = (
    "The history inherited by this branch has no record of this tool call starting. The parent "
    "session may have executed it after the fork point. Decide whether to retry from the tool "
    "semantics: retry only if the operation is read-only or idempotent; if it may have side "
    "effects, first verify external state or ask the user. Do not retry blindly."
)
INTERRUPTED_NOT_STARTED = (
    "The tool call was interrupted before the Harness recorded it as started. Retry it if it is "
    "still needed."
)
causes = pytest.mark.parametrize("cause", CAUSES, ids=[cause["kind"] for cause in CAUSES])


@causes
@pytest.mark.upstream(f"{SPEC}::{OPEN} > returns nothing for a balanced log (ends on turn/end)")
def test_balanced_log(cause: Cause) -> None:
    balanced: list[SessionEvent] = [
        user_turn_start(1, 0),
        {
            "type": "turn/end",
            "seq": 1,
            "time": 1,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    assert open_turn_closers(balanced, cause) == []


@causes
@pytest.mark.upstream(f"{SPEC}::{OPEN} > returns nothing for an empty log")
def test_empty_log(cause: Cause) -> None:
    assert open_turn_closers([], cause) == []


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > closes an open turn with no open step (turn/end with the cause reason only)"
)
def test_open_turn_without_step(cause: Cause) -> None:
    closers = open_turn_closers([user_turn_start(1, 0)], cause)
    assert types(closers) == ["turn/end"]
    end = closers[0]
    assert end["seq"] == 1
    assert end["type"] == "turn/end"
    assert end["data"]["reason"] == {"kind": cause["kind"]}


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > closes an open step before the turn (step/end then turn/end)"
)
def test_open_step_before_turn(cause: Cause) -> None:
    events = [user_turn_start(1, 0), step_start(1, 1, 1)]
    closers = open_turn_closers(events, cause)
    assert types(closers) == ["step/end", "turn/end"]
    assert [event["seq"] for event in closers] == [2, 3]


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > marks an assistant tool request with no recorded call as not started"
)
def test_tool_request_not_started(cause: Cause) -> None:
    events = [
        user_turn_start(2, 0),
        step_start(1, 2, 1),
        assistant_calls(2, 2, 1, "call-1", text="calling a tool"),
    ]
    closers = open_turn_closers(events, cause)
    # tool/result (for the orphaned call) → step/end → turn/end, contiguous seqs.
    assert types(closers) == ["tool/result", "step/end", "turn/end"]
    assert [event["seq"] for event in closers] == [3, 4, 5]
    result = closers[0]
    assert result["type"] == "tool/result"
    assert matches(
        result["data"],
        {
            "turn": 2,
            "step": 1,
            "message": {
                "id": f"{cause['kind']}-tool-result-call-1-3",
                "source": {"callId": ToolCallId("call-1")},
                "role": "tool",
                "toolCallId": ToolCallId("call-1"),
                "isError": True,
            },
            "error": {"code": TOOL_NOT_STARTED},
        },
    )
    assert result["data"]["message"]["content"] == [
        {
            "type": "text",
            "text": FORKED_NOT_STARTED if cause["kind"] == "forked" else INTERRUPTED_NOT_STARTED,
        }
    ]


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > does NOT synthesize a result for a tool-call that already has one"
)
def test_answered_call_gets_no_result(cause: Cause) -> None:
    events = [
        user_turn_start(2, 0),
        step_start(1, 2, 1),
        assistant_calls(2, 2, 1, "call-1"),
        tool_result(3, 2, 1, "call-1", "ok"),
    ]
    # The call is answered, so only the open step + turn need closing.
    assert types(open_turn_closers(events, cause)) == ["step/end", "turn/end"]


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > preserves an unanswered assistant call after the owning step already closed"
)
def test_unanswered_call_after_closed_step(cause: Cause) -> None:
    events = [
        user_turn_start(2, 0),
        step_start(1, 2, 1),
        assistant_calls(2, 2, 1, "call-1"),
        {"type": "step/end", "seq": 3, "time": 3, "data": {"turn": 2, "step": 1}},
    ]

    assert open_turn_closers(events, cause) == [
        {"type": "turn/end", "seq": 4, "time": 3, "data": {"turn": 2, "reason": cause}},
    ]


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > synthesizes results only for the still-open turn, "
    "not a committed earlier turn"
)
def test_results_only_for_open_turn(cause: Cause) -> None:
    # Turn 1 completed with its own tool call+result (balanced). Turn 2 was cut
    # open with an unanswered call. Only turn 2's call must get a synthetic result.
    events = [
        user_turn_start(1, 0),
        step_start(1, 1, 1),
        assistant_calls(2, 1, 1, "old-call"),
        tool_result(3, 1, 1, "old-call"),
        {"type": "step/end", "seq": 4, "time": 4, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 5,
            "time": 5,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
        user_turn_start(2, 6),
        step_start(7, 2, 1),
        assistant_calls(8, 2, 1, "new-call"),
    ]
    closers = open_turn_closers(events, cause)
    assert types(closers) == ["tool/result", "step/end", "turn/end"]
    assert closers[0]["data"]["message"]["source"]["callId"] == "new-call"


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > synthesizes a result for each of multiple unanswered calls, in log order"
)
def test_result_for_each_unanswered_call(cause: Cause) -> None:
    events = [
        user_turn_start(1, 0),
        step_start(1, 1, 1),
        assistant_calls(2, 1, 1, "call-a", "call-b"),
        # call-a got answered before the cut; call-b did not.
        tool_result(3, 1, 1, "call-a"),
    ]
    closers = open_turn_closers(events, cause)
    assert types(closers) == ["tool/result", "step/end", "turn/end"]
    assert closers[0]["data"]["message"]["source"]["callId"] == "call-b"


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > synthesized tool/result carries surfaceOp and sourceEventSeqs when "
    "tool/call was logged"
)
def test_result_carries_surface_metadata(cause: Cause) -> None:
    events = [
        user_turn_start(1, 0),
        step_start(1, 1, 1),
        assistant_calls(2, 1, 1, "call-1"),
        tool_call(3, 1, 1, "call-1"),
    ]
    closers = open_turn_closers(events, cause)
    assert types(closers) == ["tool/result", "step/end", "turn/end"]
    result = closers[0]
    assert result["surfaceOp"] == "append"
    assert result["sourceEventSeqs"] == [3]
    assert result["data"]["error"] == {
        "name": "ToolOutcomeUnknownError",
        "code": TOOL_OUTCOME_UNKNOWN,
    }
    content = result["data"]["message"]["content"][0]
    if content["type"] != "text":
        raise AssertionError("expected a text tool result")
    assert "retry only if the operation is read-only or idempotent" in content["text"]
    assert "first verify external state or ask the user" in content["text"]


@causes
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > handles tool/call without a matching assistant/message entry gracefully"
)
def test_orphan_tool_call(cause: Cause) -> None:
    # A raw tool/call with no assistant-registered pending call has nothing to
    # answer; repair still closes the step and turn without synthesizing a result.
    events = [user_turn_start(1, 0), step_start(1, 1, 1), tool_call(2, 1, 1, "orphan")]
    # No pending calls → no synthetic tool/result, just step/end + turn/end.
    assert types(open_turn_closers(events, cause)) == ["step/end", "turn/end"]


WORDING = "openTurnClosers model-visible wording"


def open_call(started: bool) -> list[SessionEvent]:
    """An open turn with one unanswered call; ``started`` also logs its tool/call."""
    events = [
        user_turn_start(1, 0),
        step_start(1, 1, 1),
        assistant_calls(2, 1, 1, "call-1", surface=False),
    ]
    if started:
        events.append(tool_call(3, 1, 1, "call-1"))
    return events


def result_text(closers: Sequence[SessionEvent]) -> str:
    result = closers[0] if closers else None
    if result is None or result["type"] != "tool/result":
        raise AssertionError("expected a text tool result")
    content = result["data"]["message"]["content"][0]
    if content["type"] != "text":
        raise AssertionError("expected a text tool result")
    text: str = content["text"]
    return text


# The exact wording is model-visible: pinned verbatim per cause and lifecycle.
@pytest.mark.upstream(f"{SPEC}::{WORDING} > cause %o, started=%s pins its exact text")
@pytest.mark.parametrize(
    ("cause", "started", "text"),
    [
        ({"kind": "interrupted"}, False, INTERRUPTED_NOT_STARTED),
        (
            {"kind": "interrupted"},
            True,
            "The tool call was interrupted after it was recorded, but no result was durably "
            "recorded. Its outcome is unknown. Decide whether to retry from the tool semantics: "
            "retry only if the operation is read-only or idempotent; if it may have side effects, "
            "first verify external state or ask the user. Do not retry blindly.",
        ),
        ({"kind": "forked"}, False, FORKED_NOT_STARTED),
        (
            {"kind": "forked"},
            True,
            "The history inherited by this branch records this tool call starting but does not "
            "include its result. The parent session may have completed it after the fork point. "
            "Decide whether to retry from the tool semantics: retry only if the operation is "
            "read-only or idempotent; if it may have side effects, first verify external state "
            "or ask the user. Do not retry blindly.",
        ),
    ],
)
def test_pins_exact_text(cause: Cause, started: bool, text: str) -> None:
    assert result_text(open_turn_closers(open_call(started), cause)) == text


@pytest.mark.upstream(
    f"{SPEC}::{WORDING} > shares the error codes across causes: they state the same call "
    "lifecycle fact"
)
def test_shares_error_codes_across_causes() -> None:
    for cause in CAUSES:
        not_started = open_turn_closers(open_call(False), cause)[0]
        assert not_started["type"] == "tool/result"
        assert not_started["data"]["error"] == {
            "name": "ToolNotStartedError",
            "code": TOOL_NOT_STARTED,
        }
        started = open_turn_closers(open_call(True), cause)[0]
        assert started["type"] == "tool/result"
        assert started["data"]["error"] == {
            "name": "ToolOutcomeUnknownError",
            "code": TOOL_OUTCOME_UNKNOWN,
        }


@pytest.mark.upstream(
    f"{SPEC}::{WORDING} > interruptedTurnClosers is the public interrupted-cause entry (cause "
    "selection stays internal)"
)
def test_interrupted_turn_closers_is_public_entry() -> None:
    events = open_call(True)
    assert interrupted_turn_closers(events) == open_turn_closers(events, {"kind": "interrupted"})
    end = interrupted_turn_closers(events)[-1]
    assert end["type"] == "turn/end"
    assert end["data"]["reason"] == {"kind": "interrupted"}
