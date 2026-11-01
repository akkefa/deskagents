"""Ported from packages/core/session/tests/fork.spec.ts.

``ctx.sessions`` and the plugin-returned store are the same service object, read here
with ``ctx.get("sessions")``. Upstream's ``SessionEventMap`` augmentation for the
``test/*`` event types has no Python counterpart: unknown types are appended as-is.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    CreateSessionMeta,
    CreateSessionOptions,
    Session,
    SessionEvent,
    SessionForkError,
    SessionSeq,
    SessionStore,
    session_id,
    session_log_offset,
    session_seq,
)
from deskagents.llm import ToolCallId, create_message, create_user_message
from deskagents.util.values import MAX_SAFE_INTEGER

SPEC = "packages/core/session/tests/fork.spec.ts"
MOCK_SOURCE = {"kind": "model", "provider": "mock", "model": "mock"}
COMPLETED = {"kind": "completed"}


@pytest_asyncio.fixture
async def ctx() -> AsyncIterator[Context]:
    context = Context()
    await context.plugin(SessionStore)
    yield context
    await context.fiber.dispose()


@pytest.fixture
def sessions(ctx: Context) -> SessionStore:
    return cast(SessionStore, ctx.get("sessions"))


def cwd(path: str) -> CreateSessionOptions:
    return CreateSessionOptions(meta=CreateSessionMeta(cwd=path))


class AnyStr:
    """``expect.any(String)``."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str)

    def __hash__(self) -> int:
        return 0


def matches(actual: object, expected: object) -> bool:
    """``toMatchObject``: records match a subset of keys, arrays match element-wise,
    and a compiled pattern stands in for ``expect.stringMatching``."""
    if isinstance(expected, re.Pattern):
        return isinstance(actual, str) and expected.search(actual) is not None
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


def append_closed_turn(
    session: Session,
    turn: int,
    text: str | None = None,
    reason: Mapping[str, Any] = COMPLETED,
) -> None:
    session.append("turn/start", {"turn": turn})
    session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": f"hello {turn}" if text is None else text}],
            source={"kind": "user"},
        ),
        surface_op="append",
    )
    session.append("turn/end", {"turn": turn, "reason": reason})


def append_open_turn(session: Session, turn: int) -> None:
    session.append("turn/start", {"turn": turn})
    session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": f"open {turn}"}], source={"kind": "user"}
        ),
        surface_op="append",
    )


def first_user_message(events: Sequence[SessionEvent]) -> SessionEvent:
    for event in events:
        if event["type"] == "user/message":
            return event
    raise AssertionError("missing user/message")


def last_seq(session: Session) -> SessionSeq:
    events = session.snapshot_events()
    if not events:
        raise AssertionError("missing last event")
    return SessionSeq(events[-1]["seq"])


def inherited(session: Session) -> list[SessionEvent]:
    """A seeded child's fork-inherited prefix."""
    return list(session.snapshot_events(session_log_offset(0), session.inherited_event_count))


def raises_fork_error(message: str) -> pytest.RaisesExc[SessionForkError]:
    """``toThrow(new SessionForkError(message, code))``; callers check ``code`` after."""
    return pytest.raises(SessionForkError, match=f"^{re.escape(message)}$")


FORK = "SessionStore.fork"


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > forks an empty live session as an independent empty seeded session"
)
async def test_forks_empty_live_session(sessions: SessionStore) -> None:
    source = sessions.create(session_id("empty-parent"), cwd("/workspace"))

    child = sessions.fork(source, None, session_id("empty-child"))

    assert inherited(child) == []
    assert "parentSession" not in child.header
    assert matches(
        child.header, {"id": session_id("empty-child"), "cwd": "/workspace", "isSeeded": True}
    )
    assert child.inherited_event_count == 0


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > forks the latest completed boundary by default into detached frozen "
    "seed events"
)
async def test_forks_latest_completed_boundary(sessions: SessionStore) -> None:
    source = sessions.create(session_id("parent"), cwd("/workspace"))
    append_closed_turn(source, 1, "hello")

    child = sessions.fork(session_id("parent"), None, session_id("child"))

    assert inherited(child) == list(source.snapshot_events())
    assert child.snapshot_events() is not source.snapshot_events()
    assert child.snapshot_events()[1] is not source.snapshot_events()[1]
    with pytest.raises(TypeError):
        first_user_message(child.snapshot_events())["data"]["content"][0] = {
            "type": "text",
            "text": "child mutation",
        }
    assert first_user_message(source.snapshot_events())["data"]["content"] == [
        {"type": "text", "text": "hello"}
    ]
    assert first_user_message(child.snapshot_events())["data"]["content"] == [
        {"type": "text", "text": "hello"}
    ]
    assert "parentSession" not in child.header
    assert matches(child.header, {"id": session_id("child"), "cwd": "/workspace", "isSeeded": True})
    assert child.inherited_event_count == source.seq


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > includes stable log-only events appended after a closed turn"
)
async def test_includes_log_only_events(sessions: SessionStore) -> None:
    source = sessions.create(session_id("log-only-parent"))
    append_closed_turn(source, 1, "hello")
    source.append("test/log-only", {"value": "after execution"})

    child = sessions.fork(source, None, session_id("log-only-child"))

    assert inherited(child) == list(source.snapshot_events())
    assert matches(
        inherited(child)[-1], {"type": "test/log-only", "data": {"value": "after execution"}}
    )


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > forks from an earlier turn boundary even when the source currently has "
    "an open tail"
)
async def test_forks_from_earlier_boundary(sessions: SessionStore) -> None:
    source = sessions.create(session_id("parent"), cwd("/workspace"))
    append_closed_turn(source, 1, "first")
    first_boundary = last_seq(source)
    append_closed_turn(source, 2, "second")
    append_open_turn(source, 3)

    child = sessions.fork(source, first_boundary, session_id("child-from-first"))

    assert inherited(child) == list(source.snapshot_events()[: first_boundary + 1])
    assert child.inherited_event_count == first_boundary + 1
    assert child.derive_messages() == [
        {
            "id": AnyStr(),
            "role": "user",
            "content": [{"type": "text", "text": "first"}],
            "source": {"kind": "user"},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > accepts every turn/end reason as an explicit fork boundary"
)
async def test_accepts_every_turn_end_reason(sessions: SessionStore) -> None:
    reasons: list[dict[str, Any]] = [
        {"kind": "completed"},
        {"kind": "aborted", "reason": {"kind": "user"}},
        {"kind": "error", "error": {"message": "model failed", "code": "UNKNOWN"}},
        {"kind": "aborted", "reason": {"kind": "disposed"}},
        {"kind": "max-tokens"},
        {"kind": "interrupted"},
    ]

    for index, reason in enumerate(reasons):
        source = sessions.create(session_id(f"parent-{index}"))
        append_closed_turn(source, 1, reason["kind"], reason)

        child = sessions.fork(source, last_seq(source), session_id(f"child-{index}"))

        assert inherited(child)[-1]["type"] == "turn/end"
        assert child.inherited_event_count == source.seq


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > marks a bracket the child inherited from a still-running parent"
)
async def test_marks_inherited_bracket(sessions: SessionStore) -> None:
    # The constructor placement's central claim, unreachable from the
    # persistence load path.
    parent = sessions.create(session_id("bracket-parent"), cwd("/workspace"))
    append_closed_turn(parent, 1, "work")
    opened = parent.append("test/bracket-open", {"id": "op-1"})

    child = sessions.fork(parent, None, session_id("bracket-child"))

    # Parent: no end-seed event follows the bracket, so its owner treats it as live.
    assert parent.snapshot_events()[-1] is opened
    assert not any(event["type"] == "session/end-seed" for event in parent.snapshot_events())
    # Child: the same bracket is before end-seed, so it belongs to the seed.
    boundary = child.snapshot_events()[-1]
    assert matches(boundary, {"type": "session/end-seed"})
    assert boundary["seq"] > opened["seq"]
    assert child.first_live_seq == opened["seq"] + 2
    assert matches(inherited(child)[-1], {"type": "test/bracket-open", "data": {"id": "op-1"}})


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{FORK} > rejects invalid boundaries before creating a child")
async def test_rejects_invalid_boundaries(sessions: SessionStore) -> None:
    empty = sessions.create(session_id("empty"))
    with raises_fork_error(
        'fork boundary 0 does not exist in session "empty" (last seq: none)'
    ) as caught:
        sessions.fork(empty, session_seq(0), session_id("empty-child"))
    assert caught.value.code == "INVALID_BOUNDARY"
    assert sessions.get(session_id("empty-child")) is None

    source = sessions.create(session_id("parent"))
    append_closed_turn(source, 1)
    with pytest.raises(SessionForkError, match=r"non-negative safe integer"):
        sessions.fork(source, -1, session_id("negative"))
    with pytest.raises(SessionForkError, match=r"non-negative safe integer"):
        sessions.fork(source, cast(Any, 0.5), session_id("fraction"))
    with pytest.raises(SessionForkError, match=r"non-negative safe integer"):
        sessions.fork(source, MAX_SAFE_INTEGER + 1, session_id("unsafe"))
    with raises_fork_error(
        f'fork boundary {source.seq} does not exist in session "parent" '
        f"(last seq: {source.seq - 1})"
    ) as caught:
        sessions.fork(source, session_seq(source.seq), session_id("past-end"))
    assert caught.value.code == "INVALID_BOUNDARY"


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > rejects a corrupted live source whose array index no longer matches "
    "event seq"
)
async def test_rejects_corrupted_live_source(sessions: SessionStore) -> None:
    source = sessions.create(session_id("corrupt-parent"))
    append_closed_turn(source, 1)
    mutable_log: list[SessionEvent] = source._log
    mutable_log[2] = {**mutable_log[2], "seq": session_seq(99)}

    with raises_fork_error(
        'fork boundary 2 does not match a contiguous event seq in session "corrupt-parent"'
    ) as caught:
        sessions.fork(source, session_seq(2), session_id("corrupt-child"))
    assert caught.value.code == "INVALID_BOUNDARY"
    assert sessions.get(session_id("corrupt-child")) is None


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{FORK} > rejects an unknown live session id")
async def test_rejects_unknown_session_id(sessions: SessionStore) -> None:
    with raises_fork_error('session "missing" not found') as caught:
        sessions.fork(session_id("missing"))
    assert caught.value.code == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > rejects a detached Session object that is not live in ctx.sessions"
)
async def test_rejects_detached_session(sessions: SessionStore) -> None:
    detached = Session.create(session_id("detached"))

    with raises_fork_error('session "detached" not found') as caught:
        sessions.fork(detached)
    assert caught.value.code == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > rejects a stale Session object whose id is live on a different instance"
)
async def test_rejects_stale_session(sessions: SessionStore) -> None:
    sessions.create(session_id("same-id"))
    stale = Session.create(session_id("same-id"))

    with raises_fork_error('session "same-id" is not the live store instance') as caught:
        sessions.fork(stale)
    assert caught.value.code == "SESSION_NOT_LIVE"


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > rejects a child session id that is already live with a typed fork error"
)
async def test_rejects_live_child_id(sessions: SessionStore) -> None:
    source = sessions.create(session_id("parent"))
    append_closed_turn(source, 1)
    sessions.create(session_id("child"))

    with raises_fork_error('session "child" already exists') as caught:
        sessions.fork(source, None, session_id("child"))
    assert caught.value.code == "SESSION_ALREADY_EXISTS"


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FORK} > rejects a duplicate child session id before validating the boundary"
)
async def test_rejects_duplicate_child_before_boundary(sessions: SessionStore) -> None:
    source = sessions.create(session_id("open-parent"))
    source.append("turn/start", {"turn": 1})
    sessions.create(session_id("child"))

    with raises_fork_error('session "child" already exists') as caught:
        sessions.fork(source, None, session_id("child"))
    assert caught.value.code == "SESSION_ALREADY_EXISTS"


OPEN = "fork boundaries inside an open turn"


def open_turn_source(sessions: SessionStore, sid: str) -> Session:
    """Open turn 2 after a closed turn 1 and return the source."""
    source = sessions.create(session_id(sid))
    append_closed_turn(source, 1, "first")
    source.append("turn/start", {"turn": 2})
    return source


def append_tool_request(session: Session, call_id: ToolCallId) -> None:
    """Append the open step + assistant tool request of turn 2."""
    session.append("step/start", {"turn": 2, "step": 1})
    session.append(
        "assistant/message",
        {
            "stream": [],
            "turn": 2,
            "step": 1,
            "message": create_message(
                role="assistant",
                content=[{"type": "tool-call", "id": call_id, "name": "bash", "arguments": "{}"}],
                source=MOCK_SOURCE,
            ),
        },
        surface_op="append",
    )


def child_tail(child: Session, source: Session) -> list[SessionEvent]:
    return list(child.snapshot_events()[len(source.snapshot_events()) + 1 :])


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > closes an empty open turn with a synthetic turn/end {{forked}} outside "
    "seedLength"
)
async def test_closes_empty_open_turn(sessions: SessionStore) -> None:
    source = open_turn_source(sessions, "open-empty")

    child = sessions.fork(source, last_seq(source), session_id("open-empty-child"))

    seed = child.snapshot_events()
    assert seed[: len(source.snapshot_events())] == source.snapshot_events()
    assert matches(
        seed[-1], {"type": "turn/end", "data": {"turn": 2, "reason": {"kind": "forked"}}}
    )
    # The closer is child work: `seedLength` names only events that exist in
    # the parent's log, so telemetry receivers stitching a fork's prefix from
    # the parent's stream on `(parent_id, seed_length)` read exactly the
    # copied prefix.
    assert child.inherited_event_count == len(source.snapshot_events())


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > closes only the turn when the cut lands on a closed step boundary"
)
async def test_closes_only_turn_on_closed_step(sessions: SessionStore) -> None:
    source = open_turn_source(sessions, "open-step-closed")
    source.append("step/start", {"turn": 2, "step": 1})
    source.append(
        "assistant/message",
        {
            "stream": [],
            "turn": 2,
            "step": 1,
            "message": create_message(
                role="assistant",
                content=[{"type": "text", "text": "step done"}],
                source=MOCK_SOURCE,
            ),
        },
        surface_op="append",
    )
    source.append("step/end", {"turn": 2, "step": 1})

    child = sessions.fork(source, last_seq(source), session_id("open-step-closed-child"))

    assert [event["type"] for event in child_tail(child, source)] == ["turn/end"]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > answers an assistant call cut before its tool/call with a forked "
    "not-started error result"
)
async def test_answers_not_started_call(sessions: SessionStore) -> None:
    source = open_turn_source(sessions, "open-not-started")
    append_tool_request(source, ToolCallId("call-ns"))

    child = sessions.fork(source, last_seq(source), session_id("open-not-started-child"))

    closers = child_tail(child, source)
    assert [event["type"] for event in closers] == ["tool/result", "step/end", "turn/end"]
    result = closers[0]
    assert result["type"] == "tool/result"
    assert matches(
        result["data"],
        {
            "message": {
                "id": re.compile(r"^forked-tool-result-call-ns-"),
                "source": {"callId": ToolCallId("call-ns")},
            },
            "error": {"name": "ToolNotStartedError", "code": TOOL_NOT_STARTED},
        },
    )
    # The synthetic result is part of the child's model-visible history, not a
    # request-time patch: the derived messages end with the error tool result.
    derived = child.derive_messages()
    assert matches(
        derived[-1],
        {
            "role": "tool",
            "source": {"kind": "tool", "callId": ToolCallId("call-ns")},
            "toolCallId": ToolCallId("call-ns"),
            "isError": True,
            "content": [{"type": "text"}],
        },
    )


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > answers a dispatched call cut before its result with an unknown-outcome "
    "error citing the tool/call"
)
async def test_answers_dispatched_call_with_unknown_outcome(sessions: SessionStore) -> None:
    source = open_turn_source(sessions, "open-unknown")
    append_tool_request(source, ToolCallId("call-uo"))
    call = source.append(
        "tool/call",
        {
            "turn": 2,
            "step": 1,
            "callId": ToolCallId("call-uo"),
            "name": "bash",
            "arguments": "{}",
        },
    )

    child = sessions.fork(source, last_seq(source), session_id("open-unknown-child"))

    closers = child_tail(child, source)
    assert [event["type"] for event in closers] == ["tool/result", "step/end", "turn/end"]
    result = closers[0]
    assert result["type"] == "tool/result"
    assert result["data"]["error"] == {
        "name": "ToolOutcomeUnknownError",
        "code": TOOL_OUTCOME_UNKNOWN,
    }
    assert result["sourceEventSeqs"] == [call["seq"]]


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{OPEN} > omits a failed assistant attempt from derived history")
async def test_omits_failed_attempt(sessions: SessionStore) -> None:
    source = open_turn_source(sessions, "open-mid-stream")
    source.append("step/start", {"turn": 2, "step": 1})
    source.append("assistant/attempt", {"turn": 2, "step": 1, "stream": []})

    child = sessions.fork(source, last_seq(source), session_id("open-mid-stream-child"))

    assert [event["type"] for event in child_tail(child, source)] == ["step/end", "turn/end"]
    assert not any(message["role"] == "assistant" for message in child.derive_messages())


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > preserves a closed step with an unanswered call (turn ended: %s)"
)
@pytest.mark.parametrize("turn_ended", [False, True])
async def test_preserves_closed_step_with_unanswered_call(
    sessions: SessionStore, turn_ended: bool
) -> None:
    source = open_turn_source(sessions, "dangling-closed-step")
    append_tool_request(source, ToolCallId("call-dangling"))
    source.append(
        "tool/call",
        {
            "turn": 2,
            "step": 1,
            "callId": ToolCallId("call-dangling"),
            "name": "bash",
            "arguments": "{}",
        },
    )
    source.append("step/end", {"turn": 2, "step": 1})
    if turn_ended:
        source.append(
            "turn/end",
            {
                "turn": 2,
                "reason": {
                    "kind": "error",
                    "error": {"message": "scheduler failed", "code": "UNKNOWN"},
                },
            },
        )
    copied_prefix = source.snapshot_events()

    child = sessions.fork(source)

    assert inherited(child)[: len(copied_prefix)] == list(copied_prefix)
    expected_tail: list[dict[str, Any]] = (
        []
        if turn_ended
        else [
            {
                "type": "turn/end",
                "seq": session_seq(len(copied_prefix) + 1),
                "time": copied_prefix[-1]["time"],
                "data": {"turn": 2, "reason": {"kind": "forked"}},
            }
        ]
    )
    assert list(child.snapshot_events()[len(copied_prefix) + 1 :]) == expected_tail
    assert child.inherited_event_count == len(copied_prefix)
    assert child.derive_messages() == source.derive_messages()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{OPEN} > leaves a plugin bracket left open at the cut untouched (log-only locks "
    "belong to their owner)"
)
async def test_leaves_open_plugin_bracket(sessions: SessionStore) -> None:
    source = open_turn_source(sessions, "open-bracket")
    source.append("test/bracket-open", {"id": "op-live"})

    child = sessions.fork(source, last_seq(source), session_id("open-bracket-child"))

    assert [event["type"] for event in child_tail(child, source)] == ["turn/end"]
    # The unmatched bracket sits before the child's end-seed marker, so its
    # owning plugin decides its staleness from that boundary.
    assert matches(
        child.snapshot_events()[child.inherited_event_count],
        {"type": "session/end-seed", "data": {"inherited": True}},
    )


REPLACE = "fork boundaries around a surface replacement"


def replacement_source(sessions: SessionStore, sid: str) -> tuple[Session, SessionSeq]:
    """A source whose open turn 2 replaces turn 1's user message with a summary
    (the compaction surface pattern: a ``user/message`` with a range replace op).

    Returns the source and the replacement event's seq.
    """
    source = sessions.create(session_id(sid))
    append_closed_turn(source, 1, "original request")
    replaced_seq = SessionSeq(first_user_message(source.snapshot_events())["seq"])
    source.append("turn/start", {"turn": 2})
    replacement = source.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "summary of earlier work"}],
            source={"kind": "compact-checkpoint", "compactionId": "open-bracket"},
        ),
        surface_op={"op": "replace", "startSeq": replaced_seq, "endSeq": replaced_seq},
        source_event_seqs=[replaced_seq],
    )
    return source, SessionSeq(replacement["seq"])


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{REPLACE} > derives the original message for a cut before the replacement"
)
async def test_derives_original_before_replacement(sessions: SessionStore) -> None:
    source, replacement_seq = replacement_source(sessions, "replace-before")

    child = sessions.fork(
        source, session_seq(replacement_seq - 1), session_id("replace-before-child")
    )

    assert [message["content"] for message in child.derive_messages()] == [
        [{"type": "text", "text": "original request"}],
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{REPLACE} > derives the compacted surface for a cut after the replacement"
)
async def test_derives_compacted_after_replacement(sessions: SessionStore) -> None:
    source, replacement_seq = replacement_source(sessions, "replace-after")

    child = sessions.fork(source, replacement_seq, session_id("replace-after-child"))

    assert [message["content"] for message in child.derive_messages()] == [
        [{"type": "text", "text": "summary of earlier work"}],
    ]
    assert matches(
        child.snapshot_events()[-1],
        {"type": "turn/end", "data": {"turn": 2, "reason": {"kind": "forked"}}},
    )
