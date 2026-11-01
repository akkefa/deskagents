"""Python-written session values conform to upstream's v4 persistence inventory.

Covers every core event type the Phase 2 session code writes, the synthetic closers
written by repair and fork, and the store's header. Upstream: docs/persistence-schema.json
at the pin (tests/upstream_assets).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from deepseek_cordis import Context
from support.persistence_schema import conformance_problems, event_problems, root_keys

from deskagents.core.session import (
    CreateSessionMeta,
    CreateSessionOptions,
    Session,
    SessionEvent,
    SessionStore,
    build_fork_seed,
    interrupted_turn_closers,
    session_seq,
)
from deskagents.llm import (
    ToolCallId,
    create_assistant_message,
    create_developer_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
)

CALL = ToolCallId("call-1")
TOOL = {"name": "read", "description": "Read a file.", "parameters": {"type": "object"}}


def full_turn(session: Session) -> None:
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append(
        "request/header",
        {
            "header": {
                "config": {"provider": "deepseek", "model": "deepseek-chat", "maxTokens": 1024},
                "adapterDefaults": {"maxTokens": True},
                "tools": [TOOL],
            },
            "reason": "initial",
        },
    )
    session.append(
        "request/context",
        {"provider": "deepseek", "model": "deepseek-chat", "contextWindow": 128000},
    )
    session.append(
        "system/message",
        {"turn": 1, "step": 1, "message": create_system_message("You help.")},
        surface_op="append",
    )
    session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "hi"}], source={"kind": "user"}),
        surface_op="append",
    )
    session.append(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": create_assistant_message(
                content=[
                    {"type": "text", "text": "reading"},
                    {"type": "tool-call", "id": CALL, "name": "read", "arguments": "{}"},
                ],
                source={"provider": "deepseek", "model": "deepseek-chat"},
            ),
            "stream": [],
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        },
        surface_op="append",
    )
    call = session.append(
        "tool/call", {"turn": 1, "step": 1, "callId": CALL, "name": "read", "arguments": "{}"}
    )
    session.append(
        "tool/result",
        {
            "turn": 1,
            "step": 1,
            "message": create_tool_result_message(
                call_id=CALL, content=[{"type": "text", "text": "ok"}], is_error=False
            ),
        },
        surface_op="append",
        source_event_seqs=[session_seq(call["seq"])],
    )
    session.append("assistant/attempt", {"turn": 1, "step": 1, "stream": []})
    header = session.append(
        "request/header",
        {
            "header": {"config": {"provider": "deepseek", "model": "deepseek-chat"}},
            "reason": "change",
        },
    )
    session.append(
        "developer/message",
        {
            "turn": 1,
            "step": 1,
            "message": create_developer_message(
                content=[{"type": "tool-removal", "toolName": "read"}],
                source={"kind": "tool-registry"},
            ),
        },
        surface_op="append",
    )
    assert header["seq"] > 0
    session.append("step/end", {"turn": 1, "step": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})


def open_tool_turn(session: Session) -> None:
    session.append("turn/start", {"turn": 2})
    session.append("step/start", {"turn": 2, "step": 1})
    session.append(
        "assistant/message",
        {
            "turn": 2,
            "step": 1,
            "message": create_assistant_message(
                content=[{"type": "tool-call", "id": "call-2", "name": "read", "arguments": "{}"}],
                source={"provider": "deepseek", "model": "deepseek-chat"},
            ),
            "stream": [],
        },
        surface_op="append",
    )
    session.append(
        "tool/call", {"turn": 2, "step": 1, "callId": "call-2", "name": "read", "arguments": "{}"}
    )


@pytest.fixture
def session() -> Iterator[Session]:
    session = Session.create("conformance")
    full_turn(session)
    open_tool_turn(session)
    yield session


def test_every_core_event_type_is_an_inventory_root() -> None:
    written = {
        "turn/start",
        "turn/end",
        "step/start",
        "step/end",
        "user/message",
        "developer/message",
        "system/message",
        "assistant/message",
        "assistant/attempt",
        "tool/call",
        "tool/result",
        "request/header",
        "request/context",
        "session/end-seed",
    }
    assert {f"event:{kind}" for kind in written} <= root_keys()


def test_live_events_conform(session: Session) -> None:
    for event in session.snapshot_events():
        assert event_problems(event) == [], event["type"]


def test_interrupted_closers_conform(session: Session) -> None:
    closers: list[SessionEvent] = interrupted_turn_closers(session.snapshot_events())
    assert [event["type"] for event in closers] == ["tool/result", "step/end", "turn/end"]
    for event in closers:
        assert event_problems(event) == [], event["type"]


def test_fork_seed_conforms(session: Session) -> None:
    events = session.snapshot_events()
    seed = build_fork_seed(events, session_seq(len(events) - 1))
    for event in seed:
        assert event_problems(event) == [], event["type"]
    child = Session.create(
        "child", seed, {"version": 4, "id": "child", "createdAt": 1, "isSeeded": True}, len(events)
    )
    for event in child.snapshot_events():
        assert event_problems(event) == [], event["type"]


@pytest.mark.asyncio
async def test_store_header_conforms() -> None:
    ctx = Context()
    await ctx.plugin(SessionStore)
    store = ctx.get("sessions")
    assert isinstance(store, SessionStore)
    session = store.create(
        "header",
        CreateSessionOptions(
            meta=CreateSessionMeta(
                cwd="/tmp",
                parent_session="parent",
                origin="subagent",
                delegation_depth=1,
                agent_preset="coder",
            )
        ),
    )
    assert conformance_problems("SessionHeader", session.header) == []
    await ctx.fiber.dispose()


def test_validator_reports_undeclared_and_missing_fields() -> None:
    bad = {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1, "extra": True}}
    assert any("not declared" in problem for problem in event_problems(bad))
    missing = {"type": "turn/start", "seq": 0, "time": 1, "data": {}}
    assert any("required property missing" in problem for problem in event_problems(missing))
