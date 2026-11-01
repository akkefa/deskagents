"""Ported from packages/core/session/tests/tool-history.spec.ts.

Capability-independent tool history folded from committed headers and developer messages.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from deskagents.core.session import (
    Session,
    SessionEvent,
    SessionSeq,
    session_id,
    session_log_offset,
)
from deskagents.core.session.tool_history import ToolHistoryProjection
from deskagents.llm import (
    ContentBlock,
    Message,
    ToolSchema,
    create_developer_message,
    project_tool_updates,
)
from deskagents.util.values import FrozenJsonArray, FrozenJsonObject

SPEC = "packages/core/session/tests/tool-history.spec.ts"
GROUP = "Session.toolHistory"

SEARCH: ToolSchema = {"name": "search", "description": "Search", "parameters": {}}
FETCH: ToolSchema = {"name": "fetch", "description": "Fetch", "parameters": {}}
CONFIG = {"provider": "test", "model": "test"}


def opened(tools: list[ToolSchema]) -> Session:
    session = Session.create(session_id("tool-history"))
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append(
        "request/header", {"reason": "initial", "header": {"config": CONFIG, "tools": tools}}
    )
    return session


def header(
    session: Session,
    tools: list[ToolSchema],
    reason: Literal["change", "series"] = "change",
    starts_series: bool = False,
) -> SessionSeq:
    data: dict[str, Any] = {
        "reason": reason,
        "header": {"config": CONFIG, **({"tools": tools} if tools else {})},
    }
    if starts_series:
        data["startsSeries"] = True
    seq: SessionSeq = session.append("request/header", data)["seq"]
    return seq


def update(
    session: Session, content: list[ContentBlock], header_seq: SessionSeq | None = None
) -> Message:
    message = create_developer_message(source={"kind": "test"}, content=content)
    data: dict[str, Any] = {"turn": 1, "step": 1, "message": message}
    if header_seq is not None:
        data["headerSeq"] = header_seq
    session.append("developer/message", data, surface_op="append")
    return message


def add(name: str) -> ContentBlock:
    return {"type": "tool-addition", "toolName": name}


def remove(name: str) -> ContentBlock:
    return {"type": "tool-removal", "toolName": name}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > returns frozen active declarations with no updates for an initial header"
)
def test_returns_frozen_initial_declarations() -> None:
    session = opened([SEARCH])
    history = session.tool_history()
    assert history == {"tools": [SEARCH], "updates": []}
    assert isinstance(history, FrozenJsonObject)
    assert isinstance(history["tools"], FrozenJsonArray)
    assert session.tool_history() is history


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > resolves additions from their referenced header and keeps earlier "
    "snapshots unchanged"
)
def test_resolves_additions_from_referenced_header() -> None:
    session = opened([SEARCH])
    initial = session.tool_history()
    header_seq = header(session, [SEARCH, FETCH])
    added = update(session, [add("fetch")], header_seq)
    after_addition = session.tool_history()
    assert after_addition == {
        "tools": [SEARCH],
        "updates": [{"messageId": added["id"], "additions": [FETCH]}],
    }
    header(session, [SEARCH])
    removed = update(session, [remove("fetch"), {"type": "text", "text": "fetch is disabled"}])
    assert session.tool_history() == {
        "tools": [SEARCH],
        "updates": [
            {"messageId": added["id"], "additions": [FETCH]},
            {"messageId": removed["id"], "additions": []},
        ],
    }
    assert initial == {"tools": [SEARCH], "updates": []}
    assert len(after_addition["updates"]) == 1


@pytest.mark.parametrize(
    ("reason", "starts_series"),
    [("series", False), ("change", True)],
)
@pytest.mark.upstream(f"{SPEC}::{GROUP} > restarts declarations at an explicit series start %#")
def test_restarts_at_explicit_series_start(
    reason: Literal["change", "series"], starts_series: bool
) -> None:
    session = opened([SEARCH])
    header_seq = header(session, [SEARCH, FETCH])
    update(session, [add("fetch")], header_seq)
    restart = header(session, [SEARCH, FETCH], reason, starts_series)
    assert session.tool_history() == {"tools": [SEARCH, FETCH], "updates": []}
    added = update(session, [add("fetch")], restart)
    assert session.tool_history() == {
        "tools": [SEARCH, FETCH],
        "updates": [{"messageId": added["id"], "additions": [FETCH]}],
    }
    projected = project_tool_updates([added], [SEARCH, FETCH], "in-history", session.tool_history())
    assert projected["messages"] == []


@pytest.mark.parametrize("mode", ["addition-only", "in-history"])
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > activates deferred baseline tools through folded history on %s routes"
)
def test_activates_deferred_baseline_tools(mode: str) -> None:
    deferred = {**FETCH, "deferLoading": True}
    tools = [SEARCH, deferred]
    session = opened(tools)
    baseline = next(
        event for event in session.snapshot_events() if event["type"] == "request/header"
    )
    added = update(session, [add("search"), add("fetch")], baseline["seq"])
    duplicate = update(session, [add("fetch")], baseline["seq"])
    history = session.tool_history()
    assert history["updates"] == [
        {"messageId": added["id"], "additions": [SEARCH, deferred]},
        {"messageId": duplicate["id"], "additions": [deferred]},
    ]
    projected = project_tool_updates([added, duplicate], tools, mode, history)
    assert projected["tools"] == tools
    assert projected["messages"] == [{**added, "content": [add("fetch")]}]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > restarts declarations when a retained name returns with a changed "
    "definition"
)
def test_restarts_on_changed_definition() -> None:
    session = opened([SEARCH, FETCH])
    header(session, [SEARCH])
    update(session, [remove("fetch")])
    changed = {**FETCH, "description": "Fetch v2"}
    header_seq = header(session, [SEARCH, changed])
    added = update(session, [add("fetch")], header_seq)
    assert session.tool_history() == {
        "tools": [SEARCH, changed],
        "updates": [{"messageId": added["id"], "additions": [changed]}],
    }


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > continues the series when a retained name is restored unchanged"
)
def test_continues_series_on_unchanged_restore() -> None:
    session = opened([SEARCH, FETCH])
    header(session, [SEARCH])
    removed = update(session, [remove("fetch")])
    header_seq = header(session, [SEARCH, FETCH])
    restored = update(session, [add("fetch")], header_seq)
    assert session.tool_history() == {
        "tools": [SEARCH, FETCH],
        "updates": [
            {"messageId": removed["id"], "additions": []},
            {"messageId": restored["id"], "additions": [FETCH]},
        ],
    }


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > falls back to complete active declarations while a header lacks its "
    "update record"
)
def test_falls_back_while_header_lacks_update() -> None:
    session = opened([SEARCH])
    header_seq = header(session, [SEARCH, FETCH])
    assert session.tool_history() == {"tools": [SEARCH, FETCH], "updates": []}
    added = update(session, [add("fetch")], header_seq)
    assert session.tool_history() == {
        "tools": [SEARCH],
        "updates": [{"messageId": added["id"], "additions": [FETCH]}],
    }
    header(session, [SEARCH])
    assert session.tool_history() == {"tools": [SEARCH], "updates": []}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > reconstructs inherited history on restore and continues from live events"
)
def test_reconstructs_history_on_restore() -> None:
    session = opened([SEARCH])
    header_seq = header(session, [SEARCH, FETCH])
    update(session, [add("fetch")], header_seq)
    events = session.snapshot_events()
    restored = Session.from_restore(
        session.id, events, session.header, session_log_offset(0), "shared-frozen"
    )
    assert restored.tool_history() == session.tool_history()
    header(restored, [SEARCH])
    removed = update(restored, [remove("fetch")])
    assert restored.tool_history()["updates"][-1] == {"messageId": removed["id"], "additions": []}
    assert len(session.tool_history()["updates"]) == 1


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses an addition whose referenced header lacks the definition"
)
def test_refuses_addition_missing_definition() -> None:
    projection = ToolHistoryProjection()

    def header_event(seq: int, reason: Literal["initial", "change"]) -> SessionEvent:
        return {
            "seq": SessionSeq(seq),
            "time": 0,
            "type": "request/header",
            "data": {"reason": reason, "header": {"config": CONFIG, "tools": [SEARCH]}},
        }

    message = create_developer_message(source={"kind": "test"}, content=[add("fetch")])
    developer_event: SessionEvent = {
        "seq": SessionSeq(2),
        "time": 0,
        "type": "developer/message",
        "surfaceOp": "append",
        "data": {"turn": 1, "step": 1, "message": message, "headerSeq": SessionSeq(1)},
    }
    projection.apply(header_event(0, "initial"))
    projection.apply(header_event(1, "change"))
    with pytest.raises(ValueError, match="tool history: missing definition for fetch"):
        projection.apply(developer_event)
