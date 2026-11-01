"""Ported from packages/core/session/tests/developer-header.spec.ts.

Tool additions bind to one historical request header across Session lifecycle paths.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from deskagents.core.session import (
    Session,
    SessionEvent,
    SessionSeq,
    build_fork_seed,
    fold_surface,
    session_id,
    session_log_offset,
    session_seq,
)
from deskagents.llm import Message, ToolSchema, create_developer_message

SPEC = "packages/core/session/tests/developer-header.spec.ts"
GROUP = "developer request-header references"

FIRST: ToolSchema = {"name": "search", "description": "First", "parameters": {"type": "object"}}
SECOND: ToolSchema = {
    "name": "search",
    "description": "Second",
    "parameters": {"properties": {"query": {"type": "string"}}},
}
OTHER: ToolSchema = {"name": "other", "description": "Other", "parameters": {}}
CONFIG = {"provider": "test", "model": "test"}


def addition() -> Message:
    return create_developer_message(
        source={"kind": "test"}, content=[{"type": "tool-addition", "toolName": "search"}]
    )


def opened(tools: list[ToolSchema] | None = None) -> tuple[Session, SessionSeq]:
    session = Session.create(session_id("developer-header"))
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    header = session.append(
        "request/header",
        {"reason": "initial", "header": {"config": CONFIG, "tools": tools or [FIRST]}},
    )
    return session, header["seq"]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps an older header binding through compound changes, replacement, "
    "restore, and fork"
)
def test_keeps_older_header_binding() -> None:
    session, header_seq = opened([FIRST, OTHER])
    session.append(
        "request/header", {"reason": "change", "header": {"config": CONFIG, "tools": [SECOND]}}
    )
    message = create_developer_message(
        source={"kind": "test"},
        content=[
            {"type": "tool-removal", "toolName": "search"},
            {"type": "tool-addition", "toolName": "search"},
            {"type": "tool-addition", "toolName": "other"},
        ],
    )
    initial = session.append(
        "developer/message",
        {"turn": 1, "step": 1, "headerSeq": header_seq, "message": message},
        surface_op="append",
    )
    replacement = session.append(
        "developer/message",
        {"turn": 1, "step": 1, "headerSeq": header_seq, "message": addition()},
        surface_op={"op": "replace", "startSeq": initial["seq"], "endSeq": initial["seq"]},
        source_event_seqs=[initial["seq"]],
    )
    events = session.snapshot_events()
    assert events[header_seq]["data"]["header"]["tools"] == [FIRST, OTHER]
    assert replacement["data"]["message"]["content"] == [
        {"type": "tool-addition", "toolName": "search"}
    ]
    assert fold_surface(events).nodes == [replacement["seq"]]
    restored = Session.from_restore(
        session.id, events, session.header, session_log_offset(0), "shared-frozen"
    )
    assert restored.derive_messages() == [replacement["data"]["message"]]
    child_id = session_id("developer-header-child")
    child = Session.create(
        child_id,
        build_fork_seed(events, replacement["seq"]),
        {**session.header, "id": child_id, "parentSession": session.id, "isSeeded": True},
        session_log_offset(replacement["seq"] + 1),
    )
    assert child.snapshot_events()[replacement["seq"]] == replacement
    assert child.snapshot_events()[header_seq] == events[header_seq]


@pytest.mark.parametrize("reference", [0, 3, 100])
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses a non-header or non-earlier reference %s before append"
)
def test_refuses_non_header_or_non_earlier_reference(reference: int) -> None:
    session, _ = opened()
    before = session.snapshot_events()
    with pytest.raises(ValueError, match="earlier request/header"):
        session.append(
            "developer/message",
            {"turn": 1, "step": 1, "headerSeq": session_seq(reference), "message": addition()},
            surface_op="append",
        )
    assert session.snapshot_events() is before


@pytest.mark.parametrize("tools", [[{**FIRST, "name": "different"}], [FIRST, SECOND]])
@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses missing or ambiguous historical tool names %#")
def test_refuses_missing_or_ambiguous_tool_names(tools: list[ToolSchema]) -> None:
    session, header_seq = opened(tools)
    with pytest.raises(ValueError, match="exactly one tool"):
        session.append(
            "developer/message",
            {"turn": 1, "step": 1, "headerSeq": header_seq, "message": addition()},
            surface_op="append",
        )


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses an addition bound to a tool-less header")
def test_refuses_addition_bound_to_toolless_header() -> None:
    session = Session.create(session_id("without-tools"))
    header_seq = session.append(
        "request/header", {"reason": "initial", "header": {"config": CONFIG}}
    )["seq"]
    with pytest.raises(ValueError, match="exactly one tool"):
        session.append(
            "developer/message",
            {"turn": 1, "step": 1, "headerSeq": header_seq, "message": addition()},
            surface_op="append",
        )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > requires a header exactly for additions and rejects inline definitions"
)
def test_requires_header_exactly_for_additions() -> None:
    session, header_seq = opened()
    with pytest.raises(ValueError, match="requires headerSeq"):
        session.append(
            "developer/message", {"turn": 1, "step": 1, "message": addition()}, surface_op="append"
        )
    removal = create_developer_message(
        source={"kind": "test"}, content=[{"type": "tool-removal", "toolName": "search"}]
    )
    with pytest.raises(ValueError, match="requires headerSeq"):
        session.append(
            "developer/message",
            {"turn": 1, "step": 1, "headerSeq": header_seq, "message": removal},
            surface_op="append",
        )
    inline = create_developer_message(
        source={"kind": "test"},
        content=[{"type": "tool-addition", "toolName": "search", "tool": FIRST}],
    )
    with pytest.raises(ValueError, match="omit inline"):
        session.append(
            "developer/message",
            {"turn": 1, "step": 1, "headerSeq": header_seq, "message": inline},
            surface_op="append",
        )
    unnamed = create_developer_message(
        source={"kind": "test"}, content=[{"type": "tool-addition", "toolName": ""}]
    )
    with pytest.raises(ValueError, match="nonempty toolName"):
        session.append(
            "developer/message",
            {"turn": 1, "step": 1, "headerSeq": header_seq, "message": unnamed},
            surface_op="append",
        )


@pytest.mark.parametrize(
    "tool",
    [
        {"name": "search"},
        {"name": "search", "description": None, "parameters": {}},
        {"name": "search", "description": "", "parameters": []},
        {"name": "search", "description": "", "parameters": {}, "deferLoading": False},
    ],
)
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses an incomplete referenced definition in imported history %#"
)
def test_refuses_incomplete_referenced_definition(tool: dict[str, Any]) -> None:
    session, header_seq = opened()
    source = session.snapshot_events()[header_seq]
    header = {"config": CONFIG, "tools": [tool]}
    events: list[SessionEvent] = json.loads(
        json.dumps(
            [
                *session.snapshot_events()[:header_seq],
                {**source, "data": {"reason": "initial", "header": header}},
                {
                    "type": "developer/message",
                    "seq": 3,
                    "time": 4,
                    "surfaceOp": "append",
                    "data": {"turn": 1, "step": 1, "headerSeq": header_seq, "message": addition()},
                },
            ]
        )
    )
    with pytest.raises(ValueError, match="complete tool definition|deferLoading"):
        Session.create(session_id("incomplete-tool"), events)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > applies the same binding check to imported events and full-log "
    "surface folds"
)
def test_applies_binding_check_to_imports_and_folds() -> None:
    session, _ = opened()
    invalid: SessionEvent = {
        "type": "developer/message",
        "seq": session_seq(3),
        "time": 4,
        "surfaceOp": "append",
        "data": {"turn": 1, "step": 1, "headerSeq": session_seq(0), "message": addition()},
    }
    events = [*session.snapshot_events(), invalid]
    with pytest.raises(ValueError, match="earlier request/header"):
        Session.create(session_id("bad-import"), events)
    with pytest.raises(ValueError, match="earlier request/header"):
        fold_surface(events)
