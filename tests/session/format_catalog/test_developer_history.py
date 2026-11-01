"""Ported from packages/session/session-format-catalog/tests/developer-history.spec.ts.

Not ported: "stores only the activated name in each addition" is a TypeScript
``expectTypeOf`` check with no runtime behaviour.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from deskagents.core.session import Session, build_fork_seed
from deskagents.llm import create_developer_message
from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatRestoreOptions,
    create_session_format_catalog,
)
from deskagents.session.format_catalog import session_format_catalog, session_format_catalog_options
from deskagents.session.format_v3_to_v4 import restore_released_v4_artifact

SPEC = "packages/session/session-format-catalog/tests/developer-history.spec.ts"
GROUP = "developer tool history"
STRICT_CURRENT = SessionFormatRestoreOptions(recovery="strict", validation="current")


def restore(session: Session) -> Session:
    header = {**session.header, "delegationDepth": 0}
    reader = session_format_catalog.create_restore(
        session_format_catalog.encode_current_header(header, session.inherited_event_count),
        STRICT_CURRENT,
    )
    for event in session.snapshot_events():
        reader.decode_row(session_format_catalog.encode_current_event(event))
    artifact = reader.finish()
    return Session.from_restore(
        artifact.header["id"],  # type: ignore[arg-type]
        artifact.events,
        artifact.header,
        artifact.inherited_event_count,
        "detached",
    )


def tool_states(session: Session) -> list[Any]:
    events = session.snapshot_events()
    initial = next((event for event in events if event["type"] == "request/header"), None)
    if initial is None:
        raise AssertionError("missing initial request header")
    active = {tool["name"]: tool for tool in initial["data"]["header"].get("tools", [])}
    states = [active.get("search")]
    for seq in session.surface.nodes:
        event = events[seq]
        if event["type"] != "developer/message":
            continue
        for block in event["data"]["message"]["content"]:
            if block["type"] == "tool-addition":
                source = events[event["data"]["headerSeq"]]
                if source["type"] != "request/header":
                    raise AssertionError("missing referenced header")
                tool = next(
                    (
                        t
                        for t in source["data"]["header"].get("tools", [])
                        if t["name"] == block["toolName"]
                    ),
                    None,
                )
                if tool is None:
                    raise AssertionError("missing referenced definition")
                active[block["toolName"]] = tool
            elif block["type"] == "tool-removal":
                active.pop(block["toolName"], None)
            else:
                continue
            states.append(active.get("search"))
    return states


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > defers ignorable developer interpretation until catalog completion "
    "(known: %s)"
)
@pytest.mark.parametrize("knows_developer", [False, True])
def test_defers_ignorable_developer_interpretation(knows_developer: bool) -> None:
    header: dict[str, Any] = {
        "version": 4,
        "id": "opaque-developer",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    events: list[dict[str, Any]] = [
        {
            "type": "developer/message",
            "seq": 0,
            "time": 1,
            "ignorable": True,
            "surfaceOp": {"future": True},
            "data": {
                "message": {"role": "developer", "source": {"kind": "plugin"}},
                "future": True,
            },
        },
        {"type": "turn/start", "seq": 1, "time": 2, "data": {"turn": 1}},
        {"type": "step/start", "seq": 2, "time": 3, "data": {"turn": 1, "step": 1}},
    ]
    known = {event["type"] for event in events}
    if not knows_developer:
        known.discard("developer/message")
    catalog = create_session_format_catalog(
        replace(
            session_format_catalog_options,
            restore_current=lambda candidate: restore_released_v4_artifact(candidate, known),
        )
    )
    reader = catalog.create_restore(
        session_format_catalog.encode_current_header(header, 0), STRICT_CURRENT
    )
    for event in events:
        reader.decode_row(event)
    if knows_developer:
        with pytest.raises(SessionFormatError):
            reader.finish()
    else:
        assert reader.finish() == SessionFormatArtifact(header, 0, events)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > reconstructs remove/re-add and replacement from historical headers after "
    "restart and fork"
)
def test_reconstructs_tool_history_after_restart_and_fork() -> None:
    parent = Session.create("tool-history")
    initial: dict[str, Any] = {"name": "search", "description": "Initial", "parameters": {}}
    first: dict[str, Any] = {
        "name": "search",
        "description": "First",
        "parameters": {"properties": {"query": {"type": "string"}}},
        "deferLoading": True,
    }
    second: dict[str, Any] = {
        "name": "search",
        "description": "Second",
        "parameters": {"properties": {"limit": {"type": "number"}}},
    }
    third: dict[str, Any] = {
        "name": "search",
        "description": "Third",
        "parameters": {"properties": {"exact": {"type": "boolean"}}},
    }
    expected = copy.deepcopy([initial, first, None, second, third])
    config = {"provider": "test", "model": "test"}

    def append(turn: int, tool: dict[str, Any]) -> None:
        header_seq = parent.append(
            "request/header", {"reason": "change", "header": {"config": config, "tools": [tool]}}
        )["seq"]
        message = create_developer_message(
            source={"kind": "tool-registry"},
            content=[{"type": "tool-addition", "toolName": tool["name"]}],
        )
        parent.append(
            "developer/message",
            {"turn": turn, "step": 1, "headerSeq": header_seq, "message": message},
            surface_op="append",
        )

    parent.append("turn/start", {"turn": 1})
    parent.append("step/start", {"turn": 1, "step": 1})
    parent.append(
        "request/header", {"reason": "initial", "header": {"config": config, "tools": [initial]}}
    )
    append(1, first)
    removal = create_developer_message(
        source={"kind": "tool-registry"}, content=[{"type": "tool-removal", "toolName": "search"}]
    )
    parent.append(
        "developer/message", {"turn": 1, "step": 1, "message": removal}, surface_op="append"
    )
    append(1, second)
    parent.append("step/end", {"turn": 1, "step": 1})
    boundary = parent.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})["seq"]
    parent.append("turn/start", {"turn": 2})
    parent.append("step/start", {"turn": 2, "step": 1})
    append(2, third)
    parent.append("step/end", {"turn": 2, "step": 1})
    parent.append("turn/end", {"turn": 2, "reason": {"kind": "completed"}})

    initial["description"] = "mutated initial definition"
    first["description"] = "mutated producer definition"
    first["parameters"]["properties"] = {}
    second["description"] = "mutated replacement"
    assert tool_states(parent) == expected
    assert tool_states(restore(parent)) == expected

    child_id = "tool-history-fork"
    child = Session.create(
        child_id,
        build_fork_seed(parent.snapshot_events(), boundary),
        {**parent.header, "id": child_id, "parentSession": parent.id, "isSeeded": True},
        boundary + 1,
    )
    assert tool_states(restore(child)) == expected[:4]
