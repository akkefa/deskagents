"""Ported from packages/core/session/tests/request-header.spec.ts."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from deskagents.core.session import (
    EpochHeader,
    Session,
    SessionEvent,
    canonical_header,
    fold_request_header,
    header_equals,
    session_id,
    session_seq,
)
from deskagents.llm import ReasoningEffortId, ToolSchema, create_user_message
from deskagents.util.values import FrozenJsonObject

SPEC = "packages/core/session/tests/request-header.spec.ts"

CONFIG: dict[str, Any] = {"provider": "mock", "model": "m"}


def tool(name: str, description: str = "d") -> ToolSchema:
    return {"name": name, "description": description, "parameters": {"type": "object"}}


def user_text(text: str) -> Any:
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})


@pytest.mark.upstream(
    f"{SPEC}::canonicalHeader > normalizes empty optional fields to absence and preserves "
    "populated fields"
)
def test_canonical_header_normalizes_empty_fields() -> None:
    assert canonical_header({"config": CONFIG, "adapterDefaults": {}, "tools": []}) == {
        "config": CONFIG
    }
    full = canonical_header(
        {
            "config": {**CONFIG, "maxTokens": 256_000},
            "adapterDefaults": {"maxTokens": True},
            "tools": [tool("a")],
        }
    )
    assert full == {
        "config": {**CONFIG, "maxTokens": 256_000},
        "adapterDefaults": {"maxTokens": True},
        "tools": [tool("a")],
    }


BASE = canonical_header({"config": CONFIG, "tools": [tool("a")]})


@pytest.mark.upstream(
    f"{SPEC}::headerEquals > compares every canonical field and preserves tool order"
)
def test_header_equals_compares_every_field() -> None:
    base = BASE
    assert header_equals(base, copy.deepcopy(base)) is True
    assert header_equals(base, {**base, "config": {"provider": "mock", "model": "other"}}) is False
    assert (
        header_equals(
            base,
            {**base, "config": {**base["config"], "reasoningEffort": ReasoningEffortId("high")}},
        )
        is False
    )
    assert (
        header_equals(
            {**base, "config": {**base["config"], "maxTokens": 256_000}},
            {
                **base,
                "config": {**base["config"], "maxTokens": 256_000},
                "adapterDefaults": {"maxTokens": True},
            },
        )
        is False
    )
    assert header_equals(base, {**base, "tools": []}) is False
    assert header_equals(base, {**base, "tools": [tool("a", "changed")]}) is False
    assert (
        header_equals(
            {"config": CONFIG, "tools": [tool("a"), tool("b")]},
            {"config": CONFIG, "tools": [tool("b"), tool("a")]},
        )
        is False
    )


@pytest.mark.upstream(
    f"{SPEC}::headerEquals > treats absent and empty tool arrays as equivalent canonical absence"
)
def test_header_equals_absent_and_empty_tools() -> None:
    assert header_equals({"config": CONFIG}, {"config": CONFIG, "tools": []}) is True
    assert header_equals({"config": CONFIG, "tools": []}, {"config": CONFIG}) is True
    assert header_equals({"config": CONFIG}, {"config": CONFIG}) is True


@pytest.mark.upstream(
    f"{SPEC}::foldRequestHeader > returns the supplied baseline when no snapshot follows"
)
def test_fold_returns_baseline() -> None:
    start: EpochHeader = {"config": CONFIG, "tools": [tool("baseline")]}
    unrelated: list[SessionEvent] = [
        {"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}},
    ]
    assert fold_request_header(unrelated) is None
    assert fold_request_header(unrelated, start) is start


@pytest.mark.upstream(
    f"{SPEC}::foldRequestHeader > takes the latest full snapshot and skips unrelated events"
)
def test_fold_takes_latest_snapshot() -> None:
    session = Session.create(session_id("fold"))
    session.append("turn/start", {"turn": 1})
    session.append(
        "request/header",
        {"header": {"config": CONFIG, "tools": [tool("first")]}, "reason": "initial"},
    )
    session.append("user/message", user_text("hi"), surface_op="append")
    session.append(
        "request/header",
        {"header": {"config": {"provider": "mock", "model": "other"}}, "reason": "change"},
    )
    assert fold_request_header(session.snapshot_events()) == {
        "config": {"provider": "mock", "model": "other"}
    }


CAPACITY: dict[str, Any] = {"provider": "mock", "model": "m", "contextWindow": 128_000}


def seed_with(*records: dict[str, Any]) -> list[SessionEvent]:
    """A turn-enclosed capacity record; the invariant rejects one outside a turn."""
    events: list[SessionEvent] = [
        {"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}},
    ]
    for data in records:
        events.append(
            {"type": "request/context", "seq": session_seq(len(events)), "time": 1, "data": data}
        )
    return events


@pytest.mark.upstream(f"{SPEC}::Session.requestContext > reads undefined before any record exists")
def test_request_context_undefined_before_record() -> None:
    assert Session.create(session_id("no-capacity")).request_context() is None


@pytest.mark.upstream(
    f"{SPEC}::Session.requestContext > folds a seeded log on first read, taking the last record"
)
def test_request_context_folds_seeded_log() -> None:
    # The fold watermark starts at 0 with the seed already in the log, so the
    # first read must consume the whole seed rather than skip it.
    session = Session.create(
        session_id("seeded-capacity"),
        seed_with(CAPACITY, {**CAPACITY, "model": "later", "contextWindow": 256_000}),
    )
    assert session.request_context() == {
        "provider": "mock",
        "model": "later",
        "contextWindow": 256_000,
    }


@pytest.mark.upstream(
    f"{SPEC}::Session.requestContext > advances incrementally across appends and skips "
    "unrelated events"
)
def test_request_context_advances_incrementally() -> None:
    session = Session.create(session_id("incremental-capacity"), seed_with(CAPACITY))
    assert session.request_context() == CAPACITY
    session.append("user/message", user_text("unrelated"), surface_op="append")
    assert session.request_context() == CAPACITY
    session.append("request/context", {**CAPACITY, "model": "next", "contextWindow": 64_000})
    assert session.request_context() == {
        "provider": "mock",
        "model": "next",
        "contextWindow": 64_000,
    }
    session.append("request/context", {"provider": "mock", "model": "unknown"})
    assert session.request_context() == {"provider": "mock", "model": "unknown"}


@pytest.mark.upstream(f"{SPEC}::Session.requestContext > folds a batch appended between two reads")
def test_request_context_folds_batch() -> None:
    session = Session.create(session_id("batched-capacity"), seed_with(CAPACITY))
    assert session.request_context() == CAPACITY
    session.append("request/context", {**CAPACITY, "contextWindow": 200_000})
    session.append("user/message", user_text("unrelated"), surface_op="append")
    session.append("request/context", {**CAPACITY, "contextWindow": 300_000})
    context = session.request_context()
    assert context is not None
    assert context.get("contextWindow") == 300_000


@pytest.mark.upstream(
    f"{SPEC}::Session.requestContext > exposes a frozen record so a reader cannot desync later "
    "comparisons"
)
def test_request_context_is_frozen() -> None:
    session = Session.create(session_id("frozen-capacity"), seed_with(CAPACITY))
    held = session.request_context()
    if held is None:
        raise AssertionError("expected a folded capacity record")
    assert isinstance(held, FrozenJsonObject)
    with pytest.raises(TypeError):
        held["contextWindow"] = 1
