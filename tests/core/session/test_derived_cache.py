"""Ported from packages/core/session/tests/derived-cache.spec.ts.

Derived-message cache contract against a scratch oracle: project new nodes once,
rebuild on surface replacements, return fresh lists over shared frozen messages, and
remain value-equal to replay at every step.
"""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.core.session import Session, session_id
from deskagents.llm import Message, create_message, create_user_message
from deskagents.util.values import FrozenJsonArray, FrozenJsonObject

SPEC = "packages/core/session/tests/derived-cache.spec.ts"
CACHE = "derived-message cache"
PER_EVENT = "Session.deriveEventMessage — the per-event projection"


def checkpoint_source(compaction_id: str) -> dict[str, str]:
    """Build a checkpoint source for a derived-message fixture."""
    return {"kind": "compact-checkpoint", "compactionId": compaction_id}


def user_text(session: Session, text: str) -> None:
    session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"}),
        surface_op="append",
    )


def assistant(content: list[dict[str, Any]]) -> Message:
    return create_message(
        role="assistant",
        content=content,
        source={"kind": "model", "provider": "mock", "model": "mock"},
    )


def scratch(session: Session) -> list[Message]:
    """From-scratch oracle: replay the log into a fresh session and derive."""
    replay_id = session_id(f"{session.id}-scratch-{session.seq}")
    return Session.create(replay_id, session.snapshot_events()).derive_messages()


@pytest.mark.upstream(
    f"{SPEC}::{CACHE} > stays deep-equal to a from-scratch replay derivation as the log grows"
)
def test_stays_equal_to_scratch_replay() -> None:
    session = Session.create(session_id("cache-grow"))
    session.append("turn/start", {"turn": 1})
    user_text(session, "one")
    assert session.derive_messages() == scratch(session)
    user_text(session, "two")
    session.append(
        "assistant/message",
        {
            "stream": [],
            "turn": 1,
            "step": 1,
            "message": assistant([{"type": "text", "text": "reply"}]),
        },
        surface_op="append",
    )
    assert session.derive_messages() == scratch(session)
    session.append(
        "assistant/message",
        {
            "stream": [],
            "turn": 1,
            "step": 2,
            "message": assistant([]),
            "usage": {"inputTokens": 1, "outputTokens": 0},
        },
        surface_op="append",
    )
    assert session.derive_messages() == scratch(session)


@pytest.mark.upstream(f"{SPEC}::{CACHE} > rebuilds on a surface replace and still matches scratch")
def test_rebuilds_on_surface_replace() -> None:
    session = Session.create(session_id("cache-replace"))
    session.append("turn/start", {"turn": 1})
    user_text(session, "one")
    user_text(session, "two")
    before_replace = session.derive_messages()
    assert len(before_replace) == 2

    nodes = session.surface.nodes
    session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "summary"}],
            source=checkpoint_source("derived-cache-compaction"),
        ),
        surface_op={"op": "replace", "startSeq": nodes[0], "endSeq": nodes[1]},
        source_event_seqs=[nodes[0], nodes[1]],
    )

    assert len(session.derive_messages()) == 1
    assert session.derive_messages() == scratch(session)
    assert len(before_replace) == 2


@pytest.mark.upstream(
    f"{SPEC}::{CACHE} > returns a fresh array per call: later appends never grow a held snapshot"
)
def test_returns_fresh_list_per_call() -> None:
    session = Session.create(session_id("cache-snapshot"))
    session.append("turn/start", {"turn": 1})
    user_text(session, "one")
    first = session.derive_messages()
    user_text(session, "two")
    second = session.derive_messages()
    assert len(first) == 1
    assert len(second) == 2
    # List snapshots share their frozen message projections.
    assert second[0] is first[0]
    assert isinstance(first[0], FrozenJsonObject)


@pytest.mark.upstream(
    f"{SPEC}::{PER_EVENT} > projects one appended event exactly as the full derivation "
    "projects its node"
)
def test_projects_event_like_full_derivation() -> None:
    session = Session.create(session_id("per-event"))
    session.append("turn/start", {"turn": 1})
    event = session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "hi"}], source={"kind": "user"}),
        surface_op="append",
    )
    # Full and per-event derivation share one projection.
    assert session.derive_event_message(event) == session.derive_messages()[-1]


# The spec escapes the apostrophe; the catalog checker matches the source literal.
@pytest.mark.upstream(f"{SPEC}::{PER_EVENT} > reuses the logged event\\'s already frozen content")
def test_reuses_frozen_content() -> None:
    session = Session.create(session_id("per-event-clone"))
    session.append("turn/start", {"turn": 1})
    event = session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "orig"}], source={"kind": "user"}),
        surface_op="append",
    )
    message = session.derive_event_message(event)
    assert message is not None
    assert message["content"] is event["data"]["content"]
    assert isinstance(message["content"], FrozenJsonArray)
    assert isinstance(message["content"][0], FrozenJsonObject)
    with pytest.raises(TypeError):
        message["content"][0]["text"] = "mutated"
    assert session.derive_messages()[-1]["content"] == [{"type": "text", "text": "orig"}]


@pytest.mark.upstream(
    f"{SPEC}::{PER_EVENT} > projects null for events that produce no message "
    "(boundaries, empty assistant)"
)
def test_projects_none_for_messageless_events() -> None:
    session = Session.create(session_id("per-event-null"))
    session.append("turn/start", {"turn": 1})
    boundary = session.append("step/start", {"turn": 1, "step": 1})
    assert session.derive_event_message(boundary) is None
    empty = session.append(
        "assistant/message",
        {"stream": [], "turn": 1, "step": 1, "message": assistant([])},
        surface_op="append",
    )
    assert session.derive_event_message(empty) is None
