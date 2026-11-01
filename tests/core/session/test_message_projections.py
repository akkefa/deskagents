"""Ported from packages/core/session/tests/message-projections.spec.ts."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from deepseek_cordis import Context

from deskagents.core.session import (
    MESSAGE_PROJECTION_EVENT_TYPES,
    RestoredSessionOptions,
    Session,
    SessionEvent,
    SessionMessageProjection,
    SessionMessageProjectionContext,
    SessionSeq,
    SessionStore,
    SurfaceManager,
    derive_event_message,
    fold_surface,
    session_id,
    session_log_offset,
    session_seq,
)
from deskagents.llm import Message, create_user_message
from deskagents.util.values import deep_freeze

SPEC = "packages/core/session/tests/message-projections.spec.ts"
GROUP = "plugin-owned message projections"


class TestProjection:
    """Replaces the referenced message's content with ``data.text``."""

    __test__ = False
    type = "test/project"

    def project(
        self, event: SessionEvent, context: SessionMessageProjectionContext
    ) -> Mapping[SessionSeq, Message]:
        source = context.events[event["data"]["seq"] - context.base_seq]
        original = derive_event_message(source, context.messages)
        assert original is not None
        if event["data"]["text"] == "reject":
            raise ValueError("rejected decision")
        content = [{"type": "text", "text": event["data"]["text"]}]
        return {source["seq"]: deep_freeze({**original, "content": content})}


projection = TestProjection()


def text(value: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": value}]


def input_(session: Session) -> SessionEvent:
    return session.append(
        "user/message",
        create_user_message(content=text("original"), source={"kind": "user"}),
        surface_op="append",
    )


def store_of(ctx: Context) -> SessionStore:
    store = ctx.get("sessions")
    assert isinstance(store, SessionStore)
    return store


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses required events when no interpreter is supplied")
def test_refuses_required_events_without_interpreter() -> None:
    type_ = next(iter(MESSAGE_PROJECTION_EVENT_TYPES))
    event: SessionEvent = {"type": type_, "seq": session_seq(0), "time": 0, "data": {}}
    session = Session.create(session_id("missing"))
    with pytest.raises(ValueError, match="requires a message projection"):
        session.append(event["type"], event["data"])
    with pytest.raises(ValueError, match="requires a message projection"):
        fold_surface([event])
    with pytest.raises(ValueError, match="requires a message projection"):
        Session.create(session.id, [event])
    assert session.seq == 0


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > applies generic decisions atomically and replays them through "
    "detached folds"
)
def test_applies_decisions_atomically_and_replays() -> None:
    definitions: list[SessionMessageProjection] = [projection]
    session = Session.create(session_id("pure"), None, None, None, definitions)
    source = input_(session)
    original = session.derive_messages()
    with pytest.raises(ValueError, match="rejected decision"):
        session.append("test/project", {"seq": source["seq"], "text": "reject"})
    assert session.derive_messages() == original
    session.append("test/project", {"seq": source["seq"], "text": "projected"})
    assert session.derive_messages()[0]["content"] == text("projected")
    assert session.surface.replace_generation == 0
    assert session.surface.content_generation == 1
    folded = fold_surface(session.snapshot_events(), definitions)
    assert derive_event_message(source, folded.projected_messages) == session.derive_messages()[0]
    assert source["data"]["content"] == text("original")


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > registers by fiber and refuses cached or pending decisions after disposal"
)
async def test_registers_by_fiber_and_refuses_after_disposal() -> None:
    ctx = Context()
    try:
        await ctx.plugin(SessionStore)

        def apply(owner: Context, config: object) -> None:
            store_of(owner).register_message_projection(projection)

        fiber = ctx.inject(["sessions"], apply)
        await fiber
        sessions = store_of(ctx)
        with pytest.raises(ValueError, match="already registered"):
            sessions.register_message_projection(projection)
        live = sessions.create(session_id("live"))
        source = input_(live)
        live.append("test/project", {"seq": source["seq"], "text": "changed"})
        pending = sessions.create(session_id("pending"))
        input_(pending)
        pending.append("test/project", {"seq": source["seq"], "text": "changed"})
        before = live.derive_messages()
        child = sessions.fork(live)
        assert child.derive_messages() == before
        restored = sessions.prepare(
            session_id("restore"),
            RestoredSessionOptions(
                seed=list(live.snapshot_events()),
                meta={**live.header, "id": session_id("restore")},
                inherited_event_count=session_log_offset(live.inherited_event_count),
                event_state="shared-frozen",
            ),
        )
        ctx.effect(lambda: sessions.enter(restored))
        sessions.announce(restored)
        assert restored.derive_messages() == before
        await fiber.dispose()
        assert list(sessions.message_projections) == []
        for session in (live, pending, child, restored):
            with pytest.raises(ValueError, match="was removed or replaced"):
                session.derive_messages()
            with pytest.raises(ValueError, match="was removed or replaced"):
                session.derive_event_message(source)
            with pytest.raises(ValueError, match="was removed or replaced"):
                _ = session.surface.replace_generation
            with pytest.raises(ValueError, match="was removed or replaced"):
                _ = session.surface.content_generation
            with pytest.raises(ValueError, match="was removed or replaced"):
                session.append("turn/start", {"turn": 1})
        assert before[0]["content"] == text("changed")
    finally:
        await ctx.fiber.dispose()


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > applies a supplied interpreter to a loaded window with absolute sequences"
)
def test_applies_interpreter_to_loaded_window() -> None:
    session = Session.create(session_id("window"), None, None, None, [projection])
    input_(session)
    source = input_(session)
    session.append("test/project", {"seq": source["seq"], "text": "window"})
    events = session.snapshot_events()[1:]
    surface = SurfaceManager(events, session_log_offset(1), [projection])
    assert list(surface.nodes) == [source["seq"]]
    message = surface.derive_event_message(source)
    assert message is not None
    assert message["content"] == text("window")


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > does not retain an interpreter for a candidate that never committed"
)
def test_does_not_retain_uncommitted_interpreter() -> None:
    session = Session.create(session_id("candidate"))
    source = input_(session)
    definitions: list[SessionMessageProjection] = [projection]
    surface = SurfaceManager(session.snapshot_events(), projections=definitions)
    surface.validate_next(
        {
            "type": "test/project",
            "seq": session_seq(1),
            "time": 0,
            "data": {"seq": source["seq"], "text": "unused"},
        }
    )
    definitions.clear()
    assert surface.derive_event_message(source) is source["data"]
