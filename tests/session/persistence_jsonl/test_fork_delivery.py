"""Ported from packages/session/session-persistence-jsonl/tests/fork-delivery.spec.ts.

Only the ``compression: 'none'`` rows of this upstream matrix are ported here.
Zstandard framing and delivery are covered by the backend and contract suites.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import (
    CreateSessionMeta,
    CreateSessionOptions,
    Session,
    SessionId,
    SessionLogOffset,
    SessionStore,
)
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import SessionPersistence
from deskagents.session.persistence_jsonl.format import generation_log_path

SPEC = "packages/session/session-persistence-jsonl/tests/fork-delivery.spec.ts"
DESCRIBE = "fork delivery persistence (%s)"


@dataclass
class Fixture:
    root: Path
    sessions: SessionStore
    persistence: SessionPersistence

    async def persist(self, session: Session) -> Path:
        writer = await self.persistence.create(
            session.header, inherited_event_count=session.inherited_event_count
        )
        try:
            await writer.append(session.snapshot_events())
        finally:
            await writer.close()
        return Path(generation_log_path(str(self.root), None, session.id, 4, "none"))


@pytest_asyncio.fixture
async def fixture(tmp_path: Path) -> AsyncIterator[Fixture]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(jsonl.apply, {"root": str(tmp_path), "compression": "none"})
    sessions = ctx.get("sessions")
    persistence = ctx.get("sessionPersistence")
    assert isinstance(sessions, SessionStore)
    assert isinstance(persistence, SessionPersistence)
    try:
        yield Fixture(tmp_path, sessions, persistence)
    finally:
        await ctx.fiber.dispose()


def delivered(session: Session) -> None:
    start = session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    session.append(
        "session-log-deepseek/delivery-accepted",
        {"sessionId": session.id, "sessionFormatVersion": 4, "throughSeq": start["seq"]},
    )


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > restores %s fork history through read and write access"
)
@pytest.mark.parametrize("kind", ["ordinary", "nested", "subagent", "legacy"])
@pytest.mark.asyncio
async def test_restores_fork_history(fixture: Fixture, kind: str) -> None:
    sessions = fixture.sessions
    source = sessions.create(SessionId("source"))
    delivered(source)
    ordinary = sessions.fork(source, None, SessionId("ordinary"))
    if kind == "ordinary":
        child = ordinary
    elif kind == "nested":
        child = sessions.fork(ordinary, None, SessionId("nested"))
    else:
        child = sessions.create(
            SessionId(kind),
            CreateSessionOptions(
                seed=source.snapshot_events(),
                inherited_event_count=SessionLogOffset(source.seq),
                meta=CreateSessionMeta(
                    parent_session=source.id,
                    is_seeded=True,
                    origin="subagent" if kind == "subagent" else None,
                ),
            ),
        )
    path = await fixture.persist(child)
    data = path.read_bytes()
    for access in ("read", "write"):
        try:
            handle = await fixture.persistence.open(child.id, access)
            try:
                stored = await handle.read()
                restored = Session.from_restore(
                    child.id,
                    stored.events,
                    handle.header,
                    handle.inherited_event_count,
                    stored.event_state,
                )
                assert restored.snapshot_events() == child.snapshot_events()
                assert restored.inherited_event_count == child.inherited_event_count
                expected_parent = source.id if kind in ("subagent", "legacy") else None
                assert restored.header.get("parentSession") == expected_parent
            finally:
                await handle.close()
        finally:
            assert path.read_bytes() == data


@pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > refuses %s delivery without modifying the file")
@pytest.mark.parametrize("damage", ["foreign-own", "invalid-through", "inherited-through"])
@pytest.mark.asyncio
async def test_refuses_damaged_delivery(fixture: Fixture, damage: str) -> None:
    sessions = fixture.sessions
    source = sessions.create(SessionId("source"))
    delivered(source)
    invalid_target = (
        source if damage == "inherited-through" else sessions.fork(source, None, SessionId("child"))
    )
    invalid_target.append(
        "session-log-deepseek/delivery-accepted",
        {
            "sessionId": source.id if damage == "foreign-own" else invalid_target.id,
            "sessionFormatVersion": 4,
            "throughSeq": 0 if damage == "foreign-own" else invalid_target.seq,
        },
    )
    child = sessions.fork(source) if damage == "inherited-through" else invalid_target
    path = await fixture.persist(child)
    data = path.read_bytes()
    expected = "wrong Session" if damage == "foreign-own" else "throughSeq must precede"
    for access in ("read", "write"):
        with pytest.raises(Exception, match=expected):
            await fixture.persistence.open(child.id, access)
        assert path.read_bytes() == data
