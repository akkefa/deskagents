"""Ported from packages/core/session/tests/store-membership.spec.ts."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, Literal, cast

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import (
    RestoredSessionOptions,
    Session,
    SessionEvent,
    SessionSeedEventState,
    SessionStore,
    append_plugin_record,
    session_id,
    session_log_offset,
    session_seq,
)
from deskagents.util.values import FrozenJsonObject, deep_freeze

SPEC = "packages/core/session/tests/store-membership.spec.ts"
GROUP = "Session store membership"
RECORD = "plugin:test/store-membership"

session_module = importlib.import_module("deskagents.core.session.session")

Append = Callable[[Session], object]
Kind = Literal["events", "plugin records"]


def append_event(session: Session) -> object:
    return session.append("turn/start", {"turn": 1})


def append_record(session: Session) -> object:
    return append_plugin_record(session, RECORD, {"value": 1})


APPENDERS = [
    pytest.param(append_event, id="events"),
    pytest.param(append_record, id="plugin records"),
]
KINDS = ["events", "plugin records"]


@pytest_asyncio.fixture
async def ctx() -> AsyncIterator[Context]:
    root = Context()
    await root.plugin(SessionStore)
    yield root
    await root.fiber.dispose()


def sessions(ctx: Context) -> SessionStore:
    store = ctx.get("sessions")
    assert isinstance(store, SessionStore)
    return store


async def dispose(effect: Callable[[], object]) -> None:
    result = effect()
    if result is not None:
        await result  # type: ignore[misc]


def append_reading(
    monkeypatch: pytest.MonkeyPatch, session: Session, kind: Kind, on_read: Callable[[], None]
) -> Callable[[], None]:
    """An append whose payload read runs ``on_read`` (upstream: a payload getter)."""
    original = session_module.snapshot_json_value

    def snapshot(value: object, **options: Any) -> object:
        if value is payload:
            on_read()
        return original(value, **options)

    monkeypatch.setattr(session_module, "snapshot_json_value", snapshot)
    payload: dict[str, Any] = {"turn": 1} if kind == "events" else {"value": 1}

    def append() -> None:
        if kind == "events":
            session.append("turn/start", payload)
        else:
            append_plugin_record(session, RECORD, payload)

    return append


def matches(actual: object, expected: object) -> bool:
    """Vitest ``toMatchObject``: ``expected`` is a recursive subset of ``actual``."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and matches(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps independent $name local and adopts their history without "
    "replaying notifications"
)
@pytest.mark.parametrize("append", APPENDERS)
async def test_keeps_independent_appends_local_and_adopts_history(
    ctx: Context, append: Append
) -> None:
    store = sessions(ctx)
    session = Session.create(session_id("independent"))
    observed: list[SessionEvent] = []
    initial_history: list[Sequence[SessionEvent]] = []
    disposal_errors: list[BaseException] = []

    def on_created(created: Session) -> None:
        initial_history.append(created.snapshot_events())
        with pytest.raises(ValueError, match="cannot append before session/created completes"):
            append(created)

    def on_disposed(disposed: Session) -> None:
        try:
            append(disposed)
        except Exception as error:
            disposal_errors.append(error)

    ctx.on("session/event", lambda _session, event: observed.append(event))
    ctx.on("session/created", on_created)
    ctx.on("session/disposed", on_disposed)

    append(session)
    history = session.snapshot_events()
    assert len(history) == 1
    assert isinstance(history[0], FrozenJsonObject)
    assert isinstance(history[0]["data"], FrozenJsonObject)
    assert observed == []

    leave = ctx.effect(lambda: store.enter(session))
    with pytest.raises(ValueError, match="cannot append before session/created completes"):
        append(session)
    store.announce(session)
    assert initial_history[0] is history
    assert observed == []
    append(session)
    assert len(history) == 1
    assert observed == list(session.snapshot_events(session_log_offset(1)))

    await dispose(leave)
    assert len(disposal_errors) == 1
    assert isinstance(disposal_errors[0], Exception)
    assert "cannot append outside a SessionStore" in str(disposal_errors[0])
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        append(session)
    assert session.seq == 2


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > allows $name only after creation announcement and before removal"
)
@pytest.mark.parametrize("append", APPENDERS)
async def test_allows_appends_only_between_announcement_and_removal(
    ctx: Context, append: Append
) -> None:
    store = sessions(ctx)
    session = store.prepare(session_id("membership"))
    observed: list[SessionEvent] = []
    ctx.on("session/event", lambda _session, event: observed.append(event))
    empty = session.snapshot_events()

    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        append(session)
    assert session.snapshot_events() is empty
    assert observed == []

    leave = ctx.effect(lambda: store.enter(session))
    with pytest.raises(ValueError, match="cannot append before session/created completes"):
        append(session)

    def on_created(created: Session) -> None:
        if created is not session:
            return
        with pytest.raises(ValueError, match="cannot append before session/created completes"):
            append(session)
        assert session.snapshot_events() is empty

    ctx.on("session/created", on_created)
    store.announce(session)
    append(session)
    committed = session.snapshot_events()
    assert len(committed) == 1
    assert observed == list(committed)

    await dispose(leave)
    replacement = store.create(session.id)
    assert store.get(session.id) is replacement
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        append(session)
    assert session.snapshot_events() is committed
    assert replacement.seq == 0
    assert observed == list(committed)


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{GROUP} > keeps writes disabled after a failed announcement")
async def test_keeps_writes_disabled_after_failed_announcement(ctx: Context) -> None:
    store = sessions(ctx)
    session = store.prepare(session_id("failed-announcement"))
    ctx.effect(lambda: store.enter(session))

    def fail(_session: Session) -> None:
        raise ValueError("creation observer failed")

    ctx.on("session/created", fail)
    with pytest.raises(ValueError, match="creation observer failed"):
        store.announce(session)
    with pytest.raises(ValueError, match="cannot append before session/created completes"):
        session.append("turn/start", {"turn": 1})
    with pytest.raises(ValueError, match="already announced"):
        store.announce(session)
    assert session.seq == 0


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects %s before reading payloads when writes are disabled"
)
@pytest.mark.parametrize("kind", KINDS)
async def test_rejects_before_reading_payloads_when_writes_disabled(
    ctx: Context, monkeypatch: pytest.MonkeyPatch, kind: Kind
) -> None:
    store = sessions(ctx)
    session = store.prepare(session_id("unread-payload"))
    reads: list[None] = []
    append = append_reading(monkeypatch, session, kind, lambda: reads.append(None))

    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        append()
    leave = ctx.effect(lambda: store.enter(session))
    with pytest.raises(ValueError, match="cannot append before session/created completes"):
        append()

    def on_created(_session: Session) -> None:
        with pytest.raises(ValueError, match="cannot append before session/created completes"):
            append()

    ctx.on("session/created", on_created)
    store.announce(session)
    await dispose(leave)
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        append()
    assert reads == []
    assert session.seq == 0


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects %s when reading the payload detaches the Session")
@pytest.mark.parametrize("kind", KINDS)
async def test_rejects_when_reading_payload_detaches_session(
    ctx: Context, monkeypatch: pytest.MonkeyPatch, kind: Kind
) -> None:
    store = sessions(ctx)
    session = store.prepare(session_id("payload-detach"))
    detach = store.enter(session)
    ctx.effect(lambda: detach)
    store.announce(session)
    observed: list[SessionEvent] = []
    ctx.on("session/event", lambda _session, event: observed.append(event))
    append = append_reading(monkeypatch, session, kind, detach)

    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        append()
    assert store.get(session.id) is None
    assert session.seq == 0
    assert observed == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rechecks %s after payload evaluation transfers an independent Session "
    "to a store"
)
@pytest.mark.parametrize("kind", KINDS)
async def test_rechecks_after_payload_evaluation_transfers_session(
    ctx: Context, monkeypatch: pytest.MonkeyPatch, kind: Kind
) -> None:
    store = sessions(ctx)
    session = Session.create(session_id("payload-adoption"))

    def read() -> None:
        ctx.effect(lambda: store.enter(session))

    append = append_reading(monkeypatch, session, kind, read)
    with pytest.raises(ValueError, match="cannot append before session/created completes"):
        append()
    assert store.get(session.id) is session
    assert session.seq == 0
    store.announce(session)
    session.append("turn/start", {"turn": 1})
    assert session.seq == 1


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > appends to independent constructor, restore, and fork history without "
    "publication"
)
async def test_appends_to_independent_history_without_publication(ctx: Context) -> None:
    observed: list[SessionEvent] = []
    ctx.on("session/event", lambda _session, event: observed.append(event))
    seed = cast(
        Sequence[SessionEvent],
        deep_freeze(
            [{"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}}]
        ),
    )
    sid = session_id("history")
    created = Session.create(sid, seed)
    restored = Session.from_restore(
        sid, seed, created.header, session_log_offset(0), "shared-frozen"
    )
    child_id = session_id("child")
    forked = Session.create(
        child_id,
        seed,
        {**created.header, "id": child_id, "parentSession": sid, "isSeeded": True},
        session_log_offset(1),
    )
    for session in [created, restored, forked]:
        history = session.snapshot_events()
        assert history[0] == seed[0]
        assert matches(
            history[1],
            {
                "type": "session/end-seed",
                "seq": 1,
                "data": {"inherited": True} if session is forked else {},
            },
        )
        assert isinstance(history[1], FrozenJsonObject)
        assert isinstance(history[1]["data"], FrozenJsonObject)
        session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
        assert len(history) == 2
        assert session.seq == 3
    assert observed == []


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{GROUP} > keeps prepared restored history read-only (%s)")
@pytest.mark.parametrize("event_state", ["detached", "shared-frozen"])
async def test_keeps_prepared_restored_history_read_only(
    ctx: Context, event_state: SessionSeedEventState
) -> None:
    store = sessions(ctx)
    source = Session.create(session_id("restored"))
    source.append("turn/start", {"turn": 1})
    session = store.prepare(
        source.id,
        RestoredSessionOptions(
            seed=[*source.snapshot_events()],
            meta=source.header,
            inherited_event_count=session_log_offset(0),
            event_state=event_state,
        ),
    )
    history = session.snapshot_events()
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert session.snapshot_events() is history


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{GROUP} > revokes writes when the owning fiber releases a session")
async def test_revokes_writes_when_owning_fiber_releases_session(ctx: Context) -> None:
    session = sessions(ctx).create(session_id("owned"))
    session.append("turn/start", {"turn": 1})
    await ctx.fiber.dispose()

    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert session.seq == 1
