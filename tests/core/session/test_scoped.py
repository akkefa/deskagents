"""Ported from packages/core/session/tests/scoped.spec.ts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.scope import Scope, ScopeKey, create_scope, scope_of
from deskagents.core.session import Session, SessionEvent, SessionStore

SPEC = "packages/core/session/tests/scoped.spec.ts"
CARRIERS = "session dispatch carriers"
FLUSH = "sessions.flush()"


class Key:
    """Identity scope key (upstream ``{ name }``; ``object()`` cannot be weakly referenced)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Key({self.name!r})"


@pytest_asyncio.fixture
async def ctx() -> AsyncIterator[Context]:
    root = Context()
    await root.plugin(SessionStore)
    yield root
    await root.fiber.dispose()


async def mint_scope(ctx: Context, name: str) -> Scope:
    minted: list[Scope] = []

    # The scoped context resolves services through the minting plugin's dependency
    # chain, so the minter injects what scope holders will reach.
    def host(inner: Context, config: object) -> None:
        minted.append(create_scope(inner, Key(name)))

    await ctx.inject(["sessions"], host)
    return minted[0]


def key_of(scope: Scope) -> ScopeKey:
    """The key a test scope was minted with."""
    key = scope_of(scope.ctx)
    assert key is not None
    return key


def sessions(ctx: Context) -> SessionStore:
    store = ctx.get("sessions")
    assert isinstance(store, SessionStore)
    return store


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{CARRIERS} > a session entered through a scoped context dispatches its events in "
    "that scope"
)
async def test_scoped_session_dispatches_events_in_its_scope(ctx: Context) -> None:
    scope = await mint_scope(ctx, "owner")
    other_scope = await mint_scope(ctx, "other")

    heard: list[str] = []

    def hear(label: str) -> Callable[[Session, SessionEvent], None]:
        return lambda _session, event: heard.append(f"{label}:{event['type']}")

    ctx.on("session/event", hear("global"))
    scope.ctx.on("session/event", hear("owner"))
    other_scope.ctx.on("session/event", hear("other"))
    scope.ctx.on("session/created", lambda session: heard.append(f"owner-created:{session.id}"))
    other_scope.ctx.on(
        "session/created", lambda session: heard.append(f"other-created:{session.id}")
    )

    session = sessions(scope.ctx).create(ctx=scope.ctx)
    session.append("turn/start", {"turn": 1})

    assert heard == [f"owner-created:{session.id}", "global:turn/start", "owner:turn/start"]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{CARRIERS} > a bare session dispatches subject-less: scoped listeners never hear it"
)
async def test_bare_session_dispatches_subject_less(ctx: Context) -> None:
    scope = await mint_scope(ctx, "owner")
    heard: list[str] = []

    def global_listener(_session: Session, event: SessionEvent) -> None:
        heard.append(f"global:{event['type']}")

    def owner_listener(_session: Session, event: SessionEvent) -> None:
        heard.append(f"owner:{event['type']}")

    ctx.on("session/event", global_listener)
    scope.ctx.on("session/event", owner_listener)

    bare = sessions(ctx).create()
    bare.append("turn/start", {"turn": 1})
    assert heard == ["global:turn/start"]


@pytest.mark.asyncio
async def test_fork_forwards_explicit_caller_context(ctx: Context) -> None:
    """Python service calls need ``ctx=`` where upstream Cordis binds ``this.ctx``."""
    owner = await mint_scope(ctx, "owner")
    store = sessions(ctx)
    source = store.create("fork-source")
    source.append("turn/start", {"turn": 1})

    child = store.fork(source, child_session_id="fork-child", ctx=owner.ctx)
    assert store.get("fork-child") is child

    await owner.dispose()
    assert store.get("fork-child") is None
    assert store.get("fork-source") is source


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{CARRIERS} > reuses the captured owner carrier for the paired disposal notification"
)
async def test_reuses_owner_carrier_for_disposal(ctx: Context) -> None:
    owner = await mint_scope(ctx, "owner")
    other = await mint_scope(ctx, "other")
    heard: list[str] = []
    ctx.on("session/disposed", lambda session: heard.append(f"global:{session.id}"))
    owner.ctx.on("session/disposed", lambda session: heard.append(f"owner:{session.id}"))
    other.ctx.on("session/disposed", lambda session: heard.append(f"other:{session.id}"))

    store = sessions(owner.ctx)
    session = store.prepare()
    detach = store.enter(session, ctx=owner.ctx)
    store.announce(session)
    detach()

    assert heard == [f"global:{session.id}", f"owner:{session.id}"]


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{FLUSH} > allows an ordinary flush with no listeners")
async def test_allows_flush_with_no_listeners(ctx: Context) -> None:
    session = sessions(ctx).create()

    assert await sessions(ctx).flush(session) is False


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{FLUSH} > reports a participating listener after it succeeds")
async def test_reports_participating_listener(ctx: Context) -> None:
    session = sessions(ctx).create()
    flushed: list[Session] = []
    ctx.on("session/flush", flushed.append)

    assert await sessions(ctx).flush(session) is True

    assert flushed == [session]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FLUSH} > dispatches session/flush with the owning carrier and awaits all listeners"
)
async def test_flush_uses_owning_carrier_and_awaits_listeners(ctx: Context) -> None:
    scope = await mint_scope(ctx, "owner")
    flushed: list[str] = []

    async def global_listener(session: Session) -> None:
        await asyncio.sleep(0)
        flushed.append(f"global:{session.id}")

    ctx.on("session/flush", global_listener)
    scope.ctx.on("session/flush", lambda session: flushed.append(f"owner:{session.id}"))

    owned = sessions(scope.ctx).create(ctx=scope.ctx)
    bare = sessions(ctx).create()
    await sessions(ctx).flush(owned)
    await sessions(ctx).flush(bare)

    # Parallel dispatch: listener completion order is unspecified (the global listener
    # awaits a loop turn), so assert set membership per flush instead.
    assert sorted(flushed[:2]) == [f"global:{owned.id}", f"owner:{owned.id}"]
    assert flushed[2:] == [f"global:{bare.id}"]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FLUSH} > propagates a rejecting flush listener (the caller owns the failure policy)"
)
async def test_propagates_rejecting_flush_listener(ctx: Context) -> None:
    async def reject(_session: Session) -> None:
        raise ValueError("disk full")

    ctx.on("session/flush", reject)
    session = sessions(ctx).create()
    with pytest.raises(ValueError, match="disk full"):
        await sessions(ctx).flush(session)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FLUSH} > does not let a synchronous flush failure starve later listeners"
)
async def test_synchronous_flush_failure_does_not_starve_listeners(ctx: Context) -> None:
    flushed: list[Session] = []

    def fail(_session: Session) -> None:
        raise ValueError("disk full")

    ctx.on("session/flush", fail)
    ctx.on("session/flush", flushed.append)
    session = sessions(ctx).create()

    with pytest.raises(ValueError, match="disk full"):
        await sessions(ctx).flush(session)
    assert flushed == [session]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FLUSH} > waits for slower flush listeners before reporting another listener failure"
)
async def test_waits_for_slower_listeners_before_reporting_failure(ctx: Context) -> None:
    gate: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    slow_started = False

    async def reject(_session: Session) -> None:
        raise ValueError("disk full")

    def slow(_session: Session) -> asyncio.Future[None]:
        nonlocal slow_started
        slow_started = True
        return gate

    ctx.on("session/flush", reject)
    ctx.on("session/flush", slow)
    session = sessions(ctx).create()

    flushing = asyncio.ensure_future(sessions(ctx).flush(session))
    # Python's flush is a coroutine: give its task a few loop turns to start listeners.
    for _ in range(5):
        await asyncio.sleep(0)
    assert slow_started
    assert not flushing.done()

    gate.set_result(None)
    with pytest.raises(ValueError, match="disk full"):
        await flushing
    assert flushing.done()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{FLUSH} > rejects a never-entered session instead of inventing a carrier"
)
async def test_rejects_flush_of_never_entered_session(ctx: Context) -> None:
    scope = await mint_scope(ctx, "owner")
    flushed: list[str] = []
    ctx.on("session/flush", lambda session: flushed.append(f"global:{session.id}"))
    scope.ctx.on("session/flush", lambda session: flushed.append(f"owner:{session.id}"))

    prepared = sessions(ctx).prepare()
    with pytest.raises(ValueError, match="not live"):
        await sessions(ctx).flush(prepared)
    assert flushed == []


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{FLUSH} > clears a detached carrier and rejects stale flushes")
async def test_clears_detached_carrier_and_rejects_stale_flushes(ctx: Context) -> None:
    scope = await mint_scope(ctx, "owner")
    flushed: list[str] = []
    ctx.on("session/flush", lambda session: flushed.append(f"global:{session.id}"))
    scope.ctx.on("session/flush", lambda session: flushed.append(f"owner:{session.id}"))

    store = sessions(scope.ctx)
    session = store.prepare()
    detach = store.enter(session, ctx=scope.ctx)
    await sessions(ctx).flush(session)
    assert sorted(flushed) == [f"global:{session.id}", f"owner:{session.id}"]

    detach()
    with pytest.raises(ValueError, match="not live"):
        await sessions(ctx).flush(session)
    assert len(flushed) == 2


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{FLUSH} > keyOf sanity: distinct scopes carry distinct keys")
async def test_distinct_scopes_carry_distinct_keys(ctx: Context) -> None:
    a = await mint_scope(ctx, "a")
    b = await mint_scope(ctx, "b")
    assert key_of(a) is not key_of(b)
