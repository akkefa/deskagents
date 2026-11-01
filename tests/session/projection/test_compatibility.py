"""Ported from packages/session/session-projection/tests/compatibility.spec.ts.

Not ported (no Python equivalent of JavaScript thenable assimilation; see the package
docstring):

- "rejects an async plugin that returns the legacy registration as a disposer": an
  ``async`` apply returning the registration hands Cordis the registration itself, which
  it adopts as cleanup, instead of an assimilated handle that fails as "Invalid effect".
- "leaves an async generator yield owned by the plugin instead of the composed effect":
  an async generator ``yield`` does not await its value, so the composed effect adopts the
  registration instead of receiving the handle.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
from deepseek_cordis import Context, CordisError, FiberState
from pydantic import TypeAdapter

from deskagents.core.session import SessionStore
from deskagents.session.projection import (
    ProjectionDefinition,
    ProjectionRegistration,
    ProjectionSnapshot,
    ProjectionWire,
    SessionProjectionRegistry,
)

SPEC = "packages/session/session-projection/tests/compatibility.spec.ts"
COMPAT = "projection registration compatibility"
LOGGER = "deskagents.session.projection.registry"
DEPRECATED = (
    "sessionProjections.stateOf(session, {}) is deprecated; read handle.stateOf(session) from "
    "an awaited registration, or use the owning domain service"
)

NUMBER: TypeAdapter[int] = TypeAdapter(int)

DEFINITION: ProjectionDefinition[int] = ProjectionDefinition(
    key="test/legacy",
    state_schema=NUMBER,
    state_version=1,
    init=lambda _header, _inherited: 0,
    apply=lambda state, _event: state + 1,
    wire=ProjectionWire(view_schema=NUMBER, view=lambda state: state),
)


@pytest.fixture(autouse=True)
def capture(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    return caplog


def logged(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.levelno == level
    ]


def projections_of(ctx: Context) -> SessionProjectionRegistry:
    service = ctx.get("sessionProjections")
    assert isinstance(service, SessionProjectionRegistry)
    return service


def legacy_registration(registration: ProjectionRegistration[int]) -> ProjectionRegistration[int]:
    """Exercise calls emitted by plugins written against the old disposer return type."""
    assert callable(registration)
    return registration


async def call_legacy(registration: ProjectionRegistration[int]) -> None:
    """Upstream ``await registration()``: the disposer result is awaitable or ``None``."""
    result = registration()
    if result is not None:
        await result


def plugin(apply: Callable[[Context, object], object]) -> SimpleNamespace:
    return SimpleNamespace(inject=["sessionProjections"], apply=apply)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > awaits modern registration, keeps Promise chaining, and never warns for "
    "handle cleanup"
)
async def test_awaits_modern_registration(
    ctx: Context,
    registry: SessionProjectionRegistry,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = store.create()
    settled: list[bool] = []

    async def apply(inner: Context, config: object) -> None:
        registration = projections_of(inner).register(DEFINITION)
        assert not isinstance(registration, asyncio.Future)
        assert inspect.isawaitable(registration)
        handle = await registration
        # Awaiting again (upstream's then/finally chaining) yields the same handle.
        assert await registration is handle
        settled.append(True)
        assert handle.state_of(session) == 0
        await handle.dispose()
        session.append("turn/start", {"turn": 1})
        assert handle.state_of(session) == 1

    owner = await ctx.plugin(plugin(apply))
    assert settled == [True]
    assert registry.snapshot(session).values == {}
    await owner.dispose()
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > lets an asynchronous plugin await registration later in its initialization"
)
async def test_awaits_registration_later(ctx: Context, caplog: pytest.LogCaptureFixture) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def apply(inner: Context, config: object) -> None:
        registration = projections_of(inner).register(DEFINITION)
        entered.set()
        await release.wait()
        await registration

    owner = ctx.plugin(plugin(apply))
    await entered.wait()
    release.set()
    await owner
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > loads an unawaited legacy registration and warns once per key for keyed "
    "reads"
)
async def test_unawaited_legacy_registration_warns_once_per_key(
    ctx: Context,
    registry: SessionProjectionRegistry,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = store.create()

    def apply(inner: Context, config: object) -> None:
        projections_of(inner).register(DEFINITION)

    first = await ctx.plugin(plugin(apply))
    second = await ctx.plugin(plugin(apply))
    session.append("turn/start", {"turn": 1})
    assert registry.state_of(session, "test/legacy") == 1
    assert registry.state_of(session, "test/legacy") == 1
    assert registry.state_of(session, "test/missing-legacy") is None
    assert logged(caplog, logging.WARNING) == [
        DEPRECATED.format('"test/legacy"'),
        DEPRECATED.format('"test/missing-legacy"'),
    ]
    await first.dispose()
    assert registry.state_of(session, "test/legacy") == 1
    await second.dispose()
    assert registry.state_of(session, "test/legacy") is None
    assert len(logged(caplog, logging.WARNING)) == 2


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > materializes every registered unit on a legacy keyed read without "
    "producing unrelated wire views"
)
async def test_legacy_read_materializes_every_unit(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    unrelated_view = Mock(side_effect=lambda state: state)
    host_init = Mock(return_value=0)
    await registry.register(DEFINITION)
    await registry.register(
        replace(
            DEFINITION,
            key="test/legacy-unread-wire",
            wire=ProjectionWire(view_schema=NUMBER, view=unrelated_view),
        )
    )
    await registry.register(
        ProjectionDefinition(
            key="test/legacy-unread-host",
            state_schema=DEFINITION.state_schema,
            state_version=DEFINITION.state_version,
            init=host_init,
            apply=DEFINITION.apply,
        )
    )
    assert registry.state_of(session, "test/missing-legacy") is None
    assert registry.cached_snapshot(session) is None
    host_init.assert_not_called()

    assert registry.state_of(session, "test/legacy") == 2
    unrelated_view.assert_not_called()
    assert registry.cached_snapshot(session) == ProjectionSnapshot(
        1, {"test/legacy": 2, "test/legacy-unread-wire": 2}
    )
    host_init.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > keeps callable early disposal idempotent without invalidating an "
    "awaited reader"
)
async def test_callable_early_disposal_is_idempotent(
    registry: SessionProjectionRegistry, store: SessionStore, caplog: pytest.LogCaptureFixture
) -> None:
    session = store.create()
    registration = legacy_registration(registry.register(DEFINITION))
    handle = await registration
    await call_legacy(registration)
    await call_legacy(registration)
    await handle.dispose()
    session.append("turn/start", {"turn": 1})
    assert handle.state_of(session) == 1
    assert registry.snapshot(session).values == {}
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > releases a registration called before the asynchronous handle handoff"
)
async def test_release_before_handoff(
    registry: SessionProjectionRegistry, store: SessionStore, caplog: pytest.LogCaptureFixture
) -> None:
    session = store.create()
    registration = legacy_registration(registry.register(DEFINITION))
    await call_legacy(registration)
    handle = await registration
    assert handle.state_of(session) == 0
    assert registry.snapshot(session).values == {}
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["return", "effect", "generator"])
@pytest.mark.upstream(f"{SPEC}::{COMPAT} > unloads a legacy %s registration")
async def test_unloads_legacy_registration(
    ctx: Context,
    registry: SessionProjectionRegistry,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
    style: Literal["return", "effect", "generator"],
) -> None:
    session = store.create()

    def apply(inner: Context, config: object) -> object:
        projections = projections_of(inner)
        if style == "return":
            return legacy_registration(projections.register(DEFINITION))
        if style == "effect":
            inner.effect(lambda: legacy_registration(projections.register(DEFINITION)))
            return None

        def setup() -> Iterator[object]:
            yield legacy_registration(projections.register(DEFINITION))

        inner.effect(setup)
        return None

    owner = await ctx.plugin(plugin(apply))
    assert registry.snapshot(session).values == {"test/legacy": 0}
    await owner.dispose()
    assert registry.snapshot(session).values == {}
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > transfers the original disposer into a composed effect and retains it "
    "until earlier cleanup settles"
)
async def test_transfers_disposer_into_composed_effect(
    ctx: Context, registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    release = asyncio.Event()
    closing = asyncio.Event()

    def apply(inner: Context, config: object) -> None:
        projections = projections_of(inner)

        def setup() -> Iterator[object]:
            yield legacy_registration(projections.register(DEFINITION))

            async def cleanup() -> None:
                closing.set()
                await release.wait()

            yield cleanup

        inner.effect(setup)

    owner = await ctx.plugin(plugin(apply))
    disposed = owner.dispose()
    await closing.wait()
    session.append("turn/start", {"turn": 1})
    assert registry.snapshot(session).values == {"test/legacy": 1}
    release.set()
    await disposed
    assert registry.snapshot(session).values == {}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > preserves explicit rejection handling and returns a harmless legacy "
    "disposer for failed acquisition"
)
async def test_failed_acquisition_disposer_is_harmless(
    registry: SessionProjectionRegistry, store: SessionStore, caplog: pytest.LogCaptureFixture
) -> None:
    session = store.create()
    registration = legacy_registration(registry.register(replace(DEFINITION, state_version=-1)))
    with pytest.raises(ValueError):
        await registration
    await call_legacy(registration)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert registry.snapshot(session).values == {}
    assert logged(caplog, logging.ERROR) == []
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > logs an ignored legacy registration rejection and keeps the plugin active"
)
async def test_logs_ignored_rejection(
    ctx: Context,
    registry: SessionProjectionRegistry,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = store.create()

    def apply(inner: Context, config: object) -> None:
        projections = projections_of(inner)
        projections.register(DEFINITION)
        projections.register(replace(DEFINITION, state_version=2))
        inner.provide("legacy-failed-provider", {})

    owner = await ctx.plugin(plugin(apply))
    for _ in range(10):
        if logged(caplog, logging.ERROR):
            break
        await asyncio.sleep(0)
    errors = logged(caplog, logging.ERROR)
    assert len(errors) == 1
    assert "already registered at stateVersion 1" in errors[0]
    assert owner.state is FiberState.ACTIVE
    assert ctx.get("legacy-failed-provider") is not None
    assert registry.snapshot(session).values == {"test/legacy": 0}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > lets a modern plugin handle rejected registration and finish activation"
)
async def test_modern_plugin_handles_rejection(
    ctx: Context,
    registry: SessionProjectionRegistry,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = store.create()

    async def apply(inner: Context, config: object) -> None:
        projections = projections_of(inner)
        invalid = projections.register(replace(DEFINITION, state_version=-1))
        with pytest.raises(ValueError):
            await invalid
        await projections.register(DEFINITION)

    owner = await ctx.plugin(plugin(apply))
    assert owner.state is FiberState.ACTIVE
    assert registry.snapshot(session).values == {"test/legacy": 0}
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > warns again for a new registry instance without sharing warning state "
    "between applications"
)
async def test_warns_again_for_new_registry(
    registry: SessionProjectionRegistry, store: SessionStore, caplog: pytest.LogCaptureFixture
) -> None:
    session = store.create()
    await registry.register(DEFINITION)
    assert registry.state_of(session, "test/legacy") == 0
    assert registry.state_of(session, "test/legacy") == 0
    assert len(logged(caplog, logging.WARNING)) == 1
    second = Context()
    try:
        await second.plugin(SessionProjectionRegistry)
        assert projections_of(second).state_of(session, "test/legacy") is None
        # Both registries log through one module logger: one warning each.
        assert logged(caplog, logging.WARNING) == [DEPRECATED.format('"test/legacy"')] * 2
    finally:
        await second.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > drops an ignored rejection when its plugin unloads before handoff"
)
async def test_drops_rejection_when_plugin_unloads(
    ctx: Context,
    registry: SessionProjectionRegistry,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = store.create()

    def apply(inner: Context, config: object) -> None:
        projections_of(inner).register(DEFINITION)
        inner.fiber.dispose()

    owner = ctx.plugin(plugin(apply))
    await owner
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert owner.state is FiberState.DISPOSED
    assert registry.snapshot(session).values == {}
    assert logged(caplog, logging.ERROR) == []
    assert logged(caplog, logging.WARNING) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{COMPAT} > keeps an observed registration owned when its plugin unloads before "
    "handoff"
)
async def test_observed_registration_rejects_after_unload(
    ctx: Context,
    registry: SessionProjectionRegistry,
    store: SessionStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = store.create()

    def apply(inner: Context, config: object) -> None:
        return None

    owner = await ctx.plugin(plugin(apply))
    registration = registry.register(DEFINITION, ctx=owner.ctx)
    await owner.dispose()
    # deepseek-cordis names the inactive owner INACTIVE_EFFECT (upstream "inactive context").
    with pytest.raises(CordisError) as rejected:
        await registration
    assert rejected.value.code == "INACTIVE_EFFECT"
    assert registry.snapshot(session).values == {}
    assert logged(caplog, logging.WARNING) == []
