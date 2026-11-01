"""Ported from packages/session/session-projection/tests/registration-reader.spec.ts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import Mock

import pytest
import pytest_asyncio
from deepseek_cordis import Context, CordisError, Fiber
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from deskagents.core.session import Session, SessionEvent, SessionStore
from deskagents.session.projection import (
    ProjectionDefinition,
    ProjectionWire,
    SessionProjectionRegistry,
)

SPEC = "packages/session/session-projection/tests/registration-reader.spec.ts"
READERS = "retained projection readers"

NUMBER: TypeAdapter[int] = TypeAdapter(int)


class Turn(TypedDict):
    turn: int


TURN: TypeAdapter[Turn | None] = TypeAdapter(Turn | None)

DEFINITION: ProjectionDefinition[int] = ProjectionDefinition(
    key="test/reader",
    state_schema=NUMBER,
    state_version=1,
    init=lambda _header, _inherited: 0,
    apply=lambda state, _event: state + 1,
    wire=ProjectionWire(view_schema=NUMBER, view=lambda state: state),
)


@dataclass
class Setup:
    ctx: Context
    registry: SessionProjectionRegistry
    store: SessionStore
    owner: Fiber
    session: Session


@pytest_asyncio.fixture
async def setup() -> AsyncIterator[Setup]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(SessionProjectionRegistry)

    def registration_owner(inner: Context, config: object) -> None:
        return None

    owner = await ctx.plugin(
        SimpleNamespace(inject=["sessionProjections"], apply=registration_owner)
    )
    registry = ctx.get("sessionProjections")
    store = ctx.get("sessions")
    assert isinstance(registry, SessionProjectionRegistry)
    assert isinstance(store, SessionStore)
    yield Setup(ctx, registry, store, owner, store.create())
    await ctx.fiber.dispose()


def turn_of(state: Any, event: SessionEvent) -> Any:
    return {"turn": event["data"]["turn"]} if event["type"] == "turn/start" else state


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{READERS} > binds typed host states to the exact Session without reading "
    "unrelated units"
)
async def test_binds_states_to_exact_session(setup: Setup) -> None:
    registry, owner, session = setup.registry, setup.owner, setup.session
    unrelated_init = Mock(return_value=0)
    unrelated_apply = Mock(side_effect=lambda state, _event: state)
    await registry.register(
        ProjectionDefinition(
            key="test/bound-unrelated",
            state_schema=NUMBER,
            state_version=1,
            init=unrelated_init,
            apply=unrelated_apply,
        )
    )
    count_init = Mock(side_effect=DEFINITION.init)
    count_apply = Mock(side_effect=DEFINITION.apply)
    count = await registry.register(
        replace(DEFINITION, init=count_init, apply=count_apply), ctx=owner.ctx
    )
    host_init = Mock(return_value=None)
    host_apply = Mock(side_effect=turn_of)
    host = await registry.register(
        ProjectionDefinition(
            key="test/bound-host",
            state_schema=TURN,
            state_version=1,
            init=host_init,
            apply=host_apply,
        ),
        ctx=owner.ctx,
    )
    initialized = {
        "count": count_init.call_count,
        "host": host_init.call_count,
        "unrelated": unrelated_init.call_count,
    }
    applied = {
        "count": count_apply.call_count,
        "host": host_apply.call_count,
        "unrelated": unrelated_apply.call_count,
    }
    reader = {"count": count.reader_for(session), "host": host.reader_for(session)}
    assert {
        "count": count_init.call_count,
        "host": host_init.call_count,
        "unrelated": unrelated_init.call_count,
    } == initialized
    assert {
        "count": count_apply.call_count,
        "host": host_apply.call_count,
        "unrelated": unrelated_apply.call_count,
    } == applied
    assert reader["count"]() == 0
    assert reader["host"]() is None
    assert unrelated_init.call_count == initialized["unrelated"]
    assert unrelated_apply.call_count == applied["unrelated"]

    other = setup.store.create()
    other_reader = count.reader_for(other)
    other.append("turn/start", {"turn": 1})
    assert other_reader() == 1
    assert reader["count"]() == 0
    session.append("turn/start", {"turn": 2})
    assert reader["count"]() == 1
    assert reader["host"]() == {"turn": 2}
    assert reader["host"]() is host.state_of(session)
    read = reader["host"]
    assert read() is host.state_of(session)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{READERS} > retains the bound unit after owner disposal and version replacement"
)
async def test_retains_bound_unit_after_disposal_and_replacement(setup: Setup) -> None:
    registry, owner, session = setup.registry, setup.owner, setup.session
    original = await registry.register(DEFINITION, ctx=owner.ctx)
    reader = original.reader_for(session)
    await owner.dispose()
    assert registry.checkpoint(session) == {}
    replacement = await registry.register(
        replace(DEFINITION, state_version=2, apply=lambda state, _event: state + 10)
    )
    changed = Mock()
    registry.on_changed(changed)
    session.append("turn/start", {"turn": 1})
    assert reader() == 1
    assert original.reader_for(session)() == 1
    assert replacement.state_of(session) == 10
    assert registry.snapshot(session).values == {"test/reader": 10}
    changed.assert_called_once()
    assert registry.checkpoint(session)["test/reader"]["ver"] == 2


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{READERS} > does not read during binding and preserves the original synchronous "
    "failure"
)
async def test_preserves_synchronous_failure(setup: Setup) -> None:
    failure = ValueError("host state cannot initialize")
    init = Mock(side_effect=failure)
    broken = await setup.registry.register(
        ProjectionDefinition(
            key="test/bound-failure",
            state_schema=NUMBER,
            state_version=1,
            init=init,
            apply=lambda state, _event: state,
        ),
        ctx=setup.owner.ctx,
    )
    initialized = init.call_count
    reader = broken.reader_for(setup.session)
    assert init.call_count == initialized
    with pytest.raises(ValueError) as caught:
        reader()
    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{READERS} > preserves undefined state without substituting a value")
async def test_preserves_undefined_state(setup: Setup) -> None:
    # Python's None stands in for upstream's undefined state.
    handle = await setup.registry.register(
        ProjectionDefinition(
            key="test/bound-undefined",
            state_schema=TypeAdapter(int | None),
            state_version=1,
            init=lambda _header, _inherited: None,
            apply=lambda state, event: (
                event["data"]["turn"] if event["type"] == "turn/start" else state
            ),
        ),
        ctx=setup.owner.ctx,
    )
    read = handle.reader_for(setup.session)
    assert read() is None
    setup.session.append("turn/start", {"turn": 1})
    assert read() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("release", ["handle", "fiber"])
@pytest.mark.upstream(
    f"{SPEC}::{READERS} > reads new events after %s disposal without rejoining snapshots or "
    "notifications"
)
async def test_reads_after_disposal(setup: Setup, release: Literal["handle", "fiber"]) -> None:
    registry, owner, session = setup.registry, setup.owner, setup.session
    apply = Mock(side_effect=DEFINITION.apply)
    handle = await registry.register(replace(DEFINITION, apply=apply), ctx=owner.ctx)
    read = handle.reader_for(session)
    changed = Mock()
    registry.on_changed(changed)
    assert handle.state_of(session) == 0
    session.append("turn/start", {"turn": 1})
    assert handle.state_of(session) == 1
    changed.assert_called_once()
    if release == "handle":
        await handle.dispose()
    else:
        await owner.dispose()
    session.append("turn/start", {"turn": 2})
    assert apply.call_count == 1
    assert handle.state_of(session) == 2
    assert read() == 2
    assert apply.call_count == 2
    changed.assert_called_once()
    assert registry.snapshot(session).values == {}
    assert registry.checkpoint(session) == {}
    later = setup.store.create()
    later.append("turn/start", {"turn": 1})
    assert handle.state_of(later) == 1


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{READERS} > keeps the original unit when the same key is registered with a new "
    "version"
)
async def test_keeps_original_unit_across_versions(setup: Setup) -> None:
    registry, session = setup.registry, setup.session
    original = await registry.register(DEFINITION, ctx=setup.owner.ctx)
    assert original.state_of(session) == 0
    await original.dispose()
    replacement = await registry.register(
        replace(DEFINITION, state_version=2, apply=lambda state, _event: state + 10)
    )
    session.append("turn/start", {"turn": 1})
    assert original.state_of(session) == 1
    assert replacement.state_of(session) == 10
    assert registry.snapshot(session).values == {"test/reader": 10}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{READERS} > reads only the selected unit without materializing unrelated "
    "registrations"
)
async def test_reads_only_selected_unit(setup: Setup) -> None:
    unrelated_init = Mock(return_value=0)
    await setup.registry.register(
        ProjectionDefinition(
            key="test/unrelated-reader",
            state_schema=NUMBER,
            state_version=1,
            init=unrelated_init,
            apply=lambda state, _event: state,
        )
    )
    handle = await setup.registry.register(DEFINITION, ctx=setup.owner.ctx)
    assert handle.state_of(setup.session) == 0
    unrelated_init.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{READERS} > rejects initialization when the owner unloads before the handle is "
    "returned"
)
async def test_rejects_when_owner_unloads_before_handoff(setup: Setup) -> None:
    registration = setup.registry.register(DEFINITION, ctx=setup.owner.ctx)
    await setup.owner.dispose()
    # deepseek-cordis reports the inactive owner as CordisError("INACTIVE_EFFECT").
    with pytest.raises(CordisError):
        await registration
    assert setup.registry.checkpoint(setup.session) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("release", ["twice", "owner-unload"])
@pytest.mark.upstream(f"{SPEC}::{READERS} > releases one reference once under %s disposal")
async def test_releases_one_reference_once(
    setup: Setup, release: Literal["twice", "owner-unload"]
) -> None:
    registry, owner, session = setup.registry, setup.owner, setup.session
    retained = await registry.register(DEFINITION)
    handle = await registry.register(DEFINITION, ctx=owner.ctx)
    if release == "twice":
        await handle.dispose()
        await handle.dispose()
    else:
        await asyncio.gather(owner.dispose(), handle.dispose())
        await handle.dispose()
    session.append("turn/start", {"turn": 1})
    assert registry.snapshot(session).values == {"test/reader": 1}
    assert handle.state_of(session) == 1
    await retained.dispose()
    assert registry.snapshot(session).values == {}
