"""Ported from packages/session/session-projection/tests/summary.spec.ts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Annotated, Any

import pytest
from deepseek_cordis import Context
from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from deskagents.core.session import (
    CreateSessionMeta,
    CreateSessionOptions,
    SessionEvent,
    SessionId,
    SessionSeq,
    SessionStore,
)
from deskagents.session.projection import (
    ProjectionCheckpoint,
    ProjectionDefinition,
    ProjectionWire,
    SessionProjectionRegistry,
)

SPEC = "packages/session/session-projection/tests/summary.spec.ts"
SUMMARY = "session projection summary keys"


class Text(TypedDict):
    text: str


KEPT: TypeAdapter[Text | None] = TypeAdapter(Text | None)
COUNT: TypeAdapter[int] = TypeAdapter(Annotated[int, Field(strict=True, ge=0)])
LABELS: TypeAdapter[list[str] | None] = TypeAdapter(list[str] | None)
WORKSPACE = CreateSessionOptions(meta=CreateSessionMeta(cwd="/workspace"))


def kept_unit() -> ProjectionDefinition[Any]:
    def apply(state: Any, event: SessionEvent) -> Any:
        if event["type"] == "turn/start":
            return {"text": f"turn-{event['data']['turn']}"}
        return state

    return ProjectionDefinition(
        key="test/summary-kept",
        state_schema=KEPT,
        init=lambda _header, _inherited: None,
        apply=apply,
        wire=ProjectionWire(view_schema=KEPT, view=lambda state: state),
        state_version=1,
    )


def excluded_unit() -> ProjectionDefinition[int]:
    return ProjectionDefinition(
        key="test/summary-excluded",
        state_schema=COUNT,
        init=lambda _header, _inherited: 0,
        apply=lambda state, _event: state + 1,
        wire=ProjectionWire(view_schema=COUNT, view=lambda state: state, summary=False),
        state_version=1,
    )


def excluded_nullable_unit() -> ProjectionDefinition[Any]:
    def apply(state: Any, event: SessionEvent) -> Any:
        if event["type"] == "turn/start":
            return [f"turn-{event['data']['turn']}"]
        return state

    return ProjectionDefinition(
        key="test/summary-excluded-nullable",
        state_schema=LABELS,
        init=lambda _header, _inherited: None,
        apply=apply,
        wire=ProjectionWire(view_schema=LABELS, view=lambda state: state, summary=False),
        state_version=1,
    )


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > derives the summary set from the registered units and their marks"
)
async def test_derives_summary_set(registry: SessionProjectionRegistry) -> None:
    await registry.register(kept_unit())
    await registry.register(excluded_unit())
    await registry.register(excluded_nullable_unit())
    keys = registry.summary_keys()
    assert list(keys) == ["test/summary-kept"]
    assert "test/summary-excluded" not in keys
    # The set is stable while the key table is unchanged.
    assert registry.summary_keys() is keys


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > refuses a second registration of one key with a different summary "
    "membership"
)
async def test_refuses_different_summary_membership(registry: SessionProjectionRegistry) -> None:
    await registry.register(kept_unit())
    wire = kept_unit().wire
    assert wire is not None
    with pytest.raises(
        ValueError, match=r"test/summary-kept.*already registered with summary=true"
    ):
        await registry.register(replace(kept_unit(), wire=replace(wire, summary=False)))


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > rebuilds the summary set only when a registration or its disposal "
    "changes the key table"
)
async def test_rebuilds_summary_set_on_key_table_change(
    ctx: Context, registry: SessionProjectionRegistry
) -> None:
    empty = registry.summary_keys()
    assert list(empty) == []

    async def host(host_ctx: Context, config: object) -> None:
        projections = host_ctx.get("sessionProjections")
        assert isinstance(projections, SessionProjectionRegistry)
        await projections.register(kept_unit())

    fiber = await ctx.plugin(SimpleNamespace(inject=["sessionProjections"], apply=host))
    registered = registry.summary_keys()
    assert registered is not empty
    assert list(registered) == ["test/summary-kept"]
    assert registry.summary_keys() is registered
    await fiber.dispose()
    disposed = registry.summary_keys()
    assert disposed is not registered
    assert list(disposed) == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > serves a summary-shaped block through the key-taking checkpoint face"
)
async def test_summary_block_through_checkpoint_face(registry: SessionProjectionRegistry) -> None:
    await registry.register(kept_unit())
    await registry.register(excluded_unit())
    await registry.register(excluded_nullable_unit())
    checkpoint: ProjectionCheckpoint = {
        "test/summary-kept": {"ver": 1, "seq": SessionSeq(3), "val": {"text": "kept"}},
        "test/summary-excluded": {"ver": 1, "seq": SessionSeq(4), "val": 7},
        "test/summary-excluded-nullable": {"ver": 1, "seq": SessionSeq(5), "val": ["a"]},
    }

    assert registry.view_checkpoint(checkpoint, registry.summary_keys()) == {
        "test/summary-kept": {"text": "kept"},
    }
    # Without the key subset the complete face still serves the marked keys.
    assert registry.view_checkpoint(checkpoint) == {
        "test/summary-kept": {"text": "kept"},
        "test/summary-excluded": 7,
        "test/summary-excluded-nullable": ["a"],
    }


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > serves cached summary cells through the key-taking cached face"
)
async def test_cached_summary_cells(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    await registry.register(kept_unit())
    await registry.register(excluded_unit())
    session = store.create(SessionId("summary-cached"), WORKSPACE)
    session.append("turn/start", {"turn": 1})

    # The derived set is passed as a set: no key-list copy inside the read.
    cached = registry.cached_snapshot(session, registry.summary_keys())
    assert cached is not None
    assert cached.values == {"test/summary-kept": {"text": "turn-1"}}
    complete = registry.cached_snapshot(session)
    assert complete is not None
    assert complete.values == {
        "test/summary-kept": {"text": "turn-1"},
        "test/summary-excluded": 1,
    }


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > reports an empty summary set while the complete face still serves only "
    "marked units"
)
async def test_empty_summary_set(registry: SessionProjectionRegistry, store: SessionStore) -> None:
    await registry.register(excluded_unit())
    await registry.register(excluded_nullable_unit())
    session = store.create(SessionId("summary-empty"), WORKSPACE)
    session.append("turn/start", {"turn": 1})

    assert list(registry.summary_keys()) == []
    assert registry.cached_snapshot(session, []) is None
    complete = registry.cached_snapshot(session)
    assert complete is not None
    assert complete.values == {
        "test/summary-excluded": 1,
        "test/summary-excluded-nullable": ["turn-1"],
    }


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > keeps the marked nullable entry nullable through the mark"
)
async def test_marked_nullable_entry(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    await registry.register(excluded_nullable_unit())
    before = store.create(SessionId("summary-nullable"), WORKSPACE)
    values = registry.snapshot(before).values
    assert "test/summary-excluded-nullable" in values
    assert values["test/summary-excluded-nullable"] is None


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{SUMMARY} > drops an unmaterialized cell and a version-mismatched checkpoint row"
)
async def test_drops_unmaterialized_cell_and_mismatched_row(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    await registry.register(kept_unit())
    session = store.create(SessionId("summary-unmaterialized"), WORKSPACE)
    cached = registry.cached_snapshot(session, registry.summary_keys())
    assert cached is not None
    assert cached.values == {"test/summary-kept": None}
    checkpoint: ProjectionCheckpoint = {
        "test/summary-kept": {"ver": 2, "seq": SessionSeq(1), "val": {"text": "stale"}},
    }
    assert registry.view_checkpoint(checkpoint, registry.summary_keys()) == {}
