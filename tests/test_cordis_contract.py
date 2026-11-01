"""The deepseek-cordis behaviour the harness relies on, checked against the installed version.

docs/cordis-alignment.md lists where upstream packages use each item. The core-facing
gaps closed in deepseek-cordis 0.2.0 are ordinary regression tests here; advanced
Loader composition remains outside the core compatibility profile.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator

import pytest
from deepseek_cordis import Context, CordisError, Fiber, FiberState, Service, current_context
from deepseek_cordis.loader import Loader, PluginEntry

from deskagents._cordis import emit_contained

pytestmark = pytest.mark.asyncio


def state(fiber: Fiber) -> FiberState:
    # A call keeps mypy from narrowing fiber.state across awaits.
    return fiber.state


async def test_waterfall_passes_arguments_then_next() -> None:
    root = Context()
    seen: list[object] = []

    def middleware(value: object, next_: Callable[[], object]) -> object:
        seen.append(value)
        return ("wrapped", next_())

    root.on("probe/waterfall", middleware)
    assert root.waterfall("probe/waterfall", 1, next_=lambda: "inner") == ("wrapped", "inner")
    assert seen == [1]
    await root.fiber.dispose()


async def test_generator_effect_cleans_up_on_dispose() -> None:
    root = Context()
    log: list[str] = []

    def setup() -> Iterator[Callable[[], None]]:
        log.append("setup")
        yield lambda: log.append("cleanup")

    root.effect(setup, "probe")
    assert log == ["setup"]
    await root.fiber.dispose()
    assert log == ["setup", "cleanup"]


async def test_inject_waits_for_a_service_and_unloads_with_it() -> None:
    root = Context()

    class Database(Service):
        name = "database"

    async def consumer(ctx: Context, config: object) -> None:
        assert isinstance(ctx.get("database"), Database)

    fiber = root.inject(["database"], consumer)
    await asyncio.sleep(0)
    assert state(fiber) is FiberState.PENDING
    provider = await root.plugin(Database)
    await fiber.wait()
    assert state(fiber) is FiberState.ACTIVE
    await provider.dispose()
    await asyncio.sleep(0)
    assert state(fiber) is FiberState.PENDING
    await root.fiber.dispose()


async def test_listener_options_prepend_and_global() -> None:
    root = Context()
    order: list[str] = []
    root.on("probe/order", lambda: order.append("second"))
    root.on("probe/order", lambda: order.append("first"), prepend=True, global_=True)
    root.emit("probe/order")
    assert order == ["first", "second"]
    await root.fiber.dispose()


async def test_fiber_states_match_upstream() -> None:
    names = {state.name for state in FiberState}
    assert {"PENDING", "LOADING", "ACTIVE", "FAILED", "UNLOADING", "DISPOSED"} <= names


async def test_events_dispatch_returns_the_uninvoked_listener_snapshot() -> None:
    root = Context()
    seen: list[int] = []
    root.on("probe/dispatch", seen.append)

    callbacks = root.events.dispatch("emit", "probe/dispatch", (1,))
    assert seen == []
    for callback in callbacks:
        callback(1)
    assert seen == [1]
    await root.fiber.dispose()


async def test_fiber_assert_active_rejects_a_disposed_fiber() -> None:
    root = Context()
    child = await root.plugin(lambda _ctx, _config: None)
    child.assert_active()
    await child.dispose()
    with pytest.raises(CordisError, match="INACTIVE_EFFECT"):
        child.assert_active()
    await root.fiber.dispose()


async def test_internal_status_reports_each_state_change() -> None:
    root = Context()
    changes: list[tuple[Fiber, FiberState]] = []
    root.on("internal/status", lambda fiber, old: changes.append((fiber, old)), global_=True)

    fiber = await root.plugin(lambda _ctx, _config: None)
    await fiber.dispose()
    assert [old for changed, old in changes if changed is fiber][:2] == [
        FiberState.PENDING,
        FiberState.LOADING,
    ]
    await root.fiber.dispose()


async def test_internal_plugin_publishes_creation_and_disposal() -> None:
    root = Context()
    fibers: list[Fiber] = []
    root.on("internal/plugin", fibers.append, global_=True)

    fiber = await root.plugin(lambda _ctx, _config: None)
    assert fibers == [fiber]
    await fiber.dispose()
    assert fibers == [fiber, fiber]
    await root.fiber.dispose()


async def test_loader_fibers_expose_their_entry() -> None:
    root = Context()
    entry = PluginEntry(id="probe", plugin=lambda _ctx, _config: None)
    loaded = await Loader(root).load([entry])
    assert loaded.fibers["probe"].entry is entry
    await root.fiber.dispose()


async def test_waterfall_next_is_single_use() -> None:
    root = Context()
    root.on("probe/twice", lambda next_: (next_(), next_()))
    with pytest.raises(RuntimeError, match="next\\(\\) called multiple times"):
        root.waterfall("probe/twice", next_=lambda: None)
    await root.fiber.dispose()


# Contained notifications (deskagents._cordis), the upstream pattern built on dispatch.


async def test_emit_contained_isolates_sync_and_async_listener_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = Context()
    seen: list[str] = []

    def broken(value: str) -> None:
        raise ValueError("sync failure")

    async def rejected(value: str) -> None:
        raise ValueError("async failure")

    root.on("probe/notify", broken)
    root.on("probe/notify", rejected)
    root.on("probe/notify", seen.append)
    with caplog.at_level(logging.WARNING, logger="deskagents._cordis"):
        emit_contained(root, "probe/notify", "value", describe="probe listener")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert seen == ["value"]
    assert [record.getMessage() for record in caplog.records] == [
        "probe listener threw: sync failure",
        "probe listener rejected: async failure",
    ]
    await root.fiber.dispose()


async def test_emit_contained_runs_listeners_in_the_emitter_scope() -> None:
    root = Context()
    child = root.extend()
    callers: list[Context | None] = []
    root.on("probe/scope", lambda: callers.append(current_context()))
    emit_contained(child, "probe/scope")
    assert callers == [child]
    await root.fiber.dispose()
