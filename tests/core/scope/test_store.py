"""Ported from packages/core/scope/tests/store.spec.ts."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from deepseek_cordis import Context

from deskagents.core.scope import (
    AnonymousEntries,
    NamedEntries,
    Scope,
    ScopedLayers,
    ScopeKey,
    create_scope,
)

SPEC = "packages/core/scope/tests/store.spec.ts"


class Key:
    """Identity scope key (``object()`` cannot be weakly referenced)."""


class TestLayer:
    __test__ = False

    def __init__(self, scope: ScopeKey | None) -> None:
        label = "global" if scope is None else "scoped"
        self.named: NamedEntries[int] = NamedEntries(
            lambda name: ValueError(f"{label} duplicate: {name}")
        )
        self.anonymous: AnonymousEntries[str] = AnonymousEntries()

    def is_empty(self) -> bool:
        return self.named.is_empty() and self.anonymous.is_empty()


async def mint_scope(ctx: Context, key: ScopeKey) -> Scope:
    minted: list[Scope] = []

    def host(inner: Context, config: object) -> None:
        minted.append(create_scope(inner, key))

    await ctx.plugin(host)
    return minted[0]


def noop() -> None:
    return None


@pytest.mark.upstream(
    f"{SPEC}::NamedEntries > owns duplicate diagnostics, lookup, insertion order, live "
    "iteration, and exact idempotent undo"
)
def test_named_entries_contract() -> None:
    duplicate = ValueError("caller duplicate")
    asked: list[str] = []

    def duplicate_error(name: str) -> BaseException:
        asked.append(name)
        return duplicate

    entries: NamedEntries[int] = NamedEntries(duplicate_error)
    undo_a = entries.insert("a", 1)
    values = entries.values()
    assert next(values) == 1
    undo_b = entries.insert("b", 2)

    assert list(values) == [2]
    assert list(entries.keys()) == ["a", "b"]
    assert list(entries.entries()) == [("a", 1), ("b", 2)]
    assert entries.get("a") == 1
    assert entries.get("missing") is None
    assert entries.has("b")
    assert not entries.has("missing")
    assert not entries.is_empty()
    with pytest.raises(ValueError) as raised:
        entries.insert("a", 3)
    assert raised.value is duplicate
    assert asked == ["a"]

    undo_a()
    entries.insert("a", 3)
    undo_a()
    assert entries.get("a") == 3
    undo_b()
    assert list(entries.entries()) == [("a", 3)]


@pytest.mark.upstream(
    f"{SPEC}::NamedEntries > starts a fresh iterator generation after the table drains"
)
def test_named_entries_fresh_generation() -> None:
    entries: NamedEntries[int] = NamedEntries(lambda name: ValueError(f"duplicate: {name}"))
    undo = entries.insert("first", 1)
    values = entries.values()

    assert next(values) == 1
    undo()
    entries.insert("replacement", 2)

    assert next(values, None) is None
    assert list(entries.values()) == [2]


@pytest.mark.upstream(
    f"{SPEC}::AnonymousEntries > owns equal values independently with live insertion-ordered "
    "iteration and idempotent undo"
)
def test_anonymous_entries_contract() -> None:
    entries: AnonymousEntries[object] = AnonymousEntries()
    value = object()
    undo_first = entries.append(value)
    values = entries.values()
    assert next(values) is value
    undo_second = entries.append(value)

    assert list(values) == [value]
    assert list(entries.values()) == [value, value]
    undo_first()
    undo_first()
    assert list(entries.values()) == [value]
    undo_second()
    assert entries.is_empty()


@pytest.mark.upstream(
    f"{SPEC}::AnonymousEntries > starts a fresh iterator generation after the table drains"
)
def test_anonymous_entries_fresh_generation() -> None:
    entries: AnonymousEntries[int] = AnonymousEntries()
    undo = entries.append(1)
    values = entries.values()

    assert next(values) == 1
    undo()
    entries.append(2)

    assert next(values, None) is None
    assert list(entries.values()) == [2]


@pytest.mark.upstream(
    f"{SPEC}::ScopedLayers > constructs global state eagerly while reads stay non-creating "
    "and merge named shadows in order"
)
def test_scoped_layers_reads_do_not_create() -> None:
    created: list[ScopeKey | None] = []

    def create(scope: ScopeKey | None) -> TestLayer:
        created.append(scope)
        return TestLayer(scope)

    layers = ScopedLayers(create, noop)
    key = Key()
    layers.global_.named.insert("a", 1)
    layers.global_.named.insert("shared", 2)

    assert created == [None]
    assert layers.peek(None) is None
    assert layers.peek(key) is None
    assert list(layers.merge(key, lambda layer: layer.named).items()) == [("a", 1), ("shared", 2)]
    assert created == [None]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::ScopedLayers > uses the same scoped context for lazy visibility and ownership, "
    "and reclaims only an empty aggregate"
)
async def test_scoped_layers_lazy_and_reclaim() -> None:
    ctx = Context()
    key = Key()
    scope = await mint_scope(ctx, key)
    changes: list[None] = []
    created: list[ScopeKey | None] = []

    def create(selected: ScopeKey | None) -> TestLayer:
        created.append(selected)
        return TestLayer(selected)

    layers = ScopedLayers(create, lambda: changes.append(None))
    layers.global_.named.insert("a", 1)
    layers.global_.named.insert("shared", 1)
    remove_named = layers.effect(
        scope.ctx, lambda layer: layer.named.insert("shared", 2), label="test.named", notify=False
    )
    remove_tail = layers.effect(
        scope.ctx, lambda layer: layer.named.insert("c", 3), label="test.tail", notify=False
    )
    remove_anonymous = layers.effect(
        scope.ctx,
        lambda layer: layer.anonymous.append("kept"),
        label="test.anonymous",
        notify=False,
    )

    assert created == [None, key]
    merged = list(layers.merge(key, lambda layer: layer.named).items())
    assert merged == [("a", 1), ("shared", 2), ("c", 3)]
    assert changes == []
    remove_named()
    assert layers.peek(key) is not None
    merged = list(layers.merge(key, lambda layer: layer.named).items())
    assert merged == [("a", 1), ("shared", 1), ("c", 3)]
    remove_tail()
    assert layers.peek(key) is not None
    remove_anonymous()
    assert layers.peek(key) is None
    await scope.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::ScopedLayers > runs action, notification, undo, and disposal notification in "
    "order with Cordis idempotence and labels"
)
async def test_scoped_layers_effect_order() -> None:
    ctx = Context()
    events: list[str] = []
    layers = ScopedLayers(TestLayer, lambda: events.append("notify"))

    def action(layer: TestLayer) -> Callable[[], None]:
        events.append("action")
        undo = layer.named.insert("x", 1)

        def remove() -> None:
            events.append("undo")
            undo()

        return remove

    dispose = layers.effect(ctx, action, label="store.order")

    assert events == ["action", "notify"]
    assert "store.order" in [effect.label for effect in ctx.fiber.get_effects()]
    dispose()
    dispose()
    assert events == ["action", "notify", "undo", "notify"]
    assert layers.global_.is_empty()
    await ctx.fiber.dispose()


@pytest.mark.upstream(f"{SPEC}::ScopedLayers > returns the exact context effect disposer")
def test_scoped_layers_returns_exact_disposer() -> None:
    raw_dispose = object()
    calls: list[tuple[object, str]] = []

    class FakeContext:
        metadata: dict[str, object] = {}

        def effect(self, setup: object, label: str) -> object:
            calls.append((setup, label))
            return raw_dispose

    actions: list[object] = []
    layers = ScopedLayers(TestLayer, noop)
    returned = layers.effect(
        FakeContext(),  # type: ignore[arg-type]
        lambda layer: actions.append(layer) or noop,  # type: ignore[func-returns-value]
        label="store.identity",
        notify=False,
    )

    assert returned is raw_dispose
    assert len(calls) == 1 and callable(calls[0][0]) and calls[0][1] == "store.identity"
    assert actions == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::ScopedLayers > cleans up failed factories and empty failed actions without "
    "discarding an existing layer"
)
async def test_scoped_layers_failed_factories_and_actions() -> None:
    ctx = Context()
    key = Key()
    scope = await mint_scope(ctx, key)
    fail_factory = True

    def create(selected: ScopeKey | None) -> TestLayer:
        if selected is not None and fail_factory:
            raise RuntimeError("factory failed")
        return TestLayer(selected)

    def fail(message: str) -> Callable[[TestLayer], Callable[[], None]]:
        def action(layer: TestLayer) -> Callable[[], None]:
            raise RuntimeError(message)

        return action

    layers = ScopedLayers(create, noop)
    with pytest.raises(RuntimeError, match="factory failed"):
        layers.effect(
            scope.ctx, lambda layer: layer.named.insert("never", 1), label="f", notify=False
        )
    assert layers.peek(key) is None

    fail_factory = False
    with pytest.raises(RuntimeError, match="action failed"):
        layers.effect(scope.ctx, fail("action failed"), label="store.action", notify=False)
    assert layers.peek(key) is None

    dispose = layers.effect(
        scope.ctx, lambda layer: layer.named.insert("kept", 1), label="store.kept", notify=False
    )
    with pytest.raises(RuntimeError, match="second action failed"):
        layers.effect(scope.ctx, fail("second action failed"), label="s", notify=False)
    existing = layers.peek(key)
    assert existing is not None and existing.named.get("kept") == 1
    dispose()
    await scope.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::ScopedLayers > rolls back a scoped insertion when notification throws"
)
async def test_scoped_layers_rollback_on_notification_error() -> None:
    ctx = Context()
    key = Key()
    scope = await mint_scope(ctx, key)
    events: list[str] = []
    notifications = 0

    def on_change() -> None:
        nonlocal notifications
        events.append("notify")
        notifications += 1
        if notifications == 1:
            raise RuntimeError("change failed")

    def action(layer: TestLayer) -> Callable[[], None]:
        undo = layer.named.insert("rollback", 1)

        def remove() -> None:
            events.append("undo")
            undo()

        return remove

    layers = ScopedLayers(TestLayer, on_change)
    with pytest.raises(RuntimeError, match="change failed"):
        layers.effect(scope.ctx, action, label="store.rollback")

    assert events == ["notify", "undo", "notify"]
    assert layers.peek(key) is None
    await scope.dispose()


def test_live_iteration_survives_heavy_churn_and_compaction() -> None:
    entries: NamedEntries[int] = NamedEntries(lambda name: ValueError(name))
    keep = entries.insert("keep", 0)
    for round_ in range(200):
        entries.insert(f"tmp{round_}", round_)()
    reader = entries.values()
    assert next(reader) == 0
    entries.insert("late", 1)
    assert list(reader) == [1]
    keep()
    assert list(entries.keys()) == ["late"]
