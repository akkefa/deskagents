"""Ported from packages/core/scope/tests/scope.spec.ts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

import pytest
from deepseek_cordis import Context

from deskagents.core.scope import (
    Scope,
    ScopeCarrier,
    bind_scope_parent,
    carrier_key_of,
    create_scope,
    is_scope_carrier,
    scope_chain_of,
    scope_of,
    scope_parent_of,
    scope_target,
)

SPEC = "packages/core/scope/tests/scope.spec.ts"


class Key:
    """Identity scope key (``object()`` cannot be weakly referenced)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Key({self.name!r})"


async def mint_scope(ctx: Context, key: object) -> Scope:
    """Mount a host plugin and mint a scope inside it."""
    minted: list[Scope] = []

    def host(inner: Context, config: object) -> None:
        minted.append(create_scope(inner, key))

    await ctx.plugin(host)
    return minted[0]


async def dispose(effect: Callable[[], object]) -> None:
    result = effect()
    if result is not None:
        await result  # type: ignore[misc]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::createScope > tags contexts and derived contexts, with the nearest tag winning"
)
async def test_tags_contexts_nearest_tag_wins() -> None:
    ctx = Context()
    outer_key, inner_key = Key("outer"), Key("inner")
    outer = await mint_scope(ctx, outer_key)
    inner = create_scope(outer.ctx, inner_key)

    assert scope_of(ctx) is None
    assert scope_of(outer.ctx) is outer_key
    assert scope_of(outer.ctx.extend({})) is outer_key
    assert scope_of(inner.ctx) is inner_key

    await inner.dispose()
    await outer.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::createScope > is usable synchronously before the backing fiber activates"
)
async def test_usable_synchronously() -> None:
    ctx = Context()
    events: list[str] = []
    minted: list[Scope] = []

    def host(inner: Context, config: object) -> None:
        scope = create_scope(inner, Key("sync"))
        scope.ctx.effect(lambda: lambda: events.append("disposed"))
        events.append("registered")
        minted.append(scope)

    await ctx.plugin(host)
    assert events == ["registered"]
    await minted[0].dispose()
    assert events == ["registered", "disposed"]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::createScope > shares quiescence across repeat and raw-disposer-first calls"
)
async def test_shares_quiescence() -> None:
    ctx = Context()
    scope = await mint_scope(ctx, Key("quiescence"))
    gate = asyncio.get_running_loop().create_future()
    finished = False

    async def cleanup() -> None:
        nonlocal finished
        await gate
        finished = True

    scope.ctx.effect(lambda: cleanup)

    raw = asyncio.ensure_future(scope.raw_dispose())
    public = asyncio.ensure_future(scope.dispose())
    await asyncio.sleep(0)
    assert finished is False
    gate.set_result(None)
    await asyncio.gather(raw, public, scope.dispose())
    assert finished is True


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::createScope > exposes the exact raw disposer for ordered composite teardown"
)
async def test_raw_disposer_in_composite_teardown() -> None:
    ctx = Context()
    order: list[str] = []
    effects: list[Callable[[], object]] = []

    def host(inner: Context, config: object) -> None:
        def setup() -> Iterator[Callable[[], object]]:
            yield lambda: order.append("outer")
            scope = create_scope(inner, Key("nested"))
            scope.ctx.effect(lambda: lambda: order.append("scope"))
            yield scope.raw_dispose
            yield lambda: order.append("inner")

        effects.append(inner.effect(setup))

    await ctx.plugin(host)
    await dispose(effects[0])
    assert order == ["inner", "scope", "outer"]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::scopeTarget > routes scoped listeners by key while untagged listeners remain global"
)
async def test_routes_scoped_listeners() -> None:
    ctx = Context()
    key_a, key_b = Key("A"), Key("B")
    scope_a = await mint_scope(ctx, key_a)
    scope_b = await mint_scope(ctx, key_b)
    heard: list[str] = []
    ctx.on("scope-test/ping", lambda value: heard.append(f"global:{value}"))
    scope_a.ctx.on("scope-test/ping", lambda value: heard.append(f"A:{value}"))
    scope_b.ctx.on("scope-test/ping", lambda value: heard.append(f"B:{value}"))

    ctx.emit("scope-test/ping", "a", filter_=scope_target(ctx, key_a))
    ctx.emit("scope-test/ping", "b", filter_=scope_target(ctx, key_b))
    ctx.emit("scope-test/ping", "none", filter_=scope_target(ctx, None))

    assert heard == ["global:a", "A:a", "global:b", "B:b", "global:none"]
    await asyncio.gather(scope_a.dispose(), scope_b.dispose())


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::scopeTarget > preserves a base Cordis filter and its receiver")
async def test_preserves_base_filter() -> None:
    ctx = Context()
    key = Key("A")
    scope = await mint_scope(ctx, key)
    heard: list[str] = []
    ctx.on("scope-test/ping", lambda value: heard.append(f"global:{value}"))
    scope.ctx.on("scope-test/ping", lambda value: heard.append(f"A:{value}"))
    receivers: list[object] = []

    class Base:
        def matches_scope(self, ctx: Context) -> bool:
            receivers.append(self)
            return False

    base = Base()
    ctx.emit("scope-test/ping", "vetoed", filter_=scope_target(base, key))
    assert heard == []
    assert receivers and all(receiver is base for receiver in receivers)
    await scope.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::scopeTarget > {{ global: true }} listeners retain Cordis global-listener semantics"
)
async def test_global_listeners() -> None:
    ctx = Context()
    scope = await mint_scope(ctx, Key("A"))
    heard: list[str] = []
    scope.ctx.on("scope-test/ping", heard.append, global_=True)
    ctx.emit("scope-test/ping", "foreign", filter_=scope_target(ctx, Key("other")))
    ctx.emit("scope-test/ping", "none", filter_=scope_target(ctx, None))
    assert heard == ["foreign", "none"]
    await scope.dispose()


@pytest.mark.upstream(
    f"{SPEC}::scopeTarget > uses an opaque branded carrier with a separately tracked key"
)
def test_opaque_carrier() -> None:
    key = Key("key")

    class Subject:
        value = 1

    subject = Subject()
    carrier = scope_target(subject, key)
    assert is_scope_carrier(carrier)
    assert carrier_key_of(carrier) is key
    assert not is_scope_carrier(subject)
    assert carrier_key_of(subject) is None
    assert not hasattr(carrier, "value")
    assert isinstance(carrier, ScopeCarrier)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::scope parent chain > links at mint, walks to the root, and rejects cycles"
)
async def test_parent_chain_links_and_rejects_cycles() -> None:
    ctx = Context()
    preset, agent = Key("preset"), Key("agent")
    create_scope(ctx, preset)
    create_scope(ctx, agent, parent=preset)

    assert scope_parent_of(agent) is preset
    assert scope_parent_of(preset) is None
    assert scope_chain_of(agent) == [agent, preset]
    assert scope_chain_of(None) == []
    with pytest.raises(ValueError, match="cycle"):
        bind_scope_parent(preset, agent)
    with pytest.raises(ValueError, match="cycle"):
        bind_scope_parent(preset, preset)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::scope parent chain > re-links only through the binding held by the original binder"
)
async def test_relinks_only_through_binding() -> None:
    ctx = Context()
    preset_a, preset_b, agent = Key("a"), Key("b"), Key("agent")
    create_scope(ctx, preset_a)
    create_scope(ctx, preset_b)
    binding = bind_scope_parent(agent, preset_a)
    create_scope(ctx, agent)

    with pytest.raises(ValueError, match="already bound"):
        bind_scope_parent(agent, preset_b)
    binding.rebind(preset_b)
    assert scope_chain_of(agent) == [agent, preset_b]

    child = Key("child")
    bind_scope_parent(child, agent)
    with pytest.raises(ValueError, match="cycle"):
        binding.rebind(child)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::scope parent chain > admits an ancestor-tagged listener for a descendant "
    "dispatch, never the reverse"
)
async def test_ancestor_listener_admission() -> None:
    ctx = Context()
    preset, agent, other = Key("preset"), Key("agent"), Key("other-preset")
    preset_scope = create_scope(ctx, preset)
    agent_scope = create_scope(ctx, agent, parent=preset)
    other_scope = create_scope(ctx, other)

    seen: list[str] = []
    ctx.on("probe/event", lambda: seen.append("untagged"))
    preset_scope.ctx.on("probe/event", lambda: seen.append("preset"))
    agent_scope.ctx.on("probe/event", lambda: seen.append("agent"))
    other_scope.ctx.on("probe/event", lambda: seen.append("other"))

    ctx.emit("probe/event", filter_=scope_target(object(), agent))
    assert sorted(seen) == ["agent", "preset", "untagged"]

    seen.clear()
    ctx.emit("probe/event", filter_=scope_target(object(), preset))
    assert sorted(seen) == ["preset", "untagged"]
    await ctx.fiber.dispose()


def test_scope_keys_must_be_weakly_referenceable() -> None:
    with pytest.raises(TypeError, match="weak references"):
        bind_scope_parent(object(), Key("parent"))
