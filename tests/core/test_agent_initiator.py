from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import AgentRegistry

SPEC = "packages/core/agent/tests/agent-initiator.spec.ts::AgentRegistry initiator scope > "


async def registry() -> tuple[Context, Any, AgentRegistry]:
    ctx = Context()
    fiber = await ctx.plugin(AgentRegistry)
    agents = ctx.get("agents")
    assert isinstance(agents, AgentRegistry)
    return ctx, fiber, agents


def agent(id_: str) -> Any:
    return SimpleNamespace(id=id_, session=SimpleNamespace(id=id_))


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}reports an absent initiator and requires an active boundary")
@pytest.mark.upstream(f"{SPEC}restores nested and explicitly cleared boundaries")
async def test_initiator_boundaries_are_nested_and_clearable() -> None:
    ctx, _, agents = await registry()
    parent, child = agent("parent"), agent("child")
    assert agents.current_initiator() is None
    with pytest.raises(RuntimeError, match="no initiating agent"):
        agents.require_initiator()

    def nested() -> str:
        assert agents.require_initiator() is parent
        agents.with_initiator(child, lambda: agents.require_initiator() is child)
        assert agents.require_initiator() is parent
        agents.without_initiator(lambda: agents.current_initiator() is None)
        assert agents.require_initiator() is parent
        return "result"

    assert agents.with_initiator(parent, nested) == "result"
    assert agents.current_initiator() is None
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}isolates overlapping initiators")
async def test_overlapping_initiators_keep_their_own_context() -> None:
    ctx, _, agents = await registry()
    first, second = agent("first"), agent("second")
    release = asyncio.Event()
    started = [asyncio.Event(), asyncio.Event()]

    async def operation(subject: Any, ready: asyncio.Event) -> None:
        assert agents.require_initiator() is subject
        ready.set()
        await release.wait()
        assert agents.require_initiator() is subject

    pending = [
        agents.with_initiator(first, lambda: operation(first, started[0])),
        agents.with_initiator(second, lambda: operation(second, started[1])),
    ]
    await asyncio.gather(*(event.wait() for event in started))
    assert agents.current_initiator() is None
    release.set()
    await asyncio.gather(*pending)
    assert agents.current_initiator() is None
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}stops new boundaries, drains active Promises, and invalidates retained references"
)
async def test_disposal_drains_active_initiators_and_invalidates_the_service() -> None:
    ctx, fiber, agents = await registry()
    subject = agent("draining")
    started = asyncio.Event()
    release = asyncio.Event()

    async def operation() -> None:
        assert agents.require_initiator() is subject
        started.set()
        await release.wait()
        assert agents.require_initiator() is subject

    pending = agents.with_initiator(subject, operation)
    await started.wait()
    disposal = asyncio.ensure_future(fiber.dispose())
    await asyncio.sleep(0)
    assert not disposal.done()
    with pytest.raises(RuntimeError, match="initiator scope is disposed"):
        agents.with_initiator(subject, lambda: None)
    release.set()
    await pending
    await disposal
    with pytest.raises(RuntimeError, match="initiator scope is disposed"):
        agents.current_initiator()
    with pytest.raises(RuntimeError, match="initiator scope is disposed"):
        agents.require_initiator()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}does not self-deadlock when a boundary returns service disposal")
async def test_initiator_can_dispose_its_own_service_without_deadlock() -> None:
    ctx, fiber, agents = await registry()
    subject = agent("self-dispose")

    async def dispose() -> None:
        assert agents.require_initiator() is subject
        await fiber.dispose()

    pending = agents.with_initiator(subject, dispose)
    await asyncio.wait_for(pending, 1)
    with pytest.raises(RuntimeError, match="initiator scope is disposed"):
        agents.current_initiator()
    await ctx.fiber.dispose()
