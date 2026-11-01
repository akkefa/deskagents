"""Ported from packages/core/agent-loop/tests/loop.spec.ts: ``Agent.run_maintenance``."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import AgentLoop, AgentRegistry, SystemPrompt, ToolsService
from deskagents.core.session import SessionStore
from deskagents.core.tools import NativeToolExecutor
from deskagents.llm import LlmRuntime, create_user_message
from deskagents.testing import ScriptedAdapter
from deskagents.util.abort import AbortSignal

LOOP = "packages/core/agent-loop/tests/loop.spec.ts::agent loop > "


async def harness(
    scripts: Sequence[Sequence[Mapping[str, Any]]], session_id: str
) -> tuple[Context, ScriptedAdapter, Any]:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    await ctx.plugin(SystemPrompt, {"includeHarnessIdentity": False, "personaPrefix": "P"})
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(AgentLoop)
    adapter = ScriptedAdapter(scripts)
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    llm.register_adapter(["mock"], adapter)
    agents = ctx.get("agents")
    assert isinstance(agents, AgentRegistry)
    handle = await agents.create(
        {"sessionId": session_id, "agentOptions": {"provider": "mock", "model": "mock"}}
    )
    return ctx, adapter, handle


def text_response(text: str) -> list[Mapping[str, Any]]:
    return [
        {"type": "text-delta", "index": 0, "text": text},
        {"type": "finish", "reason": {"kind": "stop"}},
    ]


def user(text: str) -> Mapping[str, Any]:
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})


def user_texts(agent: Any) -> list[str]:
    return [
        event["data"]["content"][0]["text"]
        for event in agent.session.snapshot_events()
        if event["type"] == "user/message" and event["data"]["source"]["kind"] == "user"
    ]


async def blocked_job(started: asyncio.Event, finish: asyncio.Event) -> None:
    started.set()
    await finish.wait()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{LOOP}rejects concurrent maintenance while one job owns the Agent")
@pytest.mark.upstream(f"{LOOP}rejects a second maintenance task while one is active")
async def test_a_second_maintenance_task_is_rejected() -> None:
    ctx, _, handle = await harness([text_response("unused")], "concurrent-maintenance")
    agent = handle.agent
    started, finish = asyncio.Event(), asyncio.Event()
    active = agent.run_maintenance(lambda signal: blocked_job(started, finish))
    await started.wait()
    assert agent.status == "idle"
    with pytest.raises(RuntimeError, match="already has active work"):
        agent.run_maintenance(lambda signal: blocked_job(started, finish))
    finish.set()
    await active
    await agent.when_idle()
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{LOOP}cancels queued wakeup work together with an active maintenance task")
async def test_cancel_drops_wakeups_queued_behind_maintenance() -> None:
    ctx, adapter, handle = await harness([text_response("park reply")], "cancel-maintenance")
    agent = handle.agent
    started = asyncio.Event()

    async def job(signal: AbortSignal) -> None:
        started.set()
        reason = await signal.wait()
        raise RuntimeError(f"maintenance aborted: {reason}")

    maintenance = agent.run_maintenance(job)
    await started.wait()
    agent.followup(user("discard this wakeup"))
    agent.cancel({"kind": "user"})
    agent.followup(user("park after cancellation"))
    with pytest.raises(RuntimeError, match="maintenance aborted"):
        await maintenance
    await agent.when_idle()

    assert user_texts(agent) == ["park after cancellation"]
    assert len(agent.inbox.next_turn) == 0
    assert len(adapter.requests) == 1
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{LOOP}replays a wake latched behind maintenance at convergence")
async def test_a_wake_behind_maintenance_runs_when_it_settles() -> None:
    ctx, adapter, handle = await harness([text_response("wake reply")], "maintenance-wake")
    agent = handle.agent
    started, finish = asyncio.Event(), asyncio.Event()
    maintenance = agent.run_maintenance(lambda signal: blocked_job(started, finish))
    await started.wait()
    agent.followup(user("wake behind maintenance"))
    await asyncio.sleep(0)
    assert adapter.requests == []
    finish.set()
    await maintenance
    await agent.when_idle()

    assert user_texts(agent) == ["wake behind maintenance"]
    assert len(adapter.requests) == 1
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{LOOP}suppresses the replay when a latched maintenance wake is removed")
async def test_a_removed_wake_is_not_replayed() -> None:
    ctx, adapter, handle = await harness([], "maintenance-wake-removed")
    agent = handle.agent
    started, finish = asyncio.Event(), asyncio.Event()
    maintenance = agent.run_maintenance(lambda signal: blocked_job(started, finish))
    await started.wait()
    wake = user("removed wake")
    agent.followup(wake)
    assert agent.inbox.remove(wake["id"])
    finish.set()
    await maintenance
    await agent.when_idle()

    assert user_texts(agent) == []
    assert adapter.requests == []
    turns = [e for e in agent.session.snapshot_events() if e["type"] == "turn/start"]
    assert turns == []
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_maintenance_is_refused_while_a_turn_runs() -> None:
    ctx, _, handle = await harness([text_response("ok")], "maintenance-refused")
    agent = handle.agent
    refused: list[str] = []

    async def job(signal: AbortSignal) -> None:
        return None

    def on_status(payload: Mapping[str, Any]) -> None:
        if payload["agent"] is agent and payload["status"] == "running":
            try:
                agent.run_maintenance(job)
            except RuntimeError as error:
                refused.append(str(error))

    ctx.on("agent/status", on_status)
    agent.followup(user("go"))
    await agent.when_idle()
    assert refused == [f'agent "{agent.id}" already has active work']
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_disposal_aborts_maintenance() -> None:
    ctx, _, handle = await harness([], "maintenance-disposal")
    agent = handle.agent
    started = asyncio.Event()
    reasons: list[object] = []

    async def job(signal: AbortSignal) -> str:
        started.set()
        reasons.append(await signal.wait())
        return "stopped"

    maintenance = agent.run_maintenance(job)
    await started.wait()
    await handle.dispose()
    assert await maintenance == "stopped"
    assert reasons == [{"kind": "disposed"}]
    await ctx.fiber.dispose()
