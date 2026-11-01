from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import SystemPrompt, ToolsService, define_tool
from deskagents.core.session import Session, SessionStore
from deskagents.core.tools import NativeToolExecutor
from deskagents.session import checkpoint_policy


async def setup() -> tuple[Context, SessionStore, ToolsService]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    await ctx.plugin(checkpoint_policy)
    sessions = ctx.get("sessions")
    assert isinstance(sessions, SessionStore)
    return ctx, sessions, tools


def register_write(tools: ToolsService, order: list[str]) -> None:
    async def execute(args: Mapping[str, Any], execution: object) -> object:
        order.append("tool")
        return None

    tools.register(
        define_tool(
            name="write",
            description="side effect",
            parameters={},
            output={"schema": {"type": "null"}, "render": lambda args, value: []},
            execute=execute,
        )
    )


async def run_write(tools: ToolsService, session: Session) -> Mapping[str, Any]:
    signal = type("Signal", (), {"aborted": False})()
    batch = await tools.execute(
        {
            "calls": [{"id": "write-1", "name": "write", "arguments": "{}"}],
            "signal": signal,
            "agent": SimpleNamespace(session=session),
        }
    )
    result: Mapping[str, Any] = batch["results"][0]["result"]
    return result


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/session/session-checkpoint-policy/tests/session-checkpoint-policy.spec.ts::"
    "session-checkpoint-policy tool and step boundaries > "
    "awaits the checkpoint before a top-level tool body"
)
async def test_checkpoint_precedes_a_top_level_tool_body() -> None:
    ctx, sessions, tools = await setup()
    session = sessions.create("tool-checkpoint")
    session.append("turn/start", {"turn": 1})
    gate = asyncio.Event()
    entered = asyncio.Event()
    order: list[str] = []

    async def flush(*args: object) -> None:
        order.append("flush:start")
        entered.set()
        await gate.wait()
        order.append("flush:end")

    ctx.on("session/flush", flush)
    register_write(tools, order)
    pending = asyncio.create_task(run_write(tools, session))
    await entered.wait()
    assert order == ["flush:start"]
    gate.set()
    result = await pending
    assert result["isError"] is False
    assert order == ["flush:start", "flush:end", "tool"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/session/session-checkpoint-policy/tests/session-checkpoint-policy.spec.ts::"
    "session-checkpoint-policy tool and step boundaries > "
    "turns a rejected checkpoint into an error result without running the tool body"
)
async def test_rejected_checkpoint_skips_the_tool_body() -> None:
    ctx, sessions, tools = await setup()
    session = sessions.create("tool-failure")

    async def flush(*args: object) -> None:
        raise RuntimeError("disk unavailable")

    ctx.on("session/flush", flush)
    order: list[str] = []
    register_write(tools, order)
    result = await run_write(tools, session)
    assert result["isError"] is True
    assert result["content"] == [{"type": "text", "text": "Error: disk unavailable"}]
    assert order == []
    await ctx.fiber.dispose()
