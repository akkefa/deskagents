from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from deepseek_cordis import Context
from pydantic import TypeAdapter

from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
from deskagents.core import AgentRegistry, SystemPrompt, ToolsService, define_tool
from deskagents.llm import LlmRuntime
from deskagents.session.projection import (
    ProjectionDefinition,
    ProjectionWire,
    SessionProjectionRegistry,
)
from deskagents.testing import ScriptedAdapter


def text_tool(name: str) -> Mapping[str, Any]:
    async def execute(args: Mapping[str, Any], execution: object) -> str:
        del args, execution
        return "ok"

    return define_tool(
        name=name,
        description=name,
        parameters={},
        output={
            "schema": {"type": "string"},
            "render": lambda args, value: [{"type": "text", "text": value}],
        },
        execute=execute,
    )


class LifecyclePlugin:
    inject = ["tools", "systemPrompt", "llm", "sessionProjections", "agents"]

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    async def apply(self, ctx: Context, config: object = None) -> None:
        del config
        tools = ctx.get("tools")
        prompt = ctx.get("systemPrompt")
        llm = ctx.get("llm")
        projections = ctx.get("sessionProjections")
        assert isinstance(tools, ToolsService)
        assert isinstance(prompt, SystemPrompt)
        assert isinstance(llm, LlmRuntime)
        assert isinstance(projections, SessionProjectionRegistry)

        tools.register(text_tool("owned"), ctx=ctx)
        tools.guard(
            lambda execution: "blocked by plugin" if execution["name"] == "base" else None,
            ctx=ctx,
        )
        prompt.section(
            {"name": "plugin-lifecycle", "order": 50, "text": "mounted"},
            ctx=ctx,
        )
        llm.register_adapter(["plugin-models"], ScriptedAdapter(), ctx=ctx)
        await projections.register(
            ProjectionDefinition(
                key="plugin/lifecycle",
                state_schema=TypeAdapter(int),
                init=lambda header, inherited: 1,
                apply=lambda state, event: state,
                wire=ProjectionWire(
                    view_schema=TypeAdapter(int),
                    view=lambda state: state,
                ),
                state_version=1,
            ),
            ctx=ctx,
        )

        def pre_execute(execution: Mapping[str, Any], next_: Any) -> object:
            self.state["events"].append(("pre-before", execution["name"]))
            result = next_()
            self.state["events"].append(("pre-after", execution["name"]))
            return result

        async def created(payload: Mapping[str, Any]) -> None:
            self.state["events"].append(("created", payload["agent"].id))

        def result(execution: Mapping[str, Any], outcome: Mapping[str, Any]) -> None:
            self.state["events"].append(("result", execution["name"], outcome["isError"]))

        ctx.on("tools/pre-execute", pre_execute)
        ctx.on("agent/created", created)
        ctx.on("tools/result", result)


async def execute_base(tools: ToolsService, agent: object) -> Mapping[str, Any]:
    batch = await tools.execute(
        {
            "calls": [{"id": "base-1", "name": "base", "arguments": "{}"}],
            "signal": SimpleNamespace(aborted=False),
            "agent": agent,
        }
    )
    outcome: Mapping[str, Any] = batch["results"][0]["result"]
    return outcome


@pytest.mark.asyncio
async def test_plugin_fiber_owns_registrations_events_and_cleanup() -> None:
    state: dict[str, Any] = {"events": []}
    ctx = Context()
    loader = BundleLoader(ctx, {"test.lifecycle": LifecyclePlugin(state)})
    await loader.load_file(MINIMAL_BUNDLE)
    family = await loader.load([{"id": "lifecycle", "name": "test.lifecycle"}])

    tools = ctx.get("tools")
    prompt = ctx.get("systemPrompt")
    llm = ctx.get("llm")
    projections = ctx.get("sessionProjections")
    agents = ctx.get("agents")
    assert isinstance(tools, ToolsService)
    assert isinstance(prompt, SystemPrompt)
    assert isinstance(llm, LlmRuntime)
    assert isinstance(projections, SessionProjectionRegistry)
    assert isinstance(agents, AgentRegistry)
    tools.register(text_tool("base"), ctx=ctx)
    llm.register_adapter(["host"], ScriptedAdapter(), ctx=ctx)

    assert sorted(tool["name"] for tool in tools.definitions()) == ["base", "owned"]
    assert "plugin-models" in [provider["id"] for provider in llm.list_providers()]
    assembly = await prompt.assemble({})
    assert "plugin-lifecycle" in [section["name"] for section in assembly["sections"]]

    first = await agents.create(
        {
            "sessionId": "plugin-lifecycle-first",
            "agentOptions": {"provider": "host", "model": "scripted"},
        }
    )
    assert projections.snapshot(first.agent.session).values["plugin/lifecycle"] == 1
    blocked = await execute_base(tools, first.agent)
    assert blocked["isError"] is True
    assert blocked["error"]["message"] == "blocked by plugin"
    assert state["events"] == [
        ("created", "plugin-lifecycle-first"),
        ("pre-before", "base"),
        ("pre-after", "base"),
        ("result", "base", True),
    ]

    await family.dispose()
    assert [tool["name"] for tool in tools.definitions()] == ["base"]
    assert "plugin-models" not in [provider["id"] for provider in llm.list_providers()]
    assembly = await prompt.assemble({})
    assert "plugin-lifecycle" not in [section["name"] for section in assembly["sections"]]
    assert "plugin/lifecycle" not in projections.snapshot(first.agent.session).values

    events_after_dispose = list(state["events"])
    second = await agents.create(
        {
            "sessionId": "plugin-lifecycle-second",
            "agentOptions": {"provider": "host", "model": "scripted"},
        }
    )
    allowed = await execute_base(tools, second.agent)
    assert allowed["isError"] is False
    assert state["events"] == events_after_dispose

    await second.dispose()
    await first.dispose()
    await loader.dispose()
    await ctx.fiber.dispose()
