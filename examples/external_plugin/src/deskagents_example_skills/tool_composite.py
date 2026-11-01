"""Optional composite consumer demonstrating the public single-call pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deepseek_cordis import Context

from deskagents.core import ToolsService, define_tool
from deskagents.core.tools import ToolRunContext
from deskagents.llm import HarnessError

inject = ["tools"]


def apply(ctx: Context, config: object = None) -> None:
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)

    async def execute(args: Mapping[str, Any], execution: ToolRunContext) -> str:
        nested = await tools.execute_call(
            {"callId": execution["callId"] + "/skill", "name": "skill", "arguments": dict(args)},
            parent=execution,
        )
        for item in nested.get("additionalContexts", ()):
            execution.defer_context(item)
        if nested["isError"]:
            raise HarnessError(nested["error"]["message"], "NESTED_TOOL_FAILED")
        if nested.get("concludesTurn"):
            execution.conclude_turn()
        return str(nested["value"])

    tools.register(
        define_tool(
            name="skill_composite",
            description="Load a skill through the tool pipeline.",
            parameters={"name": {"type": "string", "required": True}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
        ),
        ctx=ctx,
    )
