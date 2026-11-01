"""Consumer: advertise skills in the system prompt and load one on demand through a tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from deepseek_cordis import Context

from deskagents.core import SystemPrompt, ToolsService, define_tool
from deskagents_example_skills import SkillRegistry

inject = ["skills", "tools", "systemPrompt"]


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> None:
    registry = cast(SkillRegistry, ctx.get("skills"))
    tools = cast(ToolsService, ctx.get("tools"))
    prompt = cast(SystemPrompt, ctx.get("systemPrompt"))

    def catalog(context: Mapping[str, Any]) -> str:
        lines = [f"- {skill['name']}: {skill['description']}" for skill in registry.list()]
        return "Load a skill with the skill tool before following it.\n" + "\n".join(lines)

    async def execute(args: Mapping[str, Any], execution: object) -> object:
        skill = registry.get(args["name"])
        if skill is None:
            raise ValueError(f'unknown skill "{args["name"]}"')
        return skill["instructions"]

    prompt.section({"name": "skills", "order": 50, "text": catalog}, ctx=ctx)
    tools.register(
        define_tool(
            name="skill",
            description="Load the full instructions of one listed skill.",
            parameters={"name": {"type": "string", "required": True}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
        ),
        ctx=ctx,
    )
