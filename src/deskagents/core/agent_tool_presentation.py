"""Compatibility selector for built-in tool execution presentation.

Upstream: packages/core/agent-tool-presentation at ce7073d.
"""

from __future__ import annotations

from deepseek_cordis import Context

from .tools import NativeToolExecutor, ToolsService


def apply(ctx: Context, mode: str = "native") -> NativeToolExecutor:
    if mode != "native":
        raise ValueError("PTC presentation belongs to an external plugin; use mode='native'")
    tools = ctx.get("tools")
    if not isinstance(tools, ToolsService):
        raise RuntimeError("ToolsService is not installed")
    executor = NativeToolExecutor(ctx)
    tools.set_executor(executor, ctx=ctx)
    return executor


__all__ = ["apply"]
