"""Run a tool turn, close the runtime, and resume its durable v4 session.

Network-free: python examples/durable_agent.py /path/to/new/session-store
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from deepseek_cordis import Context

from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
from deskagents.core import AgentRegistry, ToolsService, define_tool
from deskagents.llm import LlmRuntime, create_user_message
from deskagents.llm.retry import apply as install_retry
from deskagents.llm.token_meter import TokenMeter
from deskagents.session import persistence_jsonl
from deskagents.testing import ScriptedAdapter


async def run(
    root: Path, *, resume: bool = False, compression: str = "zstd"
) -> list[dict[str, Any]]:
    ctx = Context()
    loader = BundleLoader(ctx)
    try:
        await ctx.plugin(persistence_jsonl, {"root": str(root), "compression": compression})
        await loader.load_file(MINIMAL_BUNDLE)
        await ctx.plugin(install_retry)
        await ctx.plugin(TokenMeter)
        llm, tools, agents = ctx.get("llm"), ctx.get("tools"), ctx.get("agents")
        assert isinstance(llm, LlmRuntime)
        assert isinstance(tools, ToolsService)
        assert isinstance(agents, AgentRegistry)

        async def lookup(args: Any, execution: Any) -> str:
            return "The answer is 42."

        tools.register(
            define_tool(
                name="lookup",
                description="Look up the answer",
                parameters={},
                output={
                    "schema": {"type": "string"},
                    "render": lambda args, value: [{"type": "text", "text": value}],
                },
                execute=lookup,
            )
        )
        scripts: list[Any] = (
            [
                [
                    {"type": "text-delta", "index": 0, "text": "Resumed successfully."},
                    {
                        "type": "usage",
                        "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                    },
                    {"type": "finish", "reason": {"kind": "stop"}},
                ]
            ]
            if resume
            else [
                [
                    {"type": "block-start", "index": 0, "blockType": "tool-call"},
                    {
                        "type": "tool-call-delta",
                        "index": 0,
                        "id": "lookup-1",
                        "name": "lookup",
                        "argumentsDelta": "{}",
                    },
                    {
                        "type": "block-end",
                        "index": 0,
                        "block": {
                            "type": "tool-call",
                            "id": "lookup-1",
                            "name": "lookup",
                            "arguments": "{}",
                        },
                    },
                    {"type": "finish", "reason": {"kind": "tool-calls"}},
                ],
                [
                    {"type": "text-delta", "index": 0, "text": "The answer is 42."},
                    {
                        "type": "usage",
                        "usage": {"inputTokens": 8, "outputTokens": 4, "totalTokens": 12},
                    },
                    {"type": "finish", "reason": {"kind": "stop"}},
                ],
            ]
        )
        adapter = ScriptedAdapter(scripts)
        llm.register_adapter(["scripted"], adapter)
        options = {
            "sessionId": "durable-example",
            "agentOptions": {"provider": "scripted", "model": "scripted"},
        }
        handle = await (agents.resume(options) if resume else agents.create(options))
        try:
            handle.agent.followup(
                create_user_message(
                    content=[
                        {"type": "text", "text": "Continue" if resume else "Look up the answer"}
                    ],
                    source={"kind": "user"},
                )
            )
            await handle.agent.when_idle()
            messages = handle.agent.session.derive_messages()
            assert messages[-1]["role"] == "assistant"
            print(messages[-1]["content"][0]["text"])
            return [dict(event) for event in handle.agent.session.snapshot_events()]
        finally:
            await handle.dispose()
    finally:
        await loader.dispose()
        await ctx.fiber.dispose()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    await run(args.root)
    await run(args.root, resume=True)


if __name__ == "__main__":
    asyncio.run(main())
