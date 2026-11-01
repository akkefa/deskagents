"""Opt-in, bounded live provider smoke test; never run by deterministic CI.

Set DEEPSEEK_API_KEY or OPENAI_API_KEY and pass a model available to that endpoint.
Only provider/model/endpoint and outcome flags are printed; no credentials or replies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any

from deepseek_cordis import Context

from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
from deskagents.core import AgentRegistry, ToolsService, define_tool
from deskagents.llm import LlmRuntime, create_user_message


async def smoke(
    provider: str, model: str, base_url: str | None, request_deadline: float
) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY")
    if not key:
        raise ValueError("provider API key is not set")
    import httpx

    from deskagents.llm.deepseek import DeepSeekAdapter
    from deskagents.llm.openai import OpenAIAdapter

    ctx, stream_started = Context(), asyncio.Event()
    loader = BundleLoader(ctx)
    used_tool = False
    async with httpx.AsyncClient(timeout=request_deadline) as client:
        try:
            await loader.load_file(MINIMAL_BUNDLE)
            llm, tools, agents = ctx.get("llm"), ctx.get("tools"), ctx.get("agents")
            assert isinstance(llm, LlmRuntime) and isinstance(tools, ToolsService)
            assert isinstance(agents, AgentRegistry)
            adapter_type = DeepSeekAdapter if provider == "deepseek" else OpenAIAdapter
            options: dict[str, Any] = {
                "api_key": key,
                "client": client,
                "models": ({"id": model, "name": model},),
            }
            if base_url:
                options["base_url"] = base_url
            llm.register_adapter([provider], adapter_type(**options))

            async def lookup(args: Any, execution: Any) -> str:
                nonlocal used_tool
                used_tool = True
                return "smoke-check-42"

            tools.register(
                define_tool(
                    name="smoke_lookup",
                    description="Retrieve the smoke test code.",
                    parameters={},
                    output={
                        "schema": {"type": "string"},
                        "render": lambda args, value: [{"type": "text", "text": value}],
                    },
                    execute=lookup,
                )
            )
            ctx.on("agent/assistant-stream", lambda *_: stream_started.set())
            handle = await agents.create(
                {
                    "sessionId": "live-smoke",
                    "agentOptions": {"provider": provider, "model": model, "maxTokens": 512},
                }
            )
            try:

                async def turn(prompt: str) -> None:
                    handle.agent.followup(
                        create_user_message(
                            content=[{"type": "text", "text": prompt}], source={"kind": "user"}
                        )
                    )
                    await asyncio.wait_for(handle.agent.when_idle(), request_deadline)
                    end = [
                        event
                        for event in handle.agent.session.snapshot_events()
                        if event["type"] == "turn/end"
                    ][-1]
                    if end["data"]["reason"]["kind"] != "completed":
                        raise RuntimeError("provider turn did not complete")

                await turn("Say hello in one short sentence.")
                await turn("Call smoke_lookup once and report its code.")
                if not used_tool:
                    raise RuntimeError("model did not execute the requested tool")
                await turn("What code did the tool return? Reply briefly.")
                stream_started.clear()
                handle.agent.followup(
                    create_user_message(
                        content=[{"type": "text", "text": "Count upwards slowly."}],
                        source={"kind": "user"},
                    )
                )
                await asyncio.wait_for(stream_started.wait(), request_deadline)
                handle.agent.cancel()
                await asyncio.wait_for(handle.agent.when_idle(), request_deadline)
                end = [
                    event
                    for event in handle.agent.session.snapshot_events()
                    if event["type"] == "turn/end"
                ][-1]
                if end["data"]["reason"]["kind"] != "aborted":
                    raise RuntimeError("cancellation did not produce an aborted turn")
            finally:
                handle.agent.cancel()
                await handle.dispose()
            return {
                "provider": provider,
                "model": model,
                "base_url": base_url or "adapter default",
                "checked_at": datetime.now(UTC).isoformat(),
                "text": True,
                "tool_roundtrip": True,
                "subsequent_turn": True,
                "cancellation": True,
            }
        finally:
            await loader.dispose()
            await ctx.fiber.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("deepseek", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        parser.error("timeout must be between 0 and 300 seconds")
    try:
        result = asyncio.run(smoke(args.provider, args.model, args.base_url, args.timeout))
    except Exception as error:
        # Provider bodies can contain user data; do not print the exception contents.
        raise SystemExit(
            f"Live smoke failed ({type(error).__name__}); check credentials, model and endpoint."
        ) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
