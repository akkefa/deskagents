"""Run one deterministic turn through the standalone core runtime."""

import asyncio

from deepseek_cordis import Context

from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
from deskagents.llm import LlmRuntime, create_user_message
from deskagents.testing import ScriptedAdapter


async def main() -> None:
    ctx = Context()
    await BundleLoader(ctx).load_file(MINIMAL_BUNDLE)
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    llm.register_adapter(
        ["scripted"],
        ScriptedAdapter(
            [
                [
                    {"type": "text-delta", "index": 0, "text": "Hello from the deskagents."},
                    {"type": "finish", "reason": {"kind": "stop"}},
                ]
            ]
        ),
    )
    handle = await ctx.get("agents").create(
        {
            "sessionId": "minimal-example",
            "agentOptions": {"provider": "scripted", "model": "scripted"},
        }
    )
    handle.agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Hello"}],
            source={"kind": "user"},
        )
    )
    await handle.agent.when_idle()
    print(handle.agent.session.derive_messages()[-1]["content"][0]["text"])
    await handle.dispose()
    await ctx.fiber.dispose()


if __name__ == "__main__":
    asyncio.run(main())
