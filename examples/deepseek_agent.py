"""Run the minimal harness against the DeepSeek Messages API."""

import asyncio
import os

from deepseek_cordis import Context

from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
from deskagents.llm import LlmRuntime, create_user_message
from deskagents.llm.deepseek import DeepSeekAdapter


async def main() -> None:
    ctx = Context()
    loader = BundleLoader(ctx)
    await loader.load_file(MINIMAL_BUNDLE)
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    llm.register_adapter(
        ["deepseek"],
        DeepSeekAdapter(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            models=({"id": os.getenv("DEEPSEEK_MODEL", "deepseek-chat")},),
        ),
    )
    agents = ctx.get("agents")
    handle = await agents.create(
        {
            "sessionId": "deepseek-example",
            "agentOptions": {
                "provider": "deepseek",
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            },
        }
    )
    handle.agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Say hello in one sentence."}],
            source={"kind": "user"},
        )
    )
    await handle.agent.when_idle()
    print(handle.agent.session.derive_messages()[-1]["content"][0]["text"])
    await handle.dispose()
    await loader.dispose()
    await ctx.fiber.dispose()


if __name__ == "__main__":
    asyncio.run(main())
