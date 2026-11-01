# deskagents

A Python implementation of the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
agent runtime, built on [deepseek-cordis](https://github.com/akkefa/deepseek-cordis).

PyPI distribution: `deskagents`. Python import: `deskagents`. Python 3.11+.

> **Status: alpha candidate (`0.1.0a1`).** The standalone core is runnable and the
> eight audited DeepSeek defects are fixed. Upstream compatibility remains partial;
> see the [release contract](https://github.com/akkefa/deskagents/blob/main/docs/release-contract.md) for evidence and exclusions.
> Live provider and remote platform checks are separate release gates.

## What it is

DeepSeek Harness is a TypeScript agent runtime in which every part (the model
adapter, the tool registry, the session log, the agent loop) is a Cordis plugin.
This package follows the same design in Python:

- one Python module per upstream package, mapped in [docs/upstream.json](https://github.com/akkefa/deskagents/blob/main/docs/upstream.json)
- the same event names, tool schemas and JSONL session format (v4); committed
  cross-runtime fixtures verify interoperability with the pinned `dsh` implementation
- agents composed from id-addressed YAML plugin rows through the Python bundle loader

It is an independent implementation, not an official DeepSeek product. Upstream's own
Python SDK (`deepseek_harness`) drives the Node `dsh` binary over JSON-RPC; this
package runs the harness in-process in Python instead.

## Core release

The target scope for release 0.1 is session log and persistence,
prompts, tools, the agent loop, bundle composition, and DeepSeek/OpenAI-compatible
model adapters.

Execution, skills, web, MCP, Azure, telemetry and product integrations are separate
plugin families. See the [plugin roadmap](https://github.com/akkefa/deskagents/blob/main/docs/plugin-roadmap.md), and
[extending deskagents](https://github.com/akkefa/deskagents/blob/main/docs/extending.md) for writing a plugin package in another
repository.

A manually ported headless DeepSeek Harness plugin can keep the pinned service names,
events and schemas, then load through Python entry points and bundle rows. This does not
include Node/UI/HMR plugins or automatic TypeScript source conversion.

## Quick start

Install the core with the provider extra you need:

```bash
pip install "deskagents[deepseek]"
# or: pip install "deskagents[openai]"
```

Set `DEEPSEEK_API_KEY` in your environment and optionally `DEEPSEEK_MODEL` to a
model available to your account. Run this as a Python script:

```python
import asyncio
import os

from deepseek_cordis import Context

from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
from deskagents.llm import LlmRuntime, create_user_message
from deskagents.llm.deepseek import DeepSeekAdapter


async def main() -> None:
    ctx = Context()
    loader = BundleLoader(ctx)
    try:
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
        try:
            handle.agent.followup(
                create_user_message(
                    content=[{"type": "text", "text": "Say hello in one sentence."}],
                    source={"kind": "user"},
                )
            )
            await handle.agent.when_idle()
            print(handle.agent.session.derive_messages()[-1]["content"][0]["text"])
        finally:
            await handle.dispose()
    finally:
        await loader.dispose()
        await ctx.fiber.dispose()


if __name__ == "__main__":
    asyncio.run(main())
```

`deskagents.llm.openai.OpenAIAdapter` uses the same seam and supports custom compatible
base URLs plus Azure OpenAI deployment URLs. See `examples/minimal_agent.py` for a
network-free complete turn.

## Durable agents and plugins

[examples/durable_agent.py](https://github.com/akkefa/deskagents/blob/main/examples/durable_agent.py) composes JSONL storage,
checkpointing, retry, token metering and a tool, then closes and resumes the session.
It runs without network credentials:

```sh
python examples/durable_agent.py /path/to/new/session-store
```

The minimal bundle supplies the core services and checkpoint policy; it does not
register a provider or enable persistent storage, retry, or token metering for you.
See [model adapters](https://github.com/akkefa/deskagents/blob/main/docs/model-adapters.md), [plugin development](https://github.com/akkefa/deskagents/blob/main/docs/extending.md),
and [publishing](https://github.com/akkefa/deskagents/blob/main/docs/publishing.md).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. Upstream notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
