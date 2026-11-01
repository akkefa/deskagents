# deskagents

A Python implementation of the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
agent runtime, built on [deepseek-cordis](https://github.com/akkefa/deepseek-cordis).

PyPI distribution: `deskagents`. Python import: `deskagents`. Python 3.11+.

> **Status: pre-alpha.** The standalone core runtime is runnable: Scope, Session v4,
> JSONL persistence, projections, prompts, tools, the model seam, retry policy, token
> metering, agent loop, bundle composition, DeepSeek Messages and OpenAI-compatible
> adapters are implemented. See [the plan](docs/PLAN.md).

## What it is

DeepSeek Harness is a TypeScript agent runtime in which every part (the model
adapter, the tool registry, the session log, the agent loop) is a Cordis plugin.
This package follows the same design in Python:

- one Python module per upstream package, mapped in [docs/upstream.json](docs/upstream.json)
- the same event names, tool schemas and JSONL session format (v4); committed
  cross-runtime fixtures verify interoperability with the pinned `dsh` implementation
- agents composed from id-addressed YAML plugin rows through the Python bundle loader

It is an independent implementation, not an official DeepSeek product. Upstream's own
Python SDK (`deepseek_harness`) drives the Node `dsh` binary over JSON-RPC; this
package runs the harness in-process in Python instead.

## Core release

Release 0.1 is the complete scope of this distribution: session log and persistence,
prompts, tools, the agent loop, bundle composition, and DeepSeek/OpenAI-compatible
model adapters.

Execution, skills, web, MCP, Azure, telemetry and product integrations are separate
plugin families. See the [plugin roadmap](docs/plugin-roadmap.md), and
[extending deskagents](docs/extending.md) for writing a plugin package in another
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

Create the minimal standalone runtime and register a provider adapter:

```python
from deepseek_cordis import Context
from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
from deskagents.llm import LlmRuntime
from deskagents.llm.deepseek import DeepSeekAdapter

ctx = Context()
await BundleLoader(ctx).load_file(MINIMAL_BUNDLE)
llm = ctx.get("llm")
assert isinstance(llm, LlmRuntime)
llm.register_adapter(
    ["deepseek"],
    DeepSeekAdapter(api_key="...", models=({"id": "deepseek-chat"},)),
)
```

`deskagents.llm.openai.OpenAIAdapter` uses the same seam and supports custom compatible
base URLs plus Azure OpenAI deployment URLs. See `examples/minimal_agent.py` for a
network-free complete turn.

## Name clash

The unrelated PyPI package [`deskagents`](https://pypi.org/project/harness/) also installs a
top-level `deskagents` module. Don't install both in one environment.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. Upstream notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
