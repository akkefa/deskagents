# Plugin roadmap

`deskagents` owns the standalone agent runtime: the public capability seams,
session model and persistence, model interface, tool registry and executor, agent
loop, bundle loader, and the DeepSeek/OpenAI-compatible adapters needed for a useful
0.1 release.

Everything below is intentionally outside the core distribution. Each capability is
expected to live in an independently versioned package and repository which depends on
the public `deskagents` seams and validates its providers with the contract suites
published by `deskagents.testing`.

## Compatibility profile

The core guarantees the public lifecycle, service, registry and event contracts needed
by manually converted headless plugins at the pinned DeepSeek Harness revision. Once a
Python plugin distribution is installed, entry-point discovery and bundle composition
mount it without host code imports. `deskagents.testing.run_plugin_contract` verifies
mount, execution and teardown.

This is not JavaScript source compatibility. Each family ports its own implementation
and non-core dependencies. Node hosts, client UI, HMR, pnpm profile management, nested
Loader trees and JavaScript bundle expressions remain outside the Python profile. See
[extending.md](extending.md) for the exact conversion and testing contract.

## Planned plugin families

| Family | Candidate distribution | Upstream groups or product code |
|---|---|---|
| Local execution | `deskagents-execution` | subprocess, sandbox, filesystem, shell, working directory, attachment, spill, timeout policy |
| Azure execution | `deskagents-azure` | PepGPT Azure Container Apps filesystem and subprocess providers |
| Context and control | `deskagents-control` | compaction, instructions, time context, approvals, user questions, todo and plan mode |
| Web | `deskagents-web` | web fetch, search providers and model-facing web tools |
| MCP | `deskagents-mcp` | MCP client and resources |
| Skills | `deskagents-skills` | skill seam, filesystem provider and model-facing skill tool |
| Subagents | `deskagents-subagents` | in-process subagent providers and control tools |
| Product foundation | `deskagents-product` | telemetry, settings, credentials, storage and presets |
| PepGPT compatibility | `deskagents-pepgpt` or PepGPT-owned plugins | memory/Postgres persistence, model providers, SSE/UI integration and migration helpers |

These names are placeholders, not reserved package names. Repositories should be
created only when work on that family starts.

## Boundary rules

1. Plugins depend on service definitions and event contracts exported by
   `deskagents`, never on a concrete provider or a private module.
2. Plugins may depend directly on `deepseek-cordis` for plugin lifecycle APIs, but
   may not use its private attributes. `deskagents._cordis` is private.
3. Provider packages run the applicable suites from `deskagents.testing`.
4. Optional dependencies required by a plugin stay in that plugin's distribution.
5. Durable event and model-visible names remain compatible with the pinned DeepSeek
   Harness version.
6. A plugin may be composed through YAML rows or the equivalent Python row builder;
   installing it must not mutate the core bundle implicitly. Installed plugins are
   declared in the `deskagents.plugins` entry-point group; see
   [extending.md](extending.md).

## PepGPT

PepGPT replacement is a consumer migration, not part of the core release train. Its
in-memory and Postgres persistence backends, Bedrock/Gemini/LangChain adapters, UI/SSE
host, telemetry listeners, memory, auto-skills and durable-tool behavior remain
PepGPT-owned plugins until they are useful to another consumer.

The core repository keeps the migration inventory because it exercises the stability
of the public seams. It does not provide compatibility aliases for PepGPT's old import
tree.
