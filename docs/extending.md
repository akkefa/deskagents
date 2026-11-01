# Extending deskagents from another repository

Skills, web search, MCP, execution backends and product integrations live outside
this distribution as plugin packages. A plugin package depends on `deskagents` and
`deepseek-cordis`, declares its plugins as entry points, and is mounted by bundle rows,
just as upstream DeepSeek Harness mounts npm packages. The family boundaries are in
[plugin-roadmap.md](plugin-roadmap.md).

[`examples/external_plugin`](../examples/external_plugin) is a complete plugin
distribution laid out as its own repository. `tests/bundle/test_external_plugin.py`
mounts it on the core bundle and drives a turn through it.

## Compatibility guarantee

At the upstream pin in [upstream.json](upstream.json), a manually converted
**headless** DeepSeek Harness plugin can use the corresponding public Python services,
events and lifecycle:

- install the Python distribution;
- declare its Cordis plugins in the `deskagents.plugins` entry-point group; and
- compose their names with bundle rows.

Loading and lifecycle are automatic after conversion. Source conversion is not: imports,
types, promises, cancellation and any package-specific dependencies must be translated
to Python. A plugin from a newer upstream revision may require a deskagents pin update.

The profile excludes Node host APIs, browser/client UI, HMR, pnpm profile installation,
nested Loader trees, `include`, and JavaScript `!!js` bundle expressions. Use flat
id-addressed rows and Python configuration. A headless plugin that depends on another
upstream service (for example `fs`, `web` or `skills`) also needs that service's Python
plugin package; the core does not manufacture missing capabilities.

## A plugin

A plugin is a Cordis plugin:

- a module with `apply(ctx, config)`, or
- a `deepseek_cordis.Service` subclass with a `name`, which provides `ctx.get(name)`.

Either form may declare `inject = ["tools", "systemPrompt", ...]`. The plugin starts
when those services exist and stops when any of them leaves.

A plugin may expose `Config.validate(value)` for synchronous configuration validation.
The normalized return value is passed to `apply` or the `Service` constructor.

Register through the plugin's `ctx`. Every registration below is owned by the
registering plugin and is removed when it unloads:

```python
from deskagents.core import define_tool

inject = ["tools", "systemPrompt"]


def apply(ctx, config=None):
    ctx.get("systemPrompt").section({"name": "weather", "order": 50, "text": "..."}, ctx=ctx)
    ctx.get("tools").register(define_tool(name="weather", ...), ctx=ctx)
    ctx.on("agent/turn-stopping", on_turn_stopping)
```

## A plugin distribution

```text
deskagents-weather/
  pyproject.toml
  src/deskagents_weather/__init__.py      Service Definition (optional)
  src/deskagents_weather/provider_x.py    providers
  src/deskagents_weather/tool_weather.py  model-facing consumer
```

```toml
[project]
dependencies = ["deskagents>=0.1,<0.2", "deepseek-cordis>=0.2.0,<0.3"]

[project.entry-points."deskagents.plugins"]
"deskagents_weather" = "deskagents_weather:WeatherService"
"deskagents_weather.tool_weather" = "deskagents_weather.tool_weather"
```

The entry-point name is what a bundle row's `name` refers to. A name declared by two
installed distributions is rejected.

## Converting an upstream TypeScript plugin

1. Read the package and tests at the deskagents upstream pin, including every package
   it imports.
2. Keep its public service name, event names, tool names, schemas and durable record
   vocabulary unchanged.
3. Translate a `Service` subclass or `apply(ctx, config)` module and its `inject`
   declaration. Use `async`/`await` for promises, and `deskagents.util.abort` where
   upstream uses `AbortController` and `AbortSignal` (see Cancellation below).
4. Register every listener, tool, projection, prompt section and adapter through the
   plugin's `ctx`; pass `ctx=ctx` to registry methods. Do not return an awaited
   registration as though it were a JavaScript disposer.
5. Port or depend on Python providers for every non-core service the plugin consumes.
6. Declare entry points, add bundle rows, and run the plugin contract plus any
   capability-specific contracts and upstream cases.

## Mounting

Bundle rows resolve a `name` to a plugin passed to `BundleLoader`, then a core plugin,
then an installed entry point. A host extends the core bundle with id-addressed
patches, so installing a plugin never changes the core bundle:

```python
import yaml
from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader, apply_row_patches, load_bundle_rows

rows = apply_row_patches(load_bundle_rows(MINIMAL_BUNDLE), yaml.safe_load(open("bundle.yml")))
await BundleLoader(ctx).load(rows)
```

```yaml
- insert:
    - id: weather
      name: deskagents_weather
      config: {units: metric}
    - id: tool-weather
      name: deskagents_weather.tool_weather
- id: system-prompt          # patch a core row
  config: {personaPrefix: You are the weather desk assistant.}
```

A patch replaces the named top-level fields of a row whole; `config` is not merged.

`BundleBuilder` builds the same rows in Python. `BundleLoader(ctx, plugins={...})`
mounts plugins that are not installed, such as a host application's own.

## Extension points

Registries, all taking `ctx=` for ownership:

| Service | Public import | Registration |
|---|---|---|
| `tools` | `deskagents.core` | `register(definition, when=...)`, `guard(callback)` |
| `systemPrompt` | `deskagents.core` | `section(...)`, `variable(name, provider)`, `tools(provider)` |
| `llm` | `deskagents.llm` | `register_adapter(routes, adapter)`, `models_updated(provider)`; see [model-adapters.md](model-adapters.md) |
| `deepseekLlmApiExtensions` | `deskagents.llm` | `register(field, provider)` for DeepSeek request fields |
| `sessionPersistence` | `deskagents.session.persistence` | a `SessionPersistence` subclass for a storage backend |
| `sessionProjections` | `deskagents.session.projection` | `register(definition)`, `on_changed(listener)` |
| `agents` | `deskagents.core` | `create`, `resume`, and initiator attribution for child work |

An agent's `run_maintenance(job)` is upstream's `runMaintenance`: it runs
`job(signal)` while the agent is idle, for work such as compaction or subagent
bookkeeping. Waking input sent meanwhile waits and runs when the job settles;
`cancel()` aborts the job's signal and `when_idle()` waits for it. It raises while a
turn or another maintenance job owns the agent.

Events, with the upstream names and dispatch modes. A waterfall listener receives
`next_` and returns the (possibly replaced) result. Listener failures of notification
events are logged and contained; failures of waterfall and serial listeners propagate.

| Event | Mode | Use |
|---|---|---|
| `tools/pre-execute` | waterfall | allow, deny, ask or cancel a call before dispatch |
| `tools/execute` | waterfall | wrap dispatch; `next_()` returns the normalized result; may replace `execution.signal` for the body |
| `tools/post-execute` | waterfall | inspect each result, errors included |
| `tools/result` | notification | observe the final frozen result |
| `agent/created` | serial | set up per-agent state |
| `agent/pre-step` | waterfall | admit, rewrite or reject a step's input; `startsRequestSeries` |
| `agent/request` | waterfall | adjust the model call configuration |
| `agent/request-error` | waterfall | decide whether a failed request retries |
| `agent/assistant-stream` | notification | live `start`, `chunk` and `end` frames of each attempt |
| `agent/turn-stopping` | serial | `agent.steer(...)` to continue the turn |
| `agent/status`, `agent/error`, `agent/disposed`, `agent/inbox/*` | notification | observe |
| `llm/stream` | waterfall | wrap a model stream |
| `llm/models-updated` | notification | tell catalog consumers to reload one provider |
| `session/event` | notification | observe each committed session event |
| `session/flush` | awaited | make the session durable at a checkpoint |

An `ask` decision calls the optional `approval` service. Without that service, without
an agent to route through, or when the channel is unavailable, the call degrades to a
denial. The approval service implementation remains in the external control family.
`tools/post-execute` may replace successful `value` (which is validated and rendered
again), replace `content`, block a result, and attach `additionalContexts`.

A tool definition may also declare upstream's `projectContent(execution, result)`,
which prepares content before post-execute policy, and `finalizeContent(execution,
result)`, which runs once on every final result. `timeoutMs` is only a declaration; a
`tools/execute` wrapper enforces it, as upstream's timeout policy plugin does.

## Cancellation

Cancellation is cooperative, as upstream. `agent.cancel(reason)` aborts the running
turn's signal; it never cancels a task. The loop stops starting work, an open model
stream is closed, and a started tool body runs until it returns. A success that
finishes after the abort becomes a `tool call aborted` result, and calls that never
started get `tool call aborted before dispatch`, so every tool call is answered. A
listener or tool may cancel its own agent.

Tool executions, model requests and agent events carry the signal as `signal`. Plugin
code observes it through `deskagents.util.abort`:

| Upstream | Python |
|---|---|
| `new AbortController()`, `controller.abort(reason)` | `AbortController()`, `controller.abort(reason)` |
| `signal.aborted`, `signal.reason`, `signal.throwIfAborted()` | `signal.aborted`, `signal.reason`, `signal.throw_if_aborted()` |
| `signal.addEventListener('abort', fn)` | `remove = signal.add_listener(fn)` |
| awaiting the abort event | `await signal.wait()` |
| `AbortSignal.any([...])`, `AbortSignal.timeout(ms)` | `AbortSignal.any([...])`, `AbortSignal.timeout(ms)` |

`abortable(awaitable, signal)` awaits work unless the signal aborts first, then cancels
that work and raises `AbortError`. A `tools/execute` wrapper that installs its own
deadline sets `execution.signal`, awaits `next_()`, and restores the original signal
in `finally`; the body then sees the caller's and the wrapper's signals fused.

## Skills and web search

Build these families with upstream's three seam roles: a Service Definition, providers
and a model-facing consumer that depend on the definition only.

| Family | Service Definition | Providers | Consumer |
|---|---|---|---|
| Skills | `skill/skill` (`skills`) | `skill/skill-filesystem` | `skill/tool-skill` |
| Web | `web/web` (`web`) | `web/web-fetch-http`, `web/web-search-*` | `web/tool-web` |

The skill registry and the web service need only the core seams. Upstream's filesystem
skill provider needs the `fs` seam, and `tool-skill` needs `session/working-directory`,
so those come after the execution family. `tool-web` and the HTTP fetcher use
`util/output-retention` and `util/timeout`. The example package is a reduced skill
family; a full port follows the upstream packages above and their tests.

## Testing a plugin

- Mount it with the core bundle in a `Context` and drive turns with
  `deskagents.testing.ScriptedAdapter`.
- Run `deskagents.testing.run_plugin_contract` with a factory that returns a fresh
  `PluginContractBackend`: the host `ctx` without the family, `mount` and
  `dispose_plugin` steps (each may run twice), `dispose`, and the `services` the
  family provides. The contract compares the host before mounting and after
  unloading (host-owned effects, listeners of the harness events, and the tool,
  prompt, model-provider, agent, session and projection registries) and mounts the
  family a second time, so a registration that outlives the family fails it. Optional
  `assert_active`, `exercise` and `assert_disposed` probes add the family's own checks;
  `exercise` must dispose any agents it creates. It does not depend on pytest.
- Run the capability contracts in `deskagents.testing` for persistence backends, live
  write paths and model adapters.
- Port the upstream package's behavior tests and mark each with its pinned source.

Use only public `deskagents` and `deepseek-cordis` APIs; modules and attributes whose
names start with an underscore are private.

## Single-call and composite execution

`await tools.execute_call({"callId": "unique-id", "name": "tool-name",
"arguments": {...}, "signal": signal})` dispatches parsed arguments through the
same policy/body/post-processing/finalization stages as native batches. It returns
one immutable result and publishes `tools/result`. A pre-aborted single call still
snapshots arguments, finalizes, and notifies; an unstarted batch slot retains the
batch scheduler's synthetic-result behavior.

Inside a composite body, pass `parent=execution` to inherit scope, step admission,
signal and root call identity. Use a unique child call ID. Nested calls do not
reserve another batch worker slot. They do not append separate root session log
entries: the outer call owns its model-visible result. Forward `additionalContexts`
and successful `concludesTurn` explicitly when the composite's semantics require it;
a blocked or failed nested result must not conclude the outer turn.

The optional external example module
`examples/external_plugin/src/deskagents_example_skills/tool_composite.py` demonstrates
this pattern and is exercised by the reusable plugin unload/remount contract.
Custom executors must implement `execute_call` to expose this path. Native executors
also accept optional `source` callbacks (`get`, `reject`) for executor-owned lookup
and restrictions; rejection precedes extensible policy, with cancellation taking
precedence over rejection.
