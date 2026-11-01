# Cordis alignment (plan Phase 1)

Date: 2026-10-30, updated 2026-10-31. Upstream pin: DeepSeek Harness `ce7073d`
(vendored Cordis `4.0.6-alpha.1`, Loader `1.0.7-alpha.1`). Python runtime checked:
deepseek-cordis `0.2.0` from PyPI, the version locked here.

## Method

1. `scripts/cordis_usage.py` reads every mapped package's sources at the pin (tests
   excluded) and lists the Cordis features they use, with the packages and the
   earliest milestone that needs each one. Re-run it whenever the pin moves.
2. Each feature was checked against deepseek-cordis's public API and documented
   behaviour (`PyCordis/docs/compatibility.md`), and the ones the harness depends on
   were probed against the installed package.
3. `tests/test_cordis_contract.py` keeps those probes. A gap is a strict `xfail` until
   deepseek-cordis gains the feature; it then becomes an ordinary contract test.

## Result

deepseek-cordis 0.2.0 covers everything the core distribution uses. It closes G1–G5:
public `Events.dispatch`, `Fiber.assert_active`, the `internal/status` and
`internal/plugin` events, `fiber.entry`, and config resolution. The 0.1 shims are gone;
the harness calls these APIs directly and `deskagents._cordis` now holds only the shared
contained-notification helper. G6 remains open and is not needed by the core.

### Covered

| Feature | Upstream use at the pin | deepseek-cordis |
|---|---|---|
| `ctx.on` / `once`, with `prepend` and `global` | 31 packages | `on(name, listener, prepend=, global_=)` |
| `ctx.emit`, `parallel`, `serial` | 6, 1, 1 packages | same names; `parallel` is all-settled |
| `ctx.waterfall`, listener `(*args, next)` | 7 packages (tool pipeline, prompt, fs intents) | same shape: `waterfall(name, *args, next_=)`; zero-argument `next()` |
| `ctx.effect` with a dispose callback, `function*`, `async function*` | 33 packages | callable, sync and async generator effects |
| `ctx.inject([...], callback)` with reactive reload | 16 packages | same; consumer returns to `PENDING` when a provider leaves (probed) |
| `ctx.plugin(plugin, config)` | 5 packages | same |
| `ctx.get(name)`, `ctx.provide(name, value)` | 24, 2 packages | same, plus `require` and `set` |
| `ctx.extend(meta)` | `core/scope` (0.1), `preset/agent-preset-registry` | same |
| `Service` subclasses registered by name | 33 packages | `Service(ctx, config, name=...)` or a class `name` |
| `[Service.init]` (async setup after construction) | 15 packages | `Service.start()` |
| `fiber.state`, `FiberState.*`, `fiber.uid` | 7, 4 packages | same; all six states |
| `fiber.await()` | `preset/agent-preset-registry` (0.5) | `fiber.wait()` |
| Isolated service slots (`Context.isolate`) | `preset/agent-preset-registry` (0.5) | `ctx.isolate(name, label)` and `ScopeLabel` |

### Gaps

G1–G5 are closed by deepseek-cordis 0.2.0; the table keeps the original findings.

| Id | Feature | Needed by | First needed | deepseek-cordis 0.1.1 | Proposed fix |
|---|---|---|---|---|---|
| **G1** | `ctx.events.dispatch(mode, args)`: the snapshot of listener callables, which the caller invokes itself to contain listener failures | `core/session`, `core/agent`, `core/agent-loop`, `core/tools`, `credentials/credentials` +5 | **0.1, Phase 2** (`core/session`) | private `Events._dispatch(mode, name, args, filter_)` | make it public as `Events.dispatch`, with the same snapshot and filter rules |
| **G2** | `fiber.assertActive()`: raise if the owning fiber has been disposed | `core/agent-loop`, `session/session-projection`, `session/working-directory`, `context/time-context`, `subagent/tool-subagent` | **0.1, Phase 3** (`session-projection`) | missing | add `Fiber.assert_active()`: raise `CordisError` when `uid is None`, as upstream does |
| **G3** | lifecycle events `internal/status(fiber, old_state)` on every state change, and `internal/plugin(fiber)` when a fiber is published or disposed | `core/agent` (`internal/status`); external MCP plugin (`internal/plugin`) | **0.1, Phase 5** (`core/agent`) | only `internal/dispatch` is emitted | core agent teardown uses explicit Cordis-owned effects; an external MCP plugin must either avoid this event or require a future Cordis release |
| G4 | `fiber.entry`: the loader row that created a fiber | `core/agent-default-model`, `llm/llm-deepseek-api-key`, `preset/agent-preset-registry` | 0.5 | missing | have `Loader` record the row id on the fibers it creates. Upstream code always falls back when `entry` is missing, so 0.1 does not need it. |
| G5 | Config access: `fiber.effectiveConfig`, `resolveConfig(candidate)`, raw config replacement | `settings/settings` | 0.5 | deferred in the 2026-10-28 PyCordis audit | follow upstream's Cordis rc.11 PR 1, which is defining this API now |
| G6 | Loader features: entry trees and groups, `include`, `interpolate`, JS expressions, `ctx.root.loader`, `ctx.reflect.store` | `preset/agent-preset`, `preset/agent-preset-registry`, `settings/settings` | 0.5 | deterministic flat-row loader only | revisit with presets in 0.5, after upstream's Loader rc.8 upgrade |

### Python differences (not gaps)

| Upstream | Harness | Why |
|---|---|---|
| `ctx.logger(...)` (34 packages) | `logging.getLogger(__name__)` per module | deepseek-cordis has no LoggerService and uses stdlib logging; the package logger already has a `NullHandler` |
| `getTraceable(ctx, target)`, `symbols.original` (`core/agent`) | pass the owner context explicitly | deepseek-cordis has no tracing proxies. `core/agent` uses them only to bind an agent factory to the context that registered it; Phase 5 does that binding explicitly and records it in the compatibility catalog. |
| `Volatile<T>` / `isVolatile` (cosmokit, 14 packages, including `core/agent-loop` and `llm/llm-deepseek`) | a harness-owned `Volatile` holder | `Volatile` comes from cosmokit, not from Cordis core; Cordis only re-exports the type. The harness adds a small equivalent in Phase 5 (a current value plus change notification) without changing deepseek-cordis. |
| schemastery config schemas (57 packages) | Pydantic models behind the deepseek-cordis `ConfigValidator` protocol | per the plan's translation rules; Phase 2 adds the adapter |

### Not used by any mapped package

`ctx.bail`, `[Service.check]`, `ctx.intercept`, `ctx.set`, `ctx.accessor`, `ctx.mixin`,
and the timer plugin (`ctx.setTimeout`, `ctx.sleep` and friends). None of them is
needed by this package.

## Upstream Cordis rc.11

Upstream's `vendor/cordis-rc11-upgrade-plan.md` is still a proposal at `ce7073d`. It
would move vendored Cordis to `4.0.0-rc.11` and Loader to `1.0.0-rc.8` in staged PRs;
its Loader work overlaps G6. `docs/upstream.json` records the vendored versions, and the
upstream watch reports any commit under `vendor/cordis` or `vendor/loader`, so
deepseek-cordis can follow before this package raises its range.

## Core 0.1 compatibility policy

1. The harness uses only public deepseek-cordis APIs; no module touches Cordis private
   attributes.
2. Non-vetoing notifications (`agent/*`, `tools/result`, `llm/adapters-updated`) go
   through `deskagents._cordis.emit_contained`, which takes the `Events.dispatch`
   snapshot and logs each listener's sync or async failure, as upstream does.
3. G6 belongs to external product and preset plugins and is not a core 0.1 blocker.

## Dependencies on deepseek-cordis internals

Added in Phase 3. Recheck these whenever the deepseek-cordis pin moves:

- `deskagents.session.projection` subclasses `deepseek_cordis.Effect`, through its public
  constructor `(owner, setup, label)`, so that a registration is both its handle and its
  disposer, as upstream's is. A change to that constructor or to how effects adopt
  returned values would break it; `tests/session/projection/test_compatibility.py`
  covers the adoption cases.
