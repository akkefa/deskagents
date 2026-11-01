# PepGPT migration inventory

Status: **in progress, updated 2026-10-31**. Every core phase updates the rows it covers.

This lists every symbol PepGPT imports from its in-tree harness package today
(129 symbols, from `app/`, `m365/` and `onereach_profile/` at PepGPT `dev` @
`0cffd1ff`), and where each one lands when PepGPT switches to this package. PepGPT
imports that package as `harness`; this document writes it as `deskagents`. The
import surface was re-verified unchanged at `dev` @ `0316255c` on 2026-10-31.

By status: 48 symbols are provided by this package, 30 wait for external plugin
families (execution, Azure, control, skills, web and product), 39 become PepGPT-owned
plugins on public seams, and 12 stay in PepGPT's application code.

Targets marked *proposed* follow the upstream package that owns the concept. They
are confirmed, with exact names, in the phase that ports that package.

This inventory describes a bounded migration to the core package plus external
providers. It does not promise compatibility aliases for PepGPT's current import tree.

## Status values

- **package**: provided by this package. The target column gives the module.
- **external-plugin**: provided by a separately versioned plugin family.
- **pepgpt-plugin**: stays in PepGPT as its own plugin on a public seam.
- **pepgpt-app**: stays in PepGPT's application code. It is not a harness concern.
- **dropped**: replaced by upstream behaviour; PepGPT changes its call site.

## Kernel

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.kernel.KernelContext` | package | `deepseek_cordis.Context` | 0.1 |
| `deskagents.kernel.PluginManager` | package | `deepseek_cordis.loader.Loader` + `deskagents.bundle` | 0.1 |
| `deskagents.kernel.Scope` | package | `deskagents.core.scope` | 0.1 |
| `deskagents.kernel.EffectHandle` | package | `deepseek_cordis.Effect` | 0.1 |
| `deskagents.kernel.runtime.KernelRuntime` | package | `deskagents.bundle` boot API | 0.1 |

## Session and persistence

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.core.session.Session` | package | `deskagents.core.session` | 0.1 |
| `deskagents.core.session.SessionEvent` | package | `deskagents.core.session` (v4 event types) | 0.1 |
| `deskagents.core.session.CoreEventType` | package | `deskagents.core.session` event type names | 0.1 |
| `deskagents.core.session.ModelMessage` | package | `deskagents.llm.Message` | 0.1 |
| `deskagents.core.session.thaw_json` | package | `deskagents.util.values.thaw_json` | 0.1 |
| `deskagents.persistence.SessionStore` | package | `deskagents.session.persistence` (`SessionPersistence`) | 0.1 |
| `deskagents.persistence.MemorySessionStore` | pepgpt-plugin | PepGPT-owned in-memory `SessionPersistence` backend, checked with `deskagents.testing.persistence_contract` | 0.1 |
| `deskagents.persistence.DatabaseSessionStore` | pepgpt-plugin | PepGPT Postgres `SessionPersistence` backend (`deskagents.session.persistence`), checked with `deskagents.testing.run_persistence_contract` and `run_live_write_path_contract` | 0.1 |

## System prompt and tools

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.core.system_prompt.SystemPromptRegistry` | package | `deskagents.core.system_prompt` (`ctx.system_prompt`) | 0.1 |
| `deskagents.core.system_prompt.PromptSection` | package | `deskagents.core.system_prompt` | 0.1 |
| `deskagents.core.system_prompt.PromptRenderContext` | package | `deskagents.core.system_prompt` assembly context | 0.1 |
| `deskagents.core.tools.ToolDefinition` | package | `deskagents.core.tools` | 0.1 |
| `deskagents.core.tools.ToolExecutionContext` | package | `deskagents.core.tools` | 0.1 |
| `deskagents.core.tools.ToolExecutionMode` | package | `deskagents.core.tools` | 0.1 |
| `deskagents.core.tools.ToolExecutionResult` | package | `deskagents.core.tools` | 0.1 |
| `deskagents.core.tools.ToolExecutor` | package | `deskagents.core.tool_executor_native` | 0.1 |
| `deskagents.core.tools.ToolRegistry` | package | `deskagents.core.tools` (`ctx.tools`) | 0.1 |
| `deskagents.core.tools.visibility.restrict_tool_visibility` | package | `deskagents.core.agent_tool_presentation` | 0.1 |

## Agent and loop

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.core.agent.AgentConfig` | package | `deskagents.core.agent` | 0.1 |
| `deskagents.core.agent.AgentConstruction` | package | `deskagents.core.agent` (`ctx.agents.create`) | 0.1 |
| `deskagents.core.agent.AgentRegistry` | package | `deskagents.core.agent` (`ctx.agents`) | 0.1 |
| `deskagents.core.agent.AgentTurnRunner` | package | `deskagents.core.agent` | 0.1 |
| `deskagents.core.agent.MissingParentAgentError` | package | `deskagents.core.agent` | 0.1 |
| `deskagents.core.agent.RuntimeAgent` | package | `deskagents.core.agent` (`Agent`) | 0.1 |
| `deskagents.core.agent.UserInput` | package | `deskagents.core.agent` | 0.1 |
| `deskagents.core.agent.api.UserInput` | package | `deskagents.core.agent` | 0.1 |
| `deskagents.core.agent_loop.RecursiveToolAgentLoop` | package | `deskagents.core.agent_loop` (`AgentLoop`) | 0.1 |
| `deskagents.core.agent_loop.MODEL_REQUEST_ERROR` | package | `deskagents.llm.retry` / agent request events | 0.1 |
| `deskagents.core.agent_loop.ModelRequestErrorAction` | package | `deskagents.llm.retry` | 0.1 |
| `deskagents.core.agent_loop.ModelRequestFailed` | package | `deskagents.llm.retry` | 0.1 |
| `deskagents.core.agent_loop.ask_more_tool` | pepgpt-plugin | PepGPT tool on `ctx.tools` | — |
| `deskagents.core.agent_loop.handoff_tool` | pepgpt-plugin | PepGPT tool on `ctx.tools` | — |
| `deskagents.core.agent_loop.human_wait.current_turn_clock` | pepgpt-plugin | PepGPT turn-time accounting via the control plugin's user-question events | external |
| `deskagents.config.AgentBudgets` | package | agent-loop config; optional wall-clock enforcement comes from `deskagents-execution` | 0.1 |

## Model adapters

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.llm.LLMAdapter` | package | `deskagents.llm` adapter seam | 0.1 |
| `deskagents.llm.LLMRegistry` | package | `deskagents.llm` (`ctx.llm`) | 0.1 |
| `deskagents.llm.AdapterCapabilities` | package | `deskagents.llm` | 0.1 |
| `deskagents.llm.ModelCapacity` | package | `deskagents.llm` / `deskagents.llm.token_meter` | 0.1 |
| `deskagents.llm.ModelRequest` | package | `deskagents.llm` prepared call | 0.1 |
| `deskagents.llm.ModelResponse` | package | `deskagents.llm` | 0.1 |
| `deskagents.llm.ModelStreamDelta` | package | `deskagents.llm` stream chunks | 0.1 |
| `deskagents.llm.ModelStreamEvent` | package | `deskagents.llm` stream chunks | 0.1 |
| `deskagents.llm.FinishReason` | package | `deskagents.llm` | 0.1 |
| `deskagents.llm.LLMAdapterError` | package | `deskagents.llm` | 0.1 |
| `deskagents.llm.ContextOverflowError` | package | `deskagents.llm` | 0.1 |
| `deskagents.llm.is_context_overflow_error` | package | `deskagents.llm` | 0.1 |
| `deskagents.llm.LangChainChatModelAdapter` | pepgpt-plugin | PepGPT adapter on the `ctx.llm` seam, checked with `deskagents.testing.llm_adapter_contract` | 0.1 |

PepGPT's `BedrockClaudeAdapter` and `PepGenXGeminiAdapter` (in
`app/harness_agent/providers.py`) move onto the same seam and run the same
contract suite.

## Context management and compaction

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.context_management.CompactionPlugin` | external-plugin | `deskagents-control`: `deskagents.compaction.compaction_basic` | external |
| `deskagents.context_management.CompactionPolicy` | external-plugin | `deskagents-control`: compaction config | external |
| `deskagents.context_management.ModelContextSummarizer` | external-plugin | `deskagents-control`: `deskagents.compaction.compaction_basic` | external |
| `deskagents.context_management.ToolResultPruner` | external-plugin | `deskagents-control`: `deskagents.compaction.tool_result_pruner` | external |
| `deskagents.context_management.CharacterTokenEstimator` | package | `deskagents.llm.token_meter` | 0.1 |
| `deskagents.context_management.measure.estimate_request_usage` | package | `deskagents.llm.token_meter` | 0.1 |

## Filesystem, shell, attachments, sandbox

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.plugins.FilesystemPlugin` | external-plugin | `deskagents-execution`: `deskagents.fs.tool_fs` | external |
| `deskagents.shell.BashToolPlugin` | external-plugin | `deskagents-execution`: `deskagents.shell.tool_bash` | external |
| `deskagents.attachments.Attachment` | external-plugin | `deskagents-execution`: `deskagents.attachment.attachment` | external |
| `deskagents.attachments.AttachmentError` | external-plugin | `deskagents-execution`: `deskagents.attachment.attachment` | external |
| `deskagents.attachments.AttachmentsPlugin` | external-plugin | `deskagents-execution`: `deskagents.attachment.attachment_local` | external |
| `deskagents.attachments.SandboxFiles` | external-plugin | `deskagents-execution`: `deskagents.fs.fs` seam | external |
| `deskagents.attachments.SharedFile` | pepgpt-plugin | PepGPT file sharing over the execution plugin | external |
| `deskagents.attachments.shared_file_result` | pepgpt-plugin | same | external |
| `deskagents.attachments.safe_filename` | external-plugin | `deskagents-execution`: attachment utilities | external |
| `deskagents.code_interpreter.PythonToolPlugin` | pepgpt-plugin | PepGPT tool over the execution plugin's subprocess seam | external |
| `deskagents.integrations.azure_sessions.AzureSessionPool` | external-plugin | `deskagents-azure` | external |
| `deskagents.integrations.azure_sessions.AzureSessionPoolError` | external-plugin | `deskagents-azure` | external |
| `deskagents.integrations.azure_sessions.AzureShellSession` | external-plugin | `deskagents-azure` | external |
| `deskagents.integrations.azure_sessions.AzureShellExecutor` | external-plugin | `deskagents-azure` subprocess provider | external |
| `deskagents.integrations.azure_sessions.AzureShellFilesystem` | external-plugin | `deskagents-azure` filesystem provider | external |
| `deskagents.integrations.azure_sessions.AzureSandboxFiles` | external-plugin | `deskagents-azure` filesystem provider | external |
| `deskagents.integrations.azure_sessions.AzurePythonInterpreter` | pepgpt-plugin | PepGPT code interpreter on `deskagents-azure` | external |

## Web

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.web.WebRuntime` | external-plugin | `deskagents-web`: `deskagents.web.web` (`ctx.web`) | external |
| `deskagents.web.WebToolsPlugin` | external-plugin | `deskagents-web`: `deskagents.web.tool_web` | external |
| `deskagents.web.SafeHttpFetchProvider` | external-plugin | `deskagents-web`: `deskagents.web.web_fetch_http` | external |
| `deskagents.web.GoogleCustomSearchProvider` | pepgpt-plugin | PepGPT search provider on the web plugin's `ctx.web` seam | external |
| `deskagents.web.GoogleSearchPlugin` | pepgpt-plugin | same | external |

## Skills

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.skills.SkillDefinition` | external-plugin | `deskagents-skills`: `deskagents.skill.skill` | external |
| `deskagents.skills.SkillRuntime` | external-plugin | `deskagents-skills`: `deskagents.skill.skill` (`ctx.skills`) | external |
| `deskagents.skills.StaticSkillProvider` | external-plugin | `deskagents-skills`: provider registry | external |
| `deskagents.skills.render_skill_content` | external-plugin | `deskagents-skills`: `deskagents.skill.tool_skill` | external |
| `deskagents.skills.strip_skill_content` | pepgpt-plugin | PepGPT search-budget helper | external |
| `deskagents.plugins.AutoSkillsPlugin` | pepgpt-plugin | PepGPT plugin over the skills plugin's `ctx.skills` seam | external |

## Telemetry and durable tools

| PepGPT import | Status | Target (proposed) | Milestone |
|---|---|---|---|
| `deskagents.plugins.TelemetryPlugin` | external-plugin | `deskagents-product`: telemetry providers | external |
| `deskagents.plugins.ToolTelemetryPlugin` | external-plugin | same | external |
| `deskagents.plugins.TelemetryContext` | external-plugin | `deskagents-product`: telemetry context | external |
| `deskagents.plugins.bind_telemetry_context` | external-plugin | same | external |
| `deskagents.plugins.telemetry.TelemetryContext` | external-plugin | `deskagents-product`: telemetry context | external |
| `deskagents.plugins.telemetry.bind_telemetry_context` | external-plugin | same | external |
| `deskagents.plugins.DurableToolCallsPlugin` | pepgpt-plugin | PepGPT plugin on `tools/*` events | 0.1 |

## Memory (no upstream counterpart)

All stay in PepGPT as its own plugins. They can move to `deskagents.contrib.memory`
later if another team needs them.

| PepGPT import | Status |
|---|---|
| `deskagents.plugins.auto_memory.AutoMemoryPlugin` | pepgpt-plugin |
| `deskagents.memory.control.JobSnapshot`, `MemoryControlStore`, `MemoryJob`, `MemorySettings` | pepgpt-plugin |
| `deskagents.memory.markdown.MemoryDocumentError`, `MemoryDraft`, `MemorySummary`, `validate_memory_name` | pepgpt-plugin |
| `deskagents.memory.matching.best_match` | pepgpt-plugin |
| `deskagents.memory.privacy.redact_memory_input` | pepgpt-plugin |
| `deskagents.memory.prompt.render_memory_preview` | pepgpt-plugin |
| `deskagents.memory.store.MemoryConflictError`, `MemoryDocument`, `MemoryLimitError`, `MemoryStorageError`, `StoredMemory`, `UserMemoryStore`, `account_key` | pepgpt-plugin |

## PepGPT integrations

| PepGPT import | Status | Notes |
|---|---|---|
| `deskagents.integrations.PepGPTAgentUiAdapter` | pepgpt-app | rebuilt over `ctx.agents` and `session/event`; see `examples/fastapi_sse_server.py` |
| `deskagents.integrations.AgentUiError` | pepgpt-app | |
| `deskagents.integrations.SessionOwnership`, `MemorySessionOwnership` | pepgpt-app | |
| `deskagents.integrations.pepgpt_agent_ui.SessionOwnership` | pepgpt-app | |
| `deskagents.integrations.PersistentAgentProvider` | pepgpt-app | |
| `deskagents.integrations.pepgpt.PepGPTTurnContext` | pepgpt-app | |
| `deskagents.integrations.PepGPTCanaryAdapter`, `PepGPTCanaryPolicy` | pepgpt-app | |
| `deskagents.integrations.PepGPTPostgresSessionRepository` | pepgpt-plugin | becomes the Postgres `SessionPersistence` backend |
| `deskagents.integrations.pepgpt_telemetry.ModelTelemetryIdentity`, `PepGPTArizeTelemetrySink`, `PepGPTCanaryTraceProvider` | pepgpt-plugin | Arize listeners on telemetry events |
| `deskagents.integrations.pepgpt_tool_telemetry.PepGPTArizeToolTelemetrySink` | pepgpt-plugin | same |
| `deskagents.integrations.turn_relay.Handler`, `TurnRelay`, `question_channel` | pepgpt-app | cross-pod relay; answers go through `deskagents.interaction.user_questions` |

## Session event mapping

PepGPT event types and their v4 equivalents, so PepGPT can convert stored sessions.
Confirmed in Phase 2 against the pinned inventory (`tests/upstream_assets/persistence-schema.json`,
upstream `ce7073d`); `tests/support/persistence_schema.py` checks a converted event against it.

| PepGPT event | v4 | Conversion |
|---|---|---|
| `session/created` | not an event | v4 has no durable `session/created`; creation metadata lives in the session header (`version`, `id`, `createdAt`, `cwd`, `isSeeded`, …). Drop the event and build the header. |
| `session/end-seed` | `session/end-seed` | Same. A fork child's marker at its inherited cut carries `{"inherited": true}`. |
| `session/forked` | not an event | v4 records fork lineage as header `isSeeded: true` plus the inherited `session/end-seed` marker and the exact inherited prefix length. |
| `turn/start`, `turn/end` | same | `turn/end.reason.kind` must be one of `completed`, `aborted`, `blocked`, `error`, `max-tokens`, `interrupted`, `forked`. |
| `step/start`, `step/end` | same | |
| `system/message`, `developer/message`, `user/message` | same | Messages need `id`, `role`, `content` and a `source` whose `kind` is in v4's closed list (e.g. `user`, `system-prompt`, `tool-registry`, `runtime-context`); PepGPT's own producers need a listed kind to conform. |
| `assistant/message`, `assistant/attempt` | same | v4 requires the embedded compact `stream` array (may be empty) and `source: {kind: "model", provider, model}`. |
| `tool/call`, `tool/result` | same | `tool/result.message` needs `toolCallId` equal to `source.callId`; failure details go in `error: {name, code, reason?}` with `message.isError: true`. |
| `request/header`, `request/context` | same | Headers must omit empty `tools` and empty `adapterDefaults`, and never carry `header.system`. |
| `context/summary` | `compaction/start`, `compaction/summary`, `compaction/end` | Plus a surface `replace` of the summarized range. Exact shape confirmed when `compaction` is ported (0.3). |
| `agent/child-created`, `agent/lineage` | not events | v4 keeps subagent lineage in the child's header (`parentSession`, `origin: "subagent"`, `delegationDepth`) and the `subagent/*` events. Exact mapping confirmed when `subagent` is ported (0.4). |
