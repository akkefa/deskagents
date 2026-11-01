# Model adapters

The `deskagents.llm` seam ships two optional HTTP providers. Neither extra is required for
the scripted/core runtime.

## DeepSeek Messages

Install `deskagents[deepseek]` and register
`deskagents.llm.deepseek.DeepSeekAdapter`. It supports system and user text, native
reasoning replay, tool calls/results, usage, stop/max-token outcomes, custom headers
and endpoint URLs. `deskagents.llm.deepseek_api_key` is the Cordis plugin form.

The optional `DeepSeekLlmApiExtensionRegistry` owns independent top-level request
fields. `SessionLogDeepSeekService` uses it to upload bounded event-log prefixes and
records accepted sequence watermarks in the session itself.

Attachment files and image projection remain in the separate attachment plugin family.

## OpenAI-compatible Chat Completions

Install `deskagents[openai]` and use `OpenAIAdapter`. Besides OpenAI, `base_url`
and `headers` support compatible gateways.

For Azure OpenAI, provide `azure_endpoint`, optionally `azure_deployment`, and
`api_version`. The adapter sends `api-key` authentication and constructs the deployment
Chat Completions URL.

Both adapters normalize authentication, rate-limit, quota, context-window and transport
failures to `LlmError`. The committed tests use recorded SSE responses and never access
the network.

## Configuration and compatibility

ID-only model declarations get a fallback name. DeepSeek exposes off/low/high/max
reasoning (default high); `thinking="disabled"` permits only off. A plugin may set
`reasoningEffort` as its default. Model `maxTokens` sets its default output cap.
For compatibility with the existing Python API, DeepSeek defaults to in-history
system/tool updates and an 8,192-token output cap; these are Python defaults,
not upstream's complete hosted model catalog. Override model capabilities for the
endpoint. Image/file projection and live model discovery remain external/deferred.

Native replay is accepted only for matching model identity and valid block-aligned
metadata. A model switch discards signatures while retaining durable content.
DeepSeek validates stream framing/order and reports cache-read/cache-write usage.
Malformed streams fail explicitly; truncated tool JSON is allowed only with the
max-token outcome so the shared assembler can handle it.

DeepSeek provider failures follow the pinned categories (`AUTH`, `QUOTA`,
`RATE_LIMIT`, `INVALID_REQUEST`, `CONTEXT_WINDOW_EXCEEDED`, `SERVER`, `HTTP_*`).
Retry-After and request identity are retained when supplied. This changes the old
DeepSeek `INVALID_CREDENTIAL`/`RATE_LIMITED` wire-error mapping. OpenAI-compatible
error mapping is unchanged. API-key validation still uses `INVALID_CREDENTIAL`.

Cancel a running agent with `agent.cancel()` and await `agent.when_idle()` before
disposal. The agent closes an active provider stream and preserves interrupted
visible output; started tools drain before settlement. Standalone adapter users
must cancel/close their consumer task or iterator themselves.

Live checks are opt-in via `scripts/live_provider_smoke.py`; see [publishing](publishing.md).
