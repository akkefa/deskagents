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
