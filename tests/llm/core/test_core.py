from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Mapping
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.llm import (
    AppIdentity,
    AssistantStreamAccumulator,
    BlockAssembler,
    LlmAdapter,
    LlmError,
    LlmRuntime,
    assert_usable_api_key,
    assistant_stream_first_token_time,
    attribution_headers,
    create_developer_message,
    expand_assistant_stream,
    is_context_window_exceeded_error,
    is_quota_exceeded_error,
    join_assistant_stream_text,
    normalize_api_key,
    resolve_retry_policy,
    user_agent,
)

SPEC = "packages/llm/llm"


@pytest.mark.upstream(f"{SPEC}/tests/attribution.spec.ts::renders a custom identity")
def test_provider_attribution_headers() -> None:
    identity = AppIdentity("example-runtime", "1.2.3", "https://example.test/runtime")
    assert user_agent(identity) == "example-runtime/1.2.3 (+https://example.test/runtime)"
    assert attribution_headers(identity) == {
        "user-agent": "example-runtime/1.2.3 (+https://example.test/runtime)"
    }


@pytest.mark.asyncio
async def test_model_providers_can_publish_catalog_updates() -> None:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    seen: list[str] = []

    def broken(provider: str) -> None:
        raise RuntimeError(f"cannot refresh {provider}")

    ctx.on("llm/models-updated", broken)
    ctx.on("llm/models-updated", seen.append)
    llm.models_updated("catalog")
    assert seen == ["catalog"]
    with pytest.raises(ValueError, match="non-empty"):
        llm.models_updated("")
    await ctx.fiber.dispose()


@pytest.mark.upstream(f"{SPEC}/tests/api-key.spec.ts::normalizeApiKey")
def test_api_key_normalization_and_safe_diagnostic() -> None:
    assert normalize_api_key("  sk-abc\t\n") == {"ok": True, "value": "sk-abc"}
    assert normalize_api_key("sk-abc def") == {
        "ok": False,
        "reason": "illegalCharacters",
    }
    with pytest.raises(LlmError, match="ACME_API_KEY") as caught:
        assert_usable_api_key("sk-\N{GRINNING FACE}supersecret", "llm-test", "ACME_API_KEY")
    assert caught.value.code == "INVALID_CREDENTIAL"
    assert "supersecret" not in str(caught.value)


@pytest.mark.upstream(f"{SPEC}/tests/retry-policy.spec.ts::resolves immutable normal defaults")
def test_retry_policy_defaults_and_validation() -> None:
    policy = resolve_retry_policy(None, "provider.retryPolicy")
    assert policy == {
        "mode": "normal",
        "maxRetries": 5,
        "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TIMEOUT", "TRANSPORT"],
        "initialDelayMs": 500,
        "maxDelayMs": 10_000,
        "jitterRatio": 0.1,
    }
    with pytest.raises(ValueError, match="duplicates"):
        resolve_retry_policy(
            {"mode": "normal", "retryableCodes": ["SERVER", "SERVER"]},
            "provider.retryPolicy",
        )


@pytest.mark.upstream(
    f"{SPEC}/tests/service.spec.ts::"
    "recognizes structured and model-capacity context-window overflow details"
)
def test_context_and_quota_classifiers() -> None:
    assert is_context_window_exceeded_error("input exceeds the model context window limit")
    assert not is_context_window_exceeded_error("temperature exceeds maximum allowed value")
    assert is_quota_exceeded_error("You exceeded your current quota")
    assert not is_quota_exceeded_error("HTTP 429: rate limit reached")


@pytest.mark.upstream(
    f"{SPEC}/tests/assembler.spec.ts::assembles interleaved text, reasoning, and tool-call deltas"
)
def test_block_assembler_and_max_token_tool_drop() -> None:
    assembler = BlockAssembler()
    assembler.push({"type": "reasoning-delta", "index": 0, "text": "thinking"})
    assembler.push({"type": "text-delta", "index": 1, "text": "Hello"})
    assembler.push(
        {
            "type": "tool-call-delta",
            "index": 2,
            "id": "call-1",
            "name": "echo",
            "argumentsDelta": "{}",
        }
    )
    assembler.push({"type": "finish", "reason": {"kind": "max-tokens"}})
    assert assembler.blocks() == [
        {"type": "reasoning", "text": "thinking"},
        {"type": "text", "text": "Hello"},
    ]


@pytest.mark.upstream(
    f"{SPEC}/tests/assistant-stream.spec.ts::"
    "keeps delta boundaries and timestamps while compacting one attempt"
)
def test_assistant_stream_round_trip_and_queries() -> None:
    accumulator = AssistantStreamAccumulator()
    source = [
        {"time": 10, "chunk": {"type": "text-delta", "index": 0, "text": ""}},
        {"time": 13, "chunk": {"type": "text-delta", "index": 0, "text": "hi"}},
        {"time": 20, "chunk": {"type": "finish", "reason": {"kind": "stop"}}},
    ]
    for item in source:
        accumulator.push(item)
    packed = accumulator.snapshot()
    assert expand_assistant_stream(packed) == source
    assert assistant_stream_first_token_time(packed) == 13
    assert join_assistant_stream_text(packed) == "hi"


class _ScriptedAdapter(LlmAdapter):
    def __init__(self, *, failure: Exception | None = None, tool_update: str | None = None) -> None:
        self.failure = failure
        self.tool_update = tool_update
        self.seen: Mapping[str, Any] | None = None

    async def resolve_model(
        self, provider: str, model: str, signal: object = None
    ) -> Mapping[str, Any]:
        del signal
        info = {
            "provider": provider,
            "id": model,
            "name": model,
            "defaultMaxTokens": 256,
        }
        if self.tool_update is not None:
            info["toolUpdate"] = self.tool_update
        return info

    async def _generate(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        self.seen = options
        if self.failure is not None:
            raise self.failure
        yield {"type": "finish", "reason": {"kind": "stop"}}

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        return self._generate(options)


async def _collect(stream: AsyncIterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}/tests/service.spec.ts::"
    "prepares a one-shot registration-bound call and rejects config drift"
)
async def test_runtime_preparation_waterfall_and_failure_normalization() -> None:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    runtime = ctx.get("llm")
    assert isinstance(runtime, LlmRuntime)
    adapter = _ScriptedAdapter()
    runtime.register_adapter(["route"], adapter)
    wrapped: list[str] = []

    def middleware(
        options: Mapping[str, Any],
        next_: Callable[[], AsyncIterable[Mapping[str, Any]]],
    ) -> AsyncIterable[Mapping[str, Any]]:
        wrapped.append(options["provider"])
        return next_()

    ctx.on("llm/stream", middleware)
    prepared = await runtime.prepare_call({"provider": "route", "model": "model"})
    assert prepared.config["maxTokens"] == 256
    assert prepared.adapter_defaults == {"maxTokens": True}
    assert await _collect(prepared.stream({**prepared.config, "messages": []})) == [
        {"type": "finish", "reason": {"kind": "stop"}}
    ]
    assert wrapped == ["route"]
    with pytest.raises(LlmError) as reused:
        prepared.stream({**prepared.config, "messages": []})
    assert reused.value.code == "INVALID_PREPARED_CALL"

    failed = _ScriptedAdapter(failure=LlmError("busy", "RATE_LIMIT", status=429))
    runtime.register_adapter(["failed"], failed)
    chunks = await _collect(
        runtime.stream({"provider": "failed", "model": "model", "messages": []})
    )
    assert chunks[-1] == {
        "type": "finish",
        "reason": {
            "kind": "error",
            "failure": {"message": "busy", "code": "RATE_LIMIT", "status": 429},
        },
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_runtime_projects_tool_updates_before_dispatch() -> None:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    runtime = ctx.get("llm")
    assert isinstance(runtime, LlmRuntime)
    adapter = _ScriptedAdapter(tool_update="in-history")
    runtime.register_adapter(["route"], adapter)
    search = {"name": "search", "description": "Search", "parameters": {}}
    fetch = {"name": "fetch", "description": "Fetch", "parameters": {}}
    added = create_developer_message(
        source={"kind": "test"},
        content=[{"type": "tool-addition", "toolName": "fetch"}],
    )
    await _collect(
        runtime.stream(
            {
                "provider": "route",
                "model": "model",
                "messages": [added],
                "tools": [search, fetch],
                "toolHistory": {
                    "tools": [search],
                    "updates": [{"messageId": added["id"], "additions": [fetch]}],
                },
            }
        )
    )
    assert adapter.seen is not None
    assert adapter.seen["tools"] == [search, {**fetch, "deferLoading": True}]
    assert adapter.seen["messages"] == [added]
    await ctx.fiber.dispose()
