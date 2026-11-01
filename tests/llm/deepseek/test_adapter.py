from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from deepseek_cordis import Context

from deskagents.core.session import SessionStore
from deskagents.llm import LlmError, LlmRuntime, create_user_message
from deskagents.llm.deepseek import (
    DeepSeekAdapter,
    parse_sse,
    serialize_request,
    translate_events,
)
from deskagents.llm.deepseek_llm_api_extensions import DeepSeekLlmApiExtensionRegistry
from deskagents.session.log_deepseek import SessionLogDeepSeekService, accepted_through
from deskagents.testing import AdapterContractBackend, llm_adapter_contract_cases

FIXTURE = Path(__file__).parents[1] / "fixtures" / "deepseek-messages.sse"


def _transport(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/anthropic/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["user-agent"].startswith("deskagents/")
    body = json.loads(request.content)
    assert body["thinking"]["type"] == "enabled"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=FIXTURE.read_bytes(),
    )


async def _backend() -> AdapterContractBackend:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    client = httpx.AsyncClient(transport=httpx.MockTransport(_transport))
    adapter = DeepSeekAdapter(
        api_key="test-key",
        client=client,
        models=({"id": "model", "name": "Model"},),
    )
    runtime = ctx.get("llm")
    assert isinstance(runtime, LlmRuntime)
    runtime.register_adapter(["deepseek"], adapter)

    async def dispose() -> None:
        await client.aclose()
        await ctx.fiber.dispose()

    return AdapterContractBackend(
        runtime=runtime,
        adapter=adapter,
        provider="deepseek",
        model="model",
        options={
            "provider": "deepseek",
            "model": "model",
            "messages": [
                create_user_message(
                    content=[{"type": "text", "text": "hi"}],
                    source={"kind": "user"},
                )
            ],
        },
        dispose=dispose,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("title", "check"),
    llm_adapter_contract_cases(),
    ids=[title for title, _ in llm_adapter_contract_cases()],
)
async def test_adapter_contract(title: str, check: Any) -> None:
    del title
    await check(_backend)


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/llm/deepseek-llm-api-extensions/tests/registry.spec.ts::"
    "prepares detached fields and accepts every provider exactly once"
)
async def test_extension_registry_detaches_and_accepts_once() -> None:
    ctx = Context()
    await ctx.plugin(DeepSeekLlmApiExtensionRegistry)
    registry = ctx.get("deepseekLlmApiExtensions")
    assert isinstance(registry, DeepSeekLlmApiExtensionRegistry)
    source = {"nested": [1]}
    accepted = 0

    class Provider:
        def prepare(self, request: Any) -> Any:
            del request

            def accept() -> None:
                nonlocal accepted
                accepted += 1

            return {"value": source, "accept": accept}

    registry.register("field", Provider())
    prepared = await registry.prepare({})
    source["nested"].append(2)
    assert prepared.fields["field"]["nested"] == [1]
    await prepared.accept()
    await prepared.accept()
    assert accepted == 1
    await ctx.fiber.dispose()


@pytest.mark.upstream(
    "packages/llm/llm-deepseek/tests/serialize.spec.ts::"
    "disables thinking for the off effort, passes temperature with thinking and refuses "
    "unsupported effort"
)
def test_serialize_reasoning_tools_and_tool_results() -> None:
    body = serialize_request(
        {
            "model": "model",
            "reasoningEffort": "high",
            "messages": [
                {
                    "role": "assistant",
                    "source": {"kind": "model"},
                    "content": [
                        {
                            "type": "tool-call",
                            "id": "call",
                            "name": "lookup",
                            "arguments": '{"q":"x"}',
                        }
                    ],
                },
                {
                    "role": "tool",
                    "toolCallId": "call",
                    "content": [{"type": "text", "text": "ok"}],
                },
            ],
        }
    )
    assert body["output_config"] == {"effort": "high"}
    assert body["messages"][0]["content"][0]["type"] == "tool_use"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"
    assert serialize_request({"model": "model", "reasoningEffort": "off", "messages": []})[
        "thinking"
    ] == {"type": "disabled"}
    with pytest.raises(LlmError, match="reasoning effort"):
        serialize_request({"model": "model", "reasoningEffort": "medium", "messages": []})


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/llm/llm-deepseek/tests/stream.spec.ts::"
    "emits first-seen blocks, raw tool JSON and cumulative disjoint usage before finish"
)
async def test_recorded_stream_translation() -> None:
    async def lines() -> AsyncIterator[str]:
        for line in FIXTURE.read_text().splitlines():
            yield line

    chunks = [chunk async for chunk in translate_events(parse_sse(lines()), "model")]
    assert chunks[0] == {"type": "block-start", "index": 0, "blockType": "text"}
    assert chunks[-2]["usage"]["totalTokens"] == 3
    assert chunks[-1]["reason"] == {"kind": "stop"}


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/session/session-log-deepseek/tests/upload.spec.ts::"
    "uploads the full first prefix, records acceptance, then sends only the appended suffix"
)
async def test_session_log_extension_records_acceptance() -> None:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(DeepSeekLlmApiExtensionRegistry)
    await ctx.plugin(SessionLogDeepSeekService)
    sessions = ctx.get("sessions")
    extensions = ctx.get("deepseekLlmApiExtensions")
    assert isinstance(sessions, SessionStore)
    assert isinstance(extensions, DeepSeekLlmApiExtensionRegistry)
    session = sessions.create("log")
    session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "hello"}],
            source={"kind": "user"},
        ),
        surface_op="append",
    )
    prepared = await extensions.prepare({"body": {}, "sessionId": "log", "signal": None})
    assert prepared.fields["dsh_session_log"]["events"][0]["type"] == "user/message"
    await prepared.accept()
    assert accepted_through(session) == 0
    await ctx.fiber.dispose()
