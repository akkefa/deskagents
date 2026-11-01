from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from deepseek_cordis import Context

from deskagents.llm import LlmRuntime, create_user_message
from deskagents.llm.openai import OpenAIAdapter, serialize_request, translate_sse
from deskagents.testing import AdapterContractBackend, llm_adapter_contract_cases

FIXTURE = Path(__file__).parents[1] / "fixtures" / "openai-chat.sse"


def _transport(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"].startswith("deskagents/")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=FIXTURE.read_bytes(),
    )


async def _backend() -> AdapterContractBackend:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    client = httpx.AsyncClient(transport=httpx.MockTransport(_transport))
    adapter = OpenAIAdapter(
        api_key="test-key",
        client=client,
        models=({"id": "model", "name": "Model"},),
    )
    runtime = ctx.get("llm")
    assert isinstance(runtime, LlmRuntime)
    runtime.register_adapter(["openai"], adapter)

    async def dispose() -> None:
        await client.aclose()
        await ctx.fiber.dispose()

    return AdapterContractBackend(
        runtime=runtime,
        adapter=adapter,
        provider="openai",
        model="model",
        options={
            "provider": "openai",
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


def test_serializes_tools_and_azure_independent_body() -> None:
    body = serialize_request(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool-call",
                            "id": "c",
                            "name": "lookup",
                            "arguments": '{"q":"x"}',
                        }
                    ],
                },
                {
                    "role": "tool",
                    "toolCallId": "c",
                    "content": [{"type": "text", "text": "ok"}],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object"},
                }
            ],
        }
    )
    assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert body["messages"][1]["tool_call_id"] == "c"


@pytest.mark.asyncio
async def test_translates_reasoning_and_tool_calls() -> None:
    async def lines() -> AsyncIterator[str]:
        yield "data: " + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "think",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        yield "data: " + json.dumps(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        yield "data: [DONE]"

    chunks = [chunk async for chunk in translate_sse(lines())]
    assert [chunk["type"] for chunk in chunks].count("tool-call-delta") == 2
    assert chunks[-1]["reason"] == {"kind": "tool-calls"}


def test_azure_request_uses_deployment_url_and_api_key_header() -> None:
    adapter = OpenAIAdapter(
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="deployment",
        api_version="2025-01-01-preview",
    )
    url, headers = adapter._request({"model": "ignored"})
    assert "/openai/deployments/deployment/chat/completions" in url
    assert "api-version=2025-01-01-preview" in url
    assert headers["api-key"] == "azure-key"
