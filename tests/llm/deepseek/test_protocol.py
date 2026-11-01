"""Pinned Messages wire semantics and end-to-end adapter dispatch."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from deepseek_cordis import Context

from deskagents.llm import LlmError, LlmRuntime
from deskagents.llm.deepseek import DeepSeekAdapter, parse_sse, serialize_request, translate_events
from deskagents.llm.deepseek.transport import provider_error

SERIALIZE = "packages/llm/llm-deepseek/tests/serialize.spec.ts::"
STREAM = "packages/llm/llm-deepseek/tests/stream.spec.ts::"
START: dict[str, Any] = {"type": "message_start", "message": {"usage": {"input_tokens": 1}}}
TEXT: dict[str, Any] = {
    "type": "content_block_start",
    "index": 0,
    "content_block": {"type": "text", "text": "hello"},
}
STOP: dict[str, Any] = {"type": "content_block_stop", "index": 0}
END: list[dict[str, Any]] = [
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    {"type": "message_stop"},
]


async def events(values: list[Any]) -> AsyncIterator[Any]:
    for value in values:
        yield value


def sse(values: list[Any]) -> str:
    return "".join(f"data: {json.dumps(value)}\n\n" for value in values)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values",
    [
        [START, START],
        [TEXT],
        [START, {**TEXT, "index": -1}],
        [START, {**TEXT, "index": True}],
        [START, {**TEXT, "index": 0.5}],
        [START, TEXT, TEXT],
        [START, STOP],
        [START, TEXT, STOP, STOP],
        [START, {**TEXT, "content_block": {"type": "text", "text": 2}}],
        [
            START,
            TEXT,
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "bad"},
            },
        ],
        [START, {"type": "message_stop"}],
        [START, TEXT, *END],
        [START, {"type": "message_delta", "delta": {}, "usage": {"input_tokens": -1}}],
        [START, END[0], TEXT],
    ],
)
async def test_malformed_streams_fail_closed(values: list[Any]) -> None:
    with pytest.raises(LlmError) as caught:
        _ = [chunk async for chunk in translate_events(events(values), "model")]
    assert caught.value.code == "MALFORMED_RESPONSE"


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{", "[]", "null", "NaN"])
async def test_tool_json_validation_and_truncation(arguments: str) -> None:
    values = [
        START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "call", "name": "lookup", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": arguments},
        },
        STOP,
    ]
    with pytest.raises(LlmError) as caught:
        _ = [chunk async for chunk in translate_events(events([*values, *END]), "model")]
    assert caught.value.code == "MALFORMED_RESPONSE"
    truncated = [
        *values,
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
        END[-1],
    ]
    chunks = [chunk async for chunk in translate_events(events(truncated), "model")]
    assert chunks[-1]["reason"] == {"kind": "max-tokens"}


@pytest.mark.asyncio
@pytest.mark.upstream(
    STREAM + "SSE framing and provider failures > does not flush an unterminated terminal event"
)
async def test_unterminated_sse_tail_is_not_dispatched() -> None:
    with pytest.raises(LlmError) as caught:
        _ = [
            chunk
            async for chunk in translate_events(
                parse_sse(events(sse([START, TEXT, STOP, *END]).rstrip().splitlines())), "model"
            )
        ]
    assert caught.value.code == "STREAM_CLOSED"


@pytest.mark.asyncio
async def test_multiline_sse_and_heartbeat() -> None:
    frames = ["\ufeff: heartbeat", "", "event: ping", 'data: {"type":', 'data: "ping"}', ""]
    assert [event async for event in parse_sse(events(frames))] == [{"type": "ping"}]


@pytest.mark.asyncio
@pytest.mark.upstream(STREAM + "SSE framing and provider failures > normalizes in-band overloads")
async def test_inband_overload_is_retryable_server_error() -> None:
    with pytest.raises(LlmError) as caught:
        _ = [
            chunk
            async for chunk in translate_events(
                events(
                    [{"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}]
                ),
                "model",
            )
        ]
    assert caught.value.code == "SERVER"


@pytest.mark.upstream(
    STREAM + "SSE framing and provider failures > retains request identity and valid Retry-After "
    "without inventing missing counters"
)
def test_provider_retry_metadata() -> None:
    error = provider_error(None, 429, {"Retry-After": "2", "x-request-id": "request"})
    assert error.code == "RATE_LIMIT"
    assert error.failure["providerRetryAfterMs"] == 2000
    assert error.failure["requestId"] == "request"
    assert "providerRetryAfterMs" not in provider_error(None, 500, {"retry-after": "bad"}).failure


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "AUTH"),
        (403, "AUTH"),
        (402, "QUOTA"),
        (429, "RATE_LIMIT"),
        (400, "INVALID_REQUEST"),
        (413, "INVALID_REQUEST"),
        (503, "SERVER"),
        (404, "HTTP_404"),
    ],
)
def test_http_error_codes(status: int, code: str) -> None:
    assert provider_error({}, status).code == code


@pytest.mark.upstream(
    SERIALIZE + "omits saved empty assistant text without changing tool ids or thinking signatures"
)
def test_empty_text_keeps_replay_alignment() -> None:
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}, {"type": "reasoning", "text": "think"}],
        "source": {
            "kind": "model",
            "model": "model",
            "replayState": {
                "response": {"kind": "deepseek-messages", "version": 1, "model": "model"},
                "blocks": [{"type": "text"}, {"type": "reasoning", "signature": "signed"}],
            },
        },
    }
    body = serialize_request({"model": "model", "messages": [message]})
    assert body["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "think", "signature": "signed"}
    ]


@pytest.mark.parametrize(
    "field,value", [("kind", "foreign"), ("version", 2), ("version", True), ("model", "other")]
)
def test_bad_replay_envelope_degrades(field: str, value: Any) -> None:
    response = {"kind": "deepseek-messages", "version": 1, "model": "model", field: value}
    body = serialize_request(
        {
            "model": "model",
            "messages": [
                {
                    "role": "assistant",
                    "source": {
                        "kind": "model",
                        "model": "model",
                        "replayState": {
                            "response": response,
                            "blocks": [{"type": "reasoning", "signature": "secret"}],
                        },
                    },
                    "content": [{"type": "reasoning", "text": "think"}],
                }
            ],
        }
    )
    assert "signature" not in body["messages"][0]["content"][0]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "toolCallId": "missing", "content": []}],
        [
            {
                "role": "assistant",
                "content": [{"type": "tool-call", "id": "c", "name": "lookup", "arguments": "{}"}],
            }
        ],
        [
            {
                "role": "assistant",
                "content": [{"type": "tool-call", "id": "c", "name": "lookup", "arguments": "{}"}]
                * 2,
            }
        ],
    ],
)
def test_unresolved_or_duplicate_tools_are_invalid(messages: list[Any]) -> None:
    with pytest.raises(LlmError) as caught:
        serialize_request({"model": "model", "messages": messages})
    assert caught.value.code == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_runtime_sends_tool_and_prompt_updates_with_model_switch() -> None:
    ctx = Context()
    bodies: list[Any] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, text=sse([START, TEXT, STOP, *END]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        try:
            await ctx.plugin(LlmRuntime)
            llm = ctx.get("llm")
            assert isinstance(llm, LlmRuntime)
            llm.register_adapter(
                ["deepseek"],
                DeepSeekAdapter(api_key="test", client=client, models=({"id": "a"}, {"id": "b"})),
            )
            tool = {"name": "lookup", "description": "lookup", "parameters": {"type": "object"}}
            history: list[dict[str, Any]] = [
                {"role": "system", "content": [{"type": "text", "text": "OLD"}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
            first = [
                chunk
                async for chunk in llm.stream(
                    {"provider": "deepseek", "model": "a", "messages": history, "tools": []}
                )
            ]
            assert first[-1]["reason"]["kind"] == "stop"
            history += [
                {
                    "role": "assistant",
                    "source": {
                        "kind": "model",
                        "provider": "deepseek",
                        "model": "a",
                        "replayState": {
                            "response": {"kind": "deepseek-messages", "version": 1, "model": "a"},
                            "blocks": [{"type": "reasoning", "signature": "old"}],
                        },
                    },
                    "content": [{"type": "reasoning", "text": "thought"}],
                },
                {"role": "system", "content": [{"type": "text", "text": "NEW"}]},
                {
                    "role": "developer",
                    "id": "change",
                    "content": [{"type": "tool-addition", "toolName": "lookup"}],
                },
                {"role": "user", "content": [{"type": "text", "text": "continue"}]},
            ]
            second = [
                chunk
                async for chunk in llm.stream(
                    {
                        "provider": "deepseek",
                        "model": "b",
                        "messages": history,
                        "tools": [tool],
                        "toolHistory": {
                            "tools": [],
                            "updates": [
                                {"messageId": "change", "additions": [tool], "removals": []}
                            ],
                        },
                    }
                )
            ]
            assert second[-1]["reason"]["kind"] == "stop"
            assert bodies[1]["system"] == "OLD"
            assert bodies[1]["messages"][-1]["content"][0]["type"] == "tool_addition"
            assert bodies[1]["messages"][-2]["content"][0]["text"] == "NEW"
            assert "signature" not in bodies[1]["messages"][1]["content"][0]
        finally:
            await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_cancelled_http_consumer_closes_stream() -> None:
    entered, closed = asyncio.Event(), asyncio.Event()

    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield sse([START, TEXT]).encode()
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            closed.set()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stalled()))
    ) as client:
        adapter = DeepSeekAdapter(api_key="test", client=client)

        async def consume() -> None:
            async for _ in adapter.stream({"model": "model", "messages": []}):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()


@pytest.mark.asyncio
async def test_agent_cancellation_closes_stalled_transport() -> None:
    from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader
    from deskagents.core import AgentRegistry
    from deskagents.llm import create_user_message

    entered, closed = asyncio.Event(), asyncio.Event()

    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield sse([START, TEXT]).encode()
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            closed.set()

    ctx = Context()
    loader = BundleLoader(ctx)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stalled()))
    ) as client:
        try:
            await loader.load_file(MINIMAL_BUNDLE)
            llm = ctx.get("llm")
            assert isinstance(llm, LlmRuntime)
            llm.register_adapter(["deepseek"], DeepSeekAdapter(api_key="test", client=client))
            agents = ctx.get("agents")
            assert isinstance(agents, AgentRegistry)
            handle = await agents.create(
                {
                    "sessionId": "cancel-http",
                    "agentOptions": {"provider": "deepseek", "model": "model"},
                }
            )
            try:
                handle.agent.followup(
                    create_user_message(
                        content=[{"type": "text", "text": "hi"}], source={"kind": "user"}
                    )
                )
                await asyncio.wait_for(entered.wait(), 2)
                handle.agent.cancel()
                await asyncio.wait_for(handle.agent.when_idle(), 2)
                assert closed.is_set()
                end = [
                    e for e in handle.agent.session.snapshot_events() if e["type"] == "turn/end"
                ][-1]
                assert end["data"]["reason"]["kind"] == "aborted"
                assert any(m["role"] == "assistant" for m in handle.agent.session.derive_messages())
            finally:
                await handle.dispose()
        finally:
            await loader.dispose()
            await ctx.fiber.dispose()
