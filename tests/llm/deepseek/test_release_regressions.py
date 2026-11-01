"""Open release defects DS-01..08; see docs/release-contract.md.

Regression cases for the eight defects found in the release audit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.llm import LlmError, LlmRuntime
from deskagents.llm.deepseek import DeepSeekAdapter, serialize_request, translate_events


def test_dynamic_tool_addition_serializes() -> None:
    body = serialize_request(
        {
            "model": "model",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "developer",
                    "content": [{"type": "tool-addition", "toolName": "lookup"}],
                },
            ],
            "tools": [{"name": "lookup", "description": "Lookup", "parameters": {}}],
        }
    )
    assert body["messages"][-1] == {
        "role": "system",
        "content": [
            {
                "type": "tool_addition",
                "tool": {"type": "tool_reference", "name": "lookup"},
            }
        ],
    }


def test_system_update_preserves_history_position() -> None:
    body = serialize_request(
        {
            "model": "model",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "OLD"}]},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "system", "content": [{"type": "text", "text": "NEW"}]},
                {"role": "user", "content": [{"type": "text", "text": "continue"}]},
            ],
        }
    )
    assert body["system"] == "OLD"
    assert body["messages"][-2]["role"] == "user"
    assert body["messages"][-1] == {
        "role": "system",
        "content": [{"type": "text", "text": "NEW"}],
    }


def test_cross_model_replay_discards_signature() -> None:
    body = serialize_request(
        {
            "model": "new-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "reasoning", "text": "thought"}],
                    "source": {
                        "kind": "model",
                        "provider": "deepseek",
                        "model": "old-model",
                        "replayState": {
                            "response": {
                                "kind": "deepseek-messages",
                                "version": 1,
                                "model": "old-model",
                            },
                            "blocks": [{"type": "reasoning", "signature": "old-signature"}],
                        },
                    },
                }
            ],
        }
    )
    assert body["messages"][0]["content"] == [{"type": "thinking", "thinking": "thought"}]


@pytest.mark.asyncio
async def test_id_only_model_supports_reasoning_effort() -> None:
    ctx = Context()
    try:
        await ctx.plugin(LlmRuntime)
        llm = ctx.get("llm")
        assert isinstance(llm, LlmRuntime)
        llm.register_adapter(
            ["deepseek"], DeepSeekAdapter(api_key="test-key", models=({"id": "model"},))
        )
        config = await llm.resolve_call_config(
            {"provider": "deepseek", "model": "model", "reasoningEffort": "high"}
        )
        assert config["reasoningEffort"] == "high"
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_id_only_model_can_be_listed() -> None:
    ctx = Context()
    try:
        await ctx.plugin(LlmRuntime)
        llm = ctx.get("llm")
        assert isinstance(llm, LlmRuntime)
        llm.register_adapter(
            ["deepseek"], DeepSeekAdapter(api_key="test-key", models=({"id": "model"},))
        )
        models = await llm.list_models("deepseek")
        assert len(models) == 1
        assert models[0]["id"] == "model"
        assert models[0]["name"] == "model"
    finally:
        await ctx.fiber.dispose()


def test_empty_assistant_text_is_skipped() -> None:
    body = serialize_request(
        {
            "model": "model",
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
        }
    )
    assert body["messages"] == []


async def _events(values: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_stream_without_start_is_rejected() -> None:
    events = _events(
        [
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
    )
    try:
        chunks = [chunk async for chunk in translate_events(events, "model")]
    except LlmError as error:
        assert error.code == "MALFORMED_RESPONSE"
    else:
        assert not chunks, "malformed stream emitted a successful completion"
        pytest.fail("malformed stream must raise LlmError")


@pytest.mark.asyncio
async def test_cache_tokens_are_preserved_in_usage() -> None:
    events = _events(
        [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 7,
                        "cache_creation_input_tokens": 3,
                    }
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "ok"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 4},
            },
            {"type": "message_stop"},
        ]
    )
    chunks = [chunk async for chunk in translate_events(events, "model")]
    usage = next(chunk["usage"] for chunk in chunks if chunk["type"] == "usage")
    assert usage == {
        "inputTokens": 10,
        "outputTokens": 4,
        "cacheReadTokens": 7,
        "cacheWriteTokens": 3,
        "totalTokens": 24,
    }
