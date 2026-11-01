"""The public adapter contract is exercised against ``ScriptedAdapter``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from deepseek_cordis import Context

from deskagents.llm import LlmRuntime
from deskagents.testing import (
    AdapterContractBackend,
    AdapterContractFactory,
    ScriptedAdapter,
    llm_adapter_contract_cases,
    run_llm_adapter_contract,
)


def script() -> list[dict[str, object]]:
    return [
        {"type": "block-start", "index": 0, "blockType": "text"},
        {"type": "text-delta", "index": 0, "text": "hello"},
        {
            "type": "block-end",
            "index": 0,
            "block": {"type": "text", "text": "hello"},
        },
        {"type": "usage", "usage": {"inputTokens": 1, "outputTokens": 1}},
        {"type": "finish", "reason": {"kind": "stop"}},
    ]


async def factory() -> AdapterContractBackend:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    runtime = ctx.get("llm")
    assert isinstance(runtime, LlmRuntime)
    adapter = ScriptedAdapter([script()])
    runtime.register_adapter(["scripted"], adapter, ctx=ctx)
    return AdapterContractBackend(
        runtime=runtime,
        adapter=adapter,
        provider="scripted",
        model="scripted-model",
        options={
            "provider": "scripted",
            "model": "scripted-model",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
        dispose=ctx.fiber.dispose,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("title", "check"),
    llm_adapter_contract_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
async def test_scripted_adapter_contract(
    title: str,
    check: Callable[[AdapterContractFactory], Awaitable[None]],
) -> None:
    del title
    await check(factory)


@pytest.mark.asyncio
async def test_contract_runner_reports_all_passed() -> None:
    results = await run_llm_adapter_contract(factory)
    assert [(result.title, result.outcome) for result in results] == [
        ("resolves exact model identity", "passed"),
        ("emits an assemblable terminal stream", "passed"),
        ("dispatches prepared calls exactly once", "passed"),
    ]


@pytest.mark.asyncio
async def test_scripted_adapter_failure_is_a_terminal_chunk() -> None:
    ctx = Context()
    try:
        await ctx.plugin(LlmRuntime)
        runtime = ctx.get("llm")
        assert isinstance(runtime, LlmRuntime)
        adapter = ScriptedAdapter([RuntimeError("provider failed")])
        runtime.register_adapter(["scripted"], adapter, ctx=ctx)
        chunks = [
            chunk
            async for chunk in runtime.stream(
                {"provider": "scripted", "model": "model", "messages": []}
            )
        ]
        assert chunks == [
            {
                "type": "finish",
                "reason": {
                    "kind": "error",
                    "failure": {"message": "provider failed", "code": "UNKNOWN"},
                },
            }
        ]
    finally:
        await ctx.fiber.dispose()
