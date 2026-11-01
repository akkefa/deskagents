"""Single-call and composite regressions from the pinned tool pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from core.test_tool_activation import setup
from core.test_tool_pipeline import context, kinds, returning, string_tool, tool_runtime

from deskagents.core.tools import NativeToolExecutor, ToolRunContext
from deskagents.util.abort import AbortController

SPEC = "packages/core/tools/tests/tools.spec.ts::ToolRuntime > "


@pytest.mark.asyncio
@pytest.mark.upstream(
    SPEC
    + "materializes a pre-aborted call and publishes one result without entering pipeline phases"
)
async def test_preaborted_single_call_materializes_and_notifies() -> None:
    ctx, tools = await tool_runtime()
    try:
        phases: list[str] = []
        observed: list[Any] = []
        tools.register(string_tool("echo", returning("body")))
        for event in ("tools/pre-execute", "tools/execute", "tools/post-execute"):
            ctx.on(event, lambda *args: phases.append("unexpected"))
        ctx.on("tools/result", lambda execution, result: observed.append((execution, result)))
        controller = AbortController()
        controller.abort()
        arguments = {"nested": {"value": 1}}
        result = await tools.execute_call(
            {"callId": "call", "name": "echo", "arguments": arguments, "signal": controller.signal}
        )
        assert phases == [] and len(observed) == 1
        assert result["error"]["info"]["code"] == "ABORTED_BEFORE_DISPATCH"
        assert observed[0][1] is result
        assert observed[0][0]["arguments"] is not arguments
        arguments["nested"]["value"] = 2
        assert observed[0][0]["arguments"]["nested"]["value"] == 1
        with pytest.raises(TypeError):
            result["isError"] = False
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(SPEC + "lets argument materialization failure win over a pre-aborted signal")
async def test_invalid_arguments_precede_cancellation() -> None:
    ctx, tools = await tool_runtime()
    try:
        observed: list[Any] = []
        ctx.on("tools/result", lambda *args: observed.append(args))
        controller = AbortController()
        controller.abort()
        result = await tools.execute_call(
            {
                "callId": "call",
                "name": "missing",
                "arguments": {"invalid": lambda: None},
                "signal": controller.signal,
            }
        )
        assert result["isError"] is True
        assert "losslessly JSON" in result["error"]["message"]
        assert len(observed) == 1
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    SPEC + "reports cancellation before an executor refusal without entering policy"
)
async def test_cancellation_precedes_executor_refusal() -> None:
    ctx, _ = await tool_runtime()
    try:
        phases: list[Any] = []
        ctx.on("tools/pre-execute", lambda *args: phases.append(args))
        tool = string_tool("echo", returning("body"))
        executor = NativeToolExecutor(
            ctx, source={"get": lambda *args: tool, "reject": lambda *args: RuntimeError("refused")}
        )
        controller = AbortController()
        controller.abort()
        result = await executor.execute_call(
            {"callId": "call", "name": "echo", "arguments": {}, "signal": controller.signal}
        )
        assert result["error"]["info"]["code"] == "ABORTED_BEFORE_DISPATCH"
        assert phases == []
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/activation.spec.ts::"
    "settles calls of a cancelled step through cancellation with their finalizers"
)
async def test_admitted_cancelled_call_runs_finalizer() -> None:
    h = await setup()
    try:
        h.tools.register(
            {**h.tool, "finalizeContent": lambda *_: [{"type": "text", "text": "finalized"}]},
            when=lambda _: True,
        )
        admitted = await h.resolve()
        h.controller.abort()
        result = await h.tools.execute_call(
            {
                "callId": "call",
                "name": "conditional",
                "arguments": {},
                "signal": h.controller.signal,
                "agent": h.agent,
                "toolSet": admitted,
            }
        )
        assert result["error"]["info"]["code"] == "ABORTED_BEFORE_DISPATCH"
        assert result["content"][0]["text"] == "finalized"
        assert h.calls == []
    finally:
        await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    SPEC + "projects presentation metadata for native and nested composite dispatches"
)
async def test_nested_presentation_metadata() -> None:
    ctx, tools = await tool_runtime()
    try:
        tool = string_tool("meta", returning("ok"))
        tool["output"]["presentationMeta"] = lambda *_: {"card": True}
        tools.register(tool)
        for identity in ({}, {"parent": object()}):
            result = await tools.execute_call(
                {"callId": "call", "name": "meta", "arguments": {}, **identity}
            )
            assert result["meta"] == {"card": True}
            assert result["value"] == "ok"
            with pytest.raises(TypeError):
                result["meta"]["card"] = False
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    SPEC + "carries a nested conclusion on the nested result for its composite to forward"
)
async def test_composite_forwards_only_successful_conclusion() -> None:
    ctx, tools = await tool_runtime()
    try:
        identities: list[Any] = []

        async def terminal(args: Any, execution: ToolRunContext) -> str:
            identities.append((execution["parent"], execution["rootCallId"]))
            execution.conclude_turn()
            return "terminal"

        async def composite(args: Any, execution: ToolRunContext) -> str:
            nested = await tools.execute_call(
                {"callId": "nested", "name": "terminal", "arguments": {}}, parent=execution
            )
            if nested.get("concludesTurn"):
                execution.conclude_turn()
            return "recovered" if nested["isError"] else "succeeded"

        tools.register(string_tool("terminal", terminal))
        tools.register(string_tool("composite", composite))

        async def post(execution: Any, result: Any, next_: Any) -> Any:
            if execution["name"] == "terminal":
                return {"kind": "block", "feedback": [{"type": "text", "text": "veto"}]}
            return await next_()

        veto = ctx.on("tools/post-execute", post)
        recovered = await tools.execute_call(
            {"callId": "root", "name": "composite", "arguments": {}}
        )
        assert recovered["value"] == "recovered" and "concludesTurn" not in recovered
        veto()
        # The nested call uses the same pipeline without acquiring a worker slot.
        batch = await asyncio.wait_for(
            tools.execute(
                {
                    "calls": [{"id": "root", "name": "composite", "arguments": "{}"}],
                    "signal": AbortController().signal,
                    "maxParallelToolCalls": 1,
                }
            ),
            2,
        )
        assert batch["concluded"] is True
        assert batch["results"][0]["result"]["value"] == "succeeded"
        assert all(root == "root" for _, root in identities)
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_composite_contexts_survive_body_error_and_are_dropped_by_block() -> None:
    ctx, tools = await tool_runtime()
    try:

        async def child(args: Any, execution: ToolRunContext) -> str:
            execution.defer_context(context("child", "context"))
            return "ok"

        async def outer(args: Any, execution: ToolRunContext) -> str:
            nested = await tools.execute_call(
                {"callId": "nested", "name": "child"}, parent=execution
            )
            for item in nested.get("additionalContexts", ()):
                execution.defer_context(item)
            raise ValueError("outer failed")

        tools.register(string_tool("child", child))
        tools.register(string_tool("outer", outer))
        assert kinds(await tools.execute_call({"callId": "root", "name": "outer"})) == ["child"]

        async def block(execution: Mapping[str, Any], result: Any, next_: Any) -> Any:
            if execution["name"] == "outer":
                return {"kind": "block", "feedback": [{"type": "text", "text": "blocked"}]}
            return await next_()

        ctx.on("tools/post-execute", block)
        assert kinds(await tools.execute_call({"callId": "root", "name": "outer"})) == []
    finally:
        await ctx.fiber.dispose()
