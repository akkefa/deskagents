"""Ported from packages/core/tools/tests/tools.spec.ts: one call through the pipeline stages."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import SystemPrompt, ToolsService, define_tool
from deskagents.core.tools import TOOL_ABORTED, TOOL_ABORTED_BEFORE_DISPATCH, NativeToolExecutor
from deskagents.llm import HarnessError, create_user_message
from deskagents.util.abort import AbortController

SPEC = "packages/core/tools/tests/tools.spec.ts"
RUNTIME = f"{SPEC}::ToolRuntime > "
ABORTED = {"name": "AbortError", "code": TOOL_ABORTED}
ABORTED_BEFORE_DISPATCH = {"name": "AbortError", "code": TOOL_ABORTED_BEFORE_DISPATCH}

Execute = Callable[[Mapping[str, Any], Any], Any]


async def tool_runtime() -> tuple[Context, ToolsService]:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    return ctx, tools


def string_tool(name: str, execute: Execute, **extra: Any) -> dict[str, Any]:
    return define_tool(
        name=name,
        description=name,
        parameters={},
        output={
            "schema": {"type": "string"},
            "render": lambda args, value: [{"type": "text", "text": value}],
        },
        execute=execute,
        **extra,
    )


def returning(value: str) -> Execute:
    async def execute(args: Mapping[str, Any], execution: object) -> str:
        return value

    return execute


def context(kind: str, text: str) -> Mapping[str, Any]:
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": kind})


def kinds(result: Mapping[str, Any]) -> list[str]:
    return [item["source"]["kind"] for item in result.get("additionalContexts", ())]


async def execute_one(tools: ToolsService, name: str, signal: object = None) -> Mapping[str, Any]:
    batch = await tools.execute(
        {
            "calls": [{"id": "c1", "name": name, "arguments": "{}"}],
            "signal": signal if signal is not None else AbortController().signal,
        }
    )
    result: Mapping[str, Any] = batch["results"][0]["result"]
    return result


def replace_signal(replacement: object) -> Callable[..., Any]:
    async def around(execution: Any, next_: Callable[[], Any]) -> Any:
        upstream = execution.signal
        execution.signal = replacement
        try:
            return await next_()
        finally:
            execution.signal = upstream

    return around


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}skips dispatch when an around wrapper supplies an already-aborted signal"
)
async def test_wrapper_supplied_aborted_signal_skips_dispatch() -> None:
    ctx, tools = await tool_runtime()
    dispatched = 0

    async def execute(args: Mapping[str, Any], execution: object) -> str:
        nonlocal dispatched
        dispatched += 1
        return ""

    tools.register(string_tool("must-not-run", execute))
    replacement = AbortController()
    replacement.abort("wrapper cancelled")
    ctx.on("tools/execute", replace_signal(replacement.signal))
    result = await execute_one(tools, "must-not-run")
    assert result["error"] == {
        "message": "tool call aborted before dispatch",
        "info": ABORTED_BEFORE_DISPATCH,
    }
    assert dispatched == 0
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}uses ABORTED_BEFORE_DISPATCH when cancellation overtakes a wrapper short-circuit"
)
async def test_cancelled_wrapper_short_circuit_is_aborted_before_dispatch() -> None:
    ctx, tools = await tool_runtime()
    dispatched = 0

    async def execute(args: Mapping[str, Any], execution: object) -> str:
        nonlocal dispatched
        dispatched += 1
        return ""

    tools.register(string_tool("short-circuited", execute))
    entered, release = asyncio.Event(), asyncio.Event()

    async def around(execution: object, next_: Callable[[], Any]) -> Mapping[str, Any]:
        entered.set()
        await release.wait()
        return {
            "value": "wrapper success",
            "content": [{"type": "text", "text": "wrapper success"}],
            "isError": False,
            "additionalContexts": [context("wrapper", "wrapper context")],
        }

    ctx.on("tools/execute", around)
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "short-circuited", caller.signal))
    await entered.wait()
    caller.abort("cancelled while wrapper waited")
    release.set()
    result = await pending
    assert result["content"] == [
        {"type": "text", "text": "Error: tool call aborted before dispatch"}
    ]
    assert result["isError"] is True
    assert result["error"]["info"] == ABORTED_BEFORE_DISPATCH
    assert kinds(result) == ["wrapper"]
    assert dispatched == 0
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}replaces a late wrapper success with ABORTED and preserves deferred contexts"
)
async def test_late_wrapper_success_becomes_aborted_with_deferred_contexts() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: Mapping[str, Any], execution: Any) -> str:
        execution.defer_context(context("child", "completed child work"))
        return "body complete"

    tools.register(string_tool("completed-before-wrapper", execute))
    entered, release = asyncio.Event(), asyncio.Event()

    async def around(execution: object, next_: Callable[[], Any]) -> Any:
        result = await next_()
        entered.set()
        await release.wait()
        return result

    ctx.on("tools/execute", around)
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "completed-before-wrapper", caller.signal))
    await entered.wait()
    caller.abort("cancelled while wrapper settled")
    release.set()
    result = await pending
    assert result["content"] == [{"type": "text", "text": "Error: tool call aborted"}]
    assert result["error"]["info"] == ABORTED
    assert kinds(result) == ["child"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}replaces a late post-execute success with ABORTED and preserves contexts"
)
async def test_late_post_execute_success_becomes_aborted_with_contexts() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: Mapping[str, Any], execution: Any) -> str:
        execution.defer_context(context("child", "completed child work"))
        return "body complete"

    tools.register(string_tool("completed-before-post", execute))
    entered, release = asyncio.Event(), asyncio.Event()

    async def post(
        execution: object, result: Mapping[str, Any], next_: Callable[[], Any]
    ) -> Mapping[str, Any]:
        decision = await next_()
        entered.set()
        await release.wait()
        return {**decision, "additionalContexts": [context("post", "post context")]}

    ctx.on("tools/post-execute", post)
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "completed-before-post", caller.signal))
    await entered.wait()
    caller.abort("cancelled while post policy waits")
    release.set()
    result = await pending
    assert result["content"] == [{"type": "text", "text": "Error: tool call aborted"}]
    assert result["error"]["info"] == ABORTED
    assert kinds(result) == ["child", "post"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}fuses caller cancellation back into a wrapper replacement for the running body"
)
async def test_caller_cancellation_reaches_a_body_under_a_replaced_signal() -> None:
    ctx, tools = await tool_runtime()
    entered = asyncio.Event()
    body_signals: list[Any] = []

    async def execute(args: Mapping[str, Any], execution: Any) -> str:
        body_signals.append(execution.signal)
        entered.set()
        await execution.signal.wait()
        return "stopped"

    tools.register(string_tool("cooperative", execute))
    replacement = AbortController()
    ctx.on("tools/execute", replace_signal(replacement.signal))
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "cooperative", caller.signal))
    await entered.wait()
    assert body_signals[0] is not caller.signal
    assert body_signals[0] is not replacement.signal
    caller.abort("cancel running body")
    result = await pending
    assert result["isError"] is True
    assert result["error"]["info"] == ABORTED
    assert body_signals[0].aborted
    assert not replacement.signal.aborted
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}restores the required caller signal after around dispatch")
async def test_post_execute_sees_the_caller_signal_again() -> None:
    ctx, tools = await tool_runtime()
    tools.register(string_tool("echo", returning("")))
    ctx.on("tools/execute", replace_signal(AbortController().signal))
    post_signals: list[object] = []

    def post(execution: Any, result: Mapping[str, Any], next_: Callable[[], Any]) -> Any:
        post_signals.append(execution.signal)
        return next_()

    ctx.on("tools/post-execute", post)
    caller = AbortController()
    await execute_one(tools, "echo", caller.signal)
    assert post_signals == [caller.signal]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}waits for an uncooperative started body before returning ABORTED")
async def test_cancellation_drains_an_uncooperative_body() -> None:
    ctx, tools = await tool_runtime()
    entered = asyncio.Event()
    release: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def execute(args: Mapping[str, Any], execution: Any) -> str:
        execution.defer_context(context("nested", "nested outcome"))
        entered.set()
        return await release

    tools.register(string_tool("uncooperative", execute))
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "uncooperative", caller.signal))
    await entered.wait()
    caller.abort("must still drain")
    await asyncio.sleep(0)
    assert not pending.done()
    release.set_result("settled")
    result = await pending
    assert result["error"]["info"] == ABORTED
    assert kinds(result) == ["nested"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}turns a post-execute block into a valueless failure")
async def test_post_execute_block_is_a_valueless_failure() -> None:
    ctx, tools = await tool_runtime()
    tools.register(string_tool("echo", returning("secret")))
    ctx.on(
        "tools/post-execute",
        lambda execution, result, next_: {
            "kind": "block",
            "feedback": [{"type": "text", "text": "blocked by policy"}],
        },
    )
    result = await execute_one(tools, "echo")
    assert result == {
        "isError": True,
        "error": {"message": "blocked by policy"},
        "content": [{"type": "text", "text": "blocked by policy"}],
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}derives a stable failure message from non-text or empty block feedback"
)
@pytest.mark.parametrize(
    ("feedback", "message"),
    [
        ([], "tool result blocked by post-execute policy"),
        ([{"type": "reasoning", "text": "private rationale"}], "[reasoning content]"),
    ],
)
async def test_block_message_from_non_text_or_empty_feedback(
    feedback: list[Mapping[str, Any]], message: str
) -> None:
    ctx, tools = await tool_runtime()
    tools.register(string_tool("echo", returning("")))
    ctx.on(
        "tools/post-execute",
        lambda execution, result, next_: {"kind": "block", "feedback": feedback},
    )
    result = await execute_one(tools, "echo")
    assert result["error"]["message"] == message
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}contains a non-JSON post-execute failure projection as a safe final error"
)
async def test_non_json_block_feedback_is_a_safe_error() -> None:
    ctx, tools = await tool_runtime()
    tools.register(string_tool("echo", returning("")))
    ctx.on(
        "tools/post-execute",
        lambda execution, result, next_: {
            "kind": "block",
            "feedback": [{"type": "text", "text": "blocked", "invalid": lambda: None}],
        },
    )
    result = await execute_one(tools, "echo")
    assert result["isError"] is True
    assert result["error"]["message"] == "tool result must be losslessly JSON-serializable"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}runs the captured content projector before policy and keeps later replacement "
    "authoritative"
)
async def test_captured_content_projector_runs_before_policy() -> None:
    ctx, tools = await tool_runtime()
    definition = define_tool(
        name="projected",
        description="projected result",
        parameters={},
        output={
            "schema": {"type": "string"},
            "render": lambda args, value: [{"type": "text", "text": "fallback"}],
        },
        execute=returning("canonical"),
        projectContent=lambda execution, result: [{"type": "text", "text": "prepared"}],
    )
    tools.register(definition)

    def pre(execution: object, next_: Callable[[], Any]) -> Any:
        definition["projectContent"] = lambda execution, result: [
            {"type": "text", "text": "later definition"}
        ]
        return next_()

    seen: list[object] = []

    def post(execution: object, result: Mapping[str, Any], next_: Callable[[], Any]) -> Any:
        next_()
        seen.append(result["content"])
        return {"kind": "accept", "content": [{"type": "text", "text": "policy result"}]}

    ctx.on("tools/pre-execute", pre)
    ctx.on("tools/post-execute", post)
    assert "projectContent" not in tools.schemas()[0]
    result = await execute_one(tools, "projected")
    assert seen == [[{"type": "text", "text": "prepared"}]]
    assert result["content"] == [{"type": "text", "text": "policy result"}]
    assert result["value"] == "canonical"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}normalizes a throwing content projector without running post-execute"
)
async def test_throwing_content_projector_skips_post_execute() -> None:
    ctx, tools = await tool_runtime()

    def project(execution: object, result: object) -> object:
        raise ValueError("projection failed")

    tools.register(string_tool("echo", returning(""), projectContent=project))
    post_calls = 0

    def post(execution: object, result: object, next_: Callable[[], Any]) -> Any:
        nonlocal post_calls
        post_calls += 1
        return next_()

    ctx.on("tools/post-execute", post)
    result = await execute_one(tools, "echo")
    assert result["isError"] is True
    assert "projection failed" in result["content"][0]["text"]
    assert post_calls == 0
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}runs the snapshotted final content transform after outer pipeline normalization"
)
async def test_final_content_transform_runs_on_pipeline_failures() -> None:
    ctx, tools = await tool_runtime()

    def finalize(execution: Any, result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        assert execution["name"] == "bounded"
        assert result["isError"] is True
        return [{"type": "text", "text": "bounded failure"}]

    dispose = tools.register(string_tool("bounded", returning("body"), finalizeContent=finalize))

    def pre(execution: object, next_: Callable[[], Any]) -> Any:
        dispose()
        raise HarnessError("policy failed", "POLICY_FAILED")

    ctx.on("tools/pre-execute", pre)
    result = await execute_one(tools, "bounded")
    assert result == {
        "content": [{"type": "text", "text": "bounded failure"}],
        "isError": True,
        "error": {
            "message": "policy failed",
            "info": {"name": "HarnessError", "code": "POLICY_FAILED"},
        },
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}keeps the normalized content when the final content transform returns undefined"
)
async def test_final_content_transform_may_keep_the_content() -> None:
    ctx, tools = await tool_runtime()
    finalized = 0

    def finalize(execution: object, result: object) -> None:
        nonlocal finalized
        finalized += 1

    tools.register(string_tool("identity-finalizer", returning(""), finalizeContent=finalize))
    result = await execute_one(tools, "identity-finalizer")
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": ""}]
    assert finalized == 1
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::settles every unowned call when cancellation precedes the batch")
async def test_cancellation_before_the_batch_settles_every_call() -> None:
    ctx, tools = await tool_runtime()
    invoked = False

    async def execute(args: Mapping[str, Any], execution: object) -> str:
        nonlocal invoked
        invoked = True
        return "unexpected"

    tools.register(string_tool("echo", execute))
    caller = AbortController()
    caller.abort()
    batch = await tools.execute(
        {
            "signal": caller.signal,
            "calls": [
                {"id": id_, "name": "echo", "arguments": "{}"} for id_ in ("first", "second")
            ],
        }
    )
    assert invoked is False
    assert [item["result"]["error"]["info"]["code"] for item in batch["results"]] == [
        TOOL_ABORTED_BEFORE_DISPATCH,
        TOOL_ABORTED_BEFORE_DISPATCH,
    ]
    await ctx.fiber.dispose()
