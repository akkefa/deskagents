"""Ported from packages/core/tools/tests/tools.spec.ts: one call through the pipeline stages."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import SystemPrompt, ToolsService, define_tool
from deskagents.core.tools import (
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    NativeToolExecutor,
    ToolExecutorUnavailableError,
    install_native_tool_executor,
)
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


async def execute_one(
    tools: ToolsService,
    name: str,
    signal: object = None,
    *,
    arguments: object = "{}",
    agent: object = None,
) -> Mapping[str, Any]:
    batch = await tools.execute(
        {
            "calls": [{"id": "c1", "name": name, "arguments": arguments}],
            "signal": signal if signal is not None else AbortController().signal,
            **({"agent": agent} if agent is not None else {}),
        }
    )
    result: Mapping[str, Any] = batch["results"][0]["result"]
    return result


def echo(**overrides: Any) -> dict[str, Any]:
    async def execute(args: Mapping[str, Any], execution: object) -> str:
        return str(args.get("text", ""))

    return {
        **define_tool(
            name="echo",
            description="echo arguments back",
            parameters={"text": {"type": "string"}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
        ),
        **overrides,
    }


class Hostile:
    """A member whose type probe raises, like an enumerable JavaScript getter that throws."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":
            raise RuntimeError(object.__getattribute__(self, "message"))
        return object.__getattribute__(self, name)


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
@pytest.mark.upstream(
    f"{RUNTIME}re-fuses the caller signal with an around-dispatch replacement for the body"
)
async def test_the_body_gets_the_caller_signal_fused_with_a_replacement() -> None:
    ctx, tools = await tool_runtime()
    body_signals: list[object] = []

    async def execute(args: Mapping[str, Any], execution: Any) -> str:
        body_signals.append(execution.signal)
        return "ok"

    tools.register(string_tool("signal-probe", execute))
    caller, replacement = AbortController().signal, AbortController().signal
    wrapper_signals: list[object] = []

    async def around(execution: Any, next_: Callable[[], Any]) -> Any:
        wrapper_signals.append(execution.signal)
        execution.signal = replacement
        return await next_()

    ctx.on("tools/execute", around)
    await execute_one(tools, "signal-probe", caller)
    assert wrapper_signals == [caller]
    assert len(body_signals) == 1
    assert body_signals[0] is not caller
    assert body_signals[0] is not replacement
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


OUTPUT_ERROR = {"name": "ToolOutputError", "code": "INVALID_TOOL_OUTPUT"}
UNKNOWN_ECHO = {
    "message": 'unknown tool "echo"',
    "info": {"name": "ToolNotFoundError", "code": "UNKNOWN_TOOL"},
}


def decide(decision: Mapping[str, Any]) -> Callable[..., Any]:
    async def listener(*args: Any) -> Mapping[str, Any]:
        return decision

    return listener


def raising(error: BaseException) -> Callable[..., Any]:
    async def listener(*args: Any) -> Any:
        raise error

    return listener


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}contains a throwing output.%s snapshot as one failed call")
@pytest.mark.parametrize("projector", ["render", "presentationMeta"])
async def test_a_throwing_projection_snapshot_is_one_failed_call(projector: str) -> None:
    ctx, tools = await tool_runtime()
    hostile = {"value": Hostile("snapshot getter exploded")}
    tools.register(
        define_tool(
            name=f"hostile-{projector}",
            description=projector,
            parameters={},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: (
                    hostile if projector == "render" else [{"type": "text", "text": "ok"}]
                ),
                "presentationMeta": lambda args, value: (
                    hostile if projector == "presentationMeta" else None
                ),
            },
            execute=returning("ok"),
        )
    )
    result = await execute_one(tools, f"hostile-{projector}")
    assert "snapshot getter exploded" in result["error"]["message"]
    assert result["error"]["info"] == OUTPUT_ERROR
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}classifies a throwing body snapshot as invalid tool output")
async def test_a_throwing_body_snapshot_is_invalid_output() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: object, execution: object) -> object:
        return {"value": Hostile("body snapshot getter exploded")}

    tools.register(
        define_tool(
            name="hostile-body",
            description="hostile body",
            parameters={},
            output={"schema": {"type": "json"}, "render": lambda args, value: []},
            execute=execute,
        )
    )
    result = await execute_one(tools, "hostile-body")
    assert "value snapshot failed: body snapshot getter exploded" in result["error"]["message"]
    assert result["error"]["info"] == OUTPUT_ERROR
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}finalizes errors discovered while snapshotting non-content result fields"
)
async def test_snapshot_errors_still_pass_through_the_finalizer() -> None:
    ctx, tools = await tool_runtime()
    calls = 0
    base = echo()

    def finalize(execution: object, result: Mapping[str, Any]) -> Any:
        nonlocal calls
        calls += 1
        block = result["content"][0]
        return None if block["type"] != "text" else [{"type": "text", "text": block["text"][:32]}]

    tools.register(
        {
            **base,
            "name": "throwing-meta",
            "output": {
                **base["output"],
                "presentationMeta": lambda args, value: {
                    "value": Hostile("snapshot failed: " * 100)
                },
            },
            "execute": returning("body"),
            "finalizeContent": finalize,
        }
    )
    result = await execute_one(tools, "throwing-meta")
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert text.startswith('Error: tool "throwing-meta"')
    assert len(text) == 32
    assert calls == 1
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}fails a post-execute decision that replaces both projections or supplies an "
    "invalid value"
)
async def test_invalid_post_execute_replacements_fail_the_call() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    remove = ctx.on(
        "tools/post-execute",
        decide(
            {
                "kind": "accept",
                "value": "replacement",
                "content": [{"type": "text", "text": "also replacement"}],
            }
        ),
    )
    both = await execute_one(tools, "echo")
    assert both["isError"] is True
    assert both["error"]["message"] == (
        "tools/post-execute accept decision cannot replace both value and content"
    )
    remove()
    ctx.on("tools/post-execute", decide({"kind": "accept", "value": 1}))
    invalid = await execute_one(tools, "echo")
    assert invalid["error"]["info"]["code"] == "INVALID_TOOL_OUTPUT"
    assert "value" not in invalid
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}replaces a canonical value without manufacturing additional context"
)
async def test_value_replacement_adds_no_context() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    ctx.on("tools/post-execute", decide({"kind": "accept", "value": "replacement"}))
    assert await execute_one(tools, "echo") == {
        "isError": False,
        "value": "replacement",
        "content": [{"type": "text", "text": "replacement"}],
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}rejects value replacement on a failed dispatch")
async def test_value_replacement_of_a_failed_call_fails() -> None:
    ctx, tools = await tool_runtime()

    async def fail(args: object, execution: object) -> str:
        raise RuntimeError("body failed")

    tools.register(echo(name="throw-before-replace", execute=fail))
    ctx.on("tools/post-execute", decide({"kind": "accept", "value": "replacement"}))
    result = await execute_one(tools, "throw-before-replace")
    assert result["error"]["message"] == (
        "tools/post-execute cannot replace the value of a failed result"
    )
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}fails value replacement when the owning tool disappears before post-policy resolves"
)
async def test_value_replacement_needs_the_tool_to_remain() -> None:
    ctx, tools = await tool_runtime()
    dispose = tools.register(echo())

    async def post(*args: Any) -> Mapping[str, Any]:
        dispose()
        return {"kind": "accept", "value": "replacement"}

    ctx.on("tools/post-execute", post)
    assert (await execute_one(tools, "echo"))["error"] == UNKNOWN_ECHO
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}normalizes wrapper-authored failure metadata and contexts")
async def test_wrapper_failures_keep_metadata_and_contexts() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    wrapper_context = context("test", "wrapper context")
    ctx.on(
        "tools/execute",
        decide(
            {
                "isError": True,
                "error": {"message": "wrapped failure"},
                "content": [{"type": "text", "text": "wrapper content"}],
                "meta": {"wrapped": True},
                "additionalContexts": [wrapper_context],
            }
        ),
    )
    assert await execute_one(tools, "echo") == {
        "isError": True,
        "error": {"message": "wrapped failure"},
        "content": [{"type": "text", "text": "wrapper content"}],
        "meta": {"wrapped": True},
        "additionalContexts": [wrapper_context],
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}fails wrapper-authored success normalization when the owning tool disappears"
)
async def test_wrapper_success_needs_the_tool_to_remain() -> None:
    ctx, tools = await tool_runtime()
    dispose = tools.register(echo())

    async def around(*args: Any) -> Mapping[str, Any]:
        dispose()
        return {"isError": False, "value": "replacement", "content": []}

    ctx.on("tools/execute", around)
    assert (await execute_one(tools, "echo"))["error"] == UNKNOWN_ECHO
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}normalizes a hostile thrown value whose inspection and coercion both throw"
)
async def test_an_unprintable_exception_is_still_an_error_result() -> None:
    ctx, tools = await tool_runtime()

    class Unprintable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("coercion trap")

    async def execute(args: object, execution: object) -> str:
        raise Unprintable

    tools.register(echo(name="hostile-throw", execute=execute))
    result = await execute_one(tools, "hostile-throw")
    assert result["isError"] is True
    assert result["content"] == [{"type": "text", "text": "Error: <unprintable thrown value>"}]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}lets a tools/pre-execute listener deny a call (permission pattern)"
)
async def test_pre_execute_denial_keeps_its_info_and_freezes_the_result() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    info = {
        "name": "AutoReviewDeniedError",
        "code": "AUTO_REVIEW_DENIED",
        "reason": " raw\nreason ",
    }
    ctx.on(
        "tools/pre-execute",
        decide({"kind": "deny", "reason": "denied by policy", "info": info}),
    )
    post_saw_frozen: list[bool] = []

    def post(execution: object, result: Any, next_: Callable[[], Any]) -> Any:
        try:
            result["content"] = []
        except TypeError:
            post_saw_frozen.append(True)
        return next_()

    ctx.on("tools/post-execute", post)
    result = await execute_one(tools, "echo", arguments='{"text":"hi"}')
    assert result["content"] == [{"type": "text", "text": "Error: denied by policy"}]
    assert result["error"] == {"message": "denied by policy", "info": info}
    assert post_saw_frozen == [True]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}an ask decision degrades to deny when no approval seam is mounted")
async def test_ask_with_a_reason_degrades_to_that_denial() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    ctx.on("tools/pre-execute", decide({"kind": "ask", "reason": "needs approval"}))
    result = await execute_one(tools, "echo")
    assert result["content"] == [{"type": "text", "text": "Error: needs approval"}]
    await ctx.fiber.dispose()


class Approval:
    def __init__(self, outcome: object) -> None:
        self.outcome, self.requests = outcome, 0
        self.asked = asyncio.Event()

    async def request(self, request: Mapping[str, Any]) -> object:
        self.requests += 1
        self.asked.set()
        if isinstance(self.outcome, asyncio.Future):
            return await self.outcome
        return self.outcome


ASK = f"{RUNTIME}ask routing through ctx.approval > "


@pytest.mark.asyncio
@pytest.mark.upstream(f"{ASK}denies with the user-rejection reason on rejected")
@pytest.mark.upstream(f"{ASK}denies with the cancellation reason on cancelled")
@pytest.mark.upstream(
    f"{ASK}denies with the no-channel reason when the seam is mounted but nobody answers"
)
@pytest.mark.parametrize(
    ("outcome", "text"),
    [
        ("rejected", 'Error: the user rejected tool "echo"'),
        ("cancelled", 'Error: approval for tool "echo" was cancelled'),
        (
            "unavailable",
            'Error: tool "echo" requires approval, but no approval channel is available',
        ),
    ],
)
async def test_approval_outcomes_deny_with_distinct_reasons(outcome: str, text: str) -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    ctx.provide("approval", Approval(outcome))
    ctx.on("tools/pre-execute", decide({"kind": "ask"}))
    result = await execute_one(tools, "echo", agent=object())
    assert result["isError"] is True
    assert result["content"] == [{"type": "text", "text": text}]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ASK}returns ABORTED_BEFORE_DISPATCH when caller cancellation overtakes approval"
)
async def test_cancellation_overtaking_approval_skips_dispatch() -> None:
    ctx, tools = await tool_runtime()
    dispatched = 0

    async def execute(args: object, execution: object) -> str:
        nonlocal dispatched
        dispatched += 1
        return ""

    tools.register(echo(name="approval-probe", execute=execute))
    answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    approval = Approval(answer)
    ctx.provide("approval", approval)
    ctx.on("tools/pre-execute", decide({"kind": "ask"}))
    caller = AbortController()
    pending = asyncio.ensure_future(
        execute_one(tools, "approval-probe", caller.signal, agent=object())
    )
    await approval.asked.wait()
    caller.abort("caller cancelled approval")
    answer.set_result("allowed-once")
    result = await pending
    assert result["error"]["info"] == ABORTED_BEFORE_DISPATCH
    assert dispatched == 0
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ASK}denies an agent-less execution without asking — nothing to route or audit through"
)
async def test_an_agent_less_ask_is_denied_without_asking() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    approval = Approval("allowed-once")
    ctx.provide("approval", approval)
    ctx.on("tools/pre-execute", decide({"kind": "ask"}))
    result = await execute_one(tools, "echo")
    assert approval.requests == 0
    assert result["content"] == [
        {
            "type": "text",
            "text": 'Error: tool "echo" requires approval, but the call has no agent to route '
            "it through",
        }
    ]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ASK}turns a rogue outcome from a NON-conforming approval stand-in into an isError result"
)
async def test_a_rogue_approval_outcome_is_an_error_result() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    ctx.provide("approval", Approval("yolo"))
    ctx.on("tools/pre-execute", decide({"kind": "ask"}))
    result = await execute_one(tools, "echo", agent=object())
    assert result["isError"] is True
    assert "unreachable" in result["content"][0]["text"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}a tools/post-execute listener can replace the result content (accept) "
)
@pytest.mark.upstream(
    f"{RUNTIME}a tools/post-execute block turns the call into an isError with corrective feedback"
)
async def test_post_execute_accept_and_block() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    remove = ctx.on(
        "tools/post-execute",
        decide({"kind": "accept", "content": [{"type": "text", "text": "rewritten"}]}),
    )
    accepted = await execute_one(tools, "echo", arguments='{"text":"hi"}')
    assert accepted["isError"] is False
    assert accepted["content"] == [{"type": "text", "text": "rewritten"}]
    remove()
    feedback = [{"type": "text", "text": "output rejected: try again"}]
    ctx.on("tools/post-execute", decide({"kind": "block", "feedback": feedback}))
    blocked = await execute_one(tools, "echo", arguments='{"text":"hi"}')
    assert blocked["isError"] is True
    assert blocked["content"] == feedback
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}a block decision can ALSO attach additionalContexts")
@pytest.mark.upstream(
    f"{RUNTIME}post-execute additionalContexts ride on the result for the loop to buffer"
)
async def test_post_execute_decisions_attach_contexts() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    why = context("test", "why it was rejected")
    remove = ctx.on(
        "tools/post-execute",
        decide(
            {
                "kind": "block",
                "feedback": [{"type": "text", "text": "rejected"}],
                "additionalContexts": [why],
            }
        ),
    )
    blocked = await execute_one(tools, "echo")
    assert blocked["isError"] is True
    assert blocked["additionalContexts"] == [why]
    remove()
    fyi = context("test", "fyi")
    ctx.on("tools/post-execute", decide({"kind": "accept", "additionalContexts": [fyi]}))
    assert (await execute_one(tools, "echo"))["additionalContexts"] == [fyi]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}keeps deferred contexts when a composite tool throws, but drops them when the "
    "outer call is blocked"
)
async def test_deferred_contexts_survive_failure_but_not_a_block() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: object, execution: Any) -> str:
        execution.defer_context(context("nested", "nested"))
        raise RuntimeError("outer failure")

    tools.register(echo(name="failing-composite", execute=execute))
    failed = await execute_one(tools, "failing-composite")
    assert failed["isError"] is True
    assert kinds(failed) == ["nested"]
    ctx.on(
        "tools/post-execute",
        decide(
            {
                "kind": "block",
                "feedback": [{"type": "text", "text": "blocked"}],
                "additionalContexts": [context("blocker", "block-only")],
            }
        ),
    )
    blocked = await execute_one(tools, "failing-composite")
    assert kinds(blocked) == ["blocker"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}composes pre + post waterfalls around dispatch (sandbox-wrap pattern)"
)
@pytest.mark.upstream(
    f"{RUNTIME}runs tools/execute after an allowed pre-execute, around dispatch, and before "
    "post-execute"
)
async def test_pre_execute_wrapper_and_post_execute_run_in_order() -> None:
    ctx, tools = await tool_runtime()
    order: list[str] = []

    async def execute(args: Mapping[str, Any], execution: object) -> str:
        order.append("dispatch")
        return str(args.get("text", ""))

    tools.register(echo(execute=execute))

    async def pre(execution: object, next_: Callable[[], Any]) -> Any:
        order.append("pre:before")
        decision = await next_()
        order.append("pre:after")
        return decision

    async def around(execution: object, next_: Callable[[], Any]) -> Any:
        order.append("execute:before")
        result = await next_()
        order.append("execute:after")
        return result

    async def post(execution: object, result: object, next_: Callable[[], Any]) -> Any:
        order.append("post:before")
        decision = await next_()
        order.append("post:after")
        return decision

    ctx.on("tools/pre-execute", pre)
    ctx.on("tools/execute", around)
    ctx.on("tools/post-execute", post)
    result = await execute_one(tools, "echo", arguments='{"text":"hi"}')
    assert result == {"content": [{"type": "text", "text": "hi"}], "isError": False, "value": "hi"}
    assert order == [
        "pre:before",
        "pre:after",
        "execute:before",
        "dispatch",
        "execute:after",
        "post:before",
        "post:after",
    ]
    await ctx.fiber.dispose()


def must_not_run() -> tuple[dict[str, Any], list[int]]:
    dispatched: list[int] = []

    async def execute(args: object, execution: object) -> str:
        dispatched.append(1)
        return ""

    return echo(name="must-not-run", execute=execute), dispatched


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}skips dispatch when caller cancellation arrives while pre-execute awaits"
)
@pytest.mark.upstream(
    f"{RUNTIME}rechecks caller cancellation after an async around-dispatch wrapper delegates"
)
@pytest.mark.parametrize("event", ["tools/pre-execute", "tools/execute"])
async def test_cancellation_while_a_policy_waits_skips_dispatch(event: str) -> None:
    ctx, tools = await tool_runtime()
    tool, dispatched = must_not_run()
    tools.register(tool)
    entered, release = asyncio.Event(), asyncio.Event()

    async def waiting(execution: object, next_: Callable[[], Any]) -> Any:
        entered.set()
        await release.wait()
        return await next_()

    ctx.on(event, waiting)
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "must-not-run", caller.signal))
    await entered.wait()
    caller.abort("cancelled in policy")
    release.set()
    result = await pending
    assert result["content"] == [
        {"type": "text", "text": "Error: tool call aborted before dispatch"}
    ]
    assert result["error"]["info"] == ABORTED_BEFORE_DISPATCH
    assert dispatched == []
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}maps an explicit pre-execute cancellation to the canonical before-dispatch result"
)
async def test_explicit_cancel_decision_is_aborted_before_dispatch() -> None:
    ctx, tools = await tool_runtime()
    tool, dispatched = must_not_run()
    tools.register(tool)
    ctx.on("tools/pre-execute", decide({"kind": "cancel"}))
    assert await execute_one(tools, "must-not-run") == {
        "content": [{"type": "text", "text": "Error: tool call aborted before dispatch"}],
        "isError": True,
        "error": {
            "message": "tool call aborted before dispatch",
            "info": ABORTED_BEFORE_DISPATCH,
        },
    }
    assert dispatched == []
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}preserves a pre-execute denial that settles after cancellation")
@pytest.mark.upstream(
    f"{RUNTIME}preserves an async pre-execute failure that settles after cancellation"
)
@pytest.mark.upstream(
    f"{RUNTIME}preserves an around-dispatch failure that settles after cancellation"
)
@pytest.mark.upstream(f"{RUNTIME}preserves a post-policy failure that settles after cancellation")
@pytest.mark.parametrize(
    ("event", "outcome", "expected"),
    [
        (
            "tools/pre-execute",
            {"kind": "deny", "reason": "policy denied the call"},
            {"message": "policy denied the call"},
        ),
        ("tools/pre-execute", RuntimeError("gate interrupted"), {"message": "gate interrupted"}),
        (
            "tools/execute",
            HarnessError("wrapper failed", "WRAPPER_FAILURE"),
            {
                "message": "wrapper failed",
                "info": {"name": "HarnessError", "code": "WRAPPER_FAILURE"},
            },
        ),
        (
            "tools/post-execute",
            HarnessError("post-policy failed", "POST_FAILURE"),
            {
                "message": "post-policy failed",
                "info": {"name": "HarnessError", "code": "POST_FAILURE"},
            },
        ),
    ],
)
async def test_failures_settling_after_cancellation_are_kept(
    event: str, outcome: object, expected: Mapping[str, Any]
) -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    entered, release = asyncio.Event(), asyncio.Event()

    async def waiting(*args: Any) -> Any:
        entered.set()
        await release.wait()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    ctx.on(event, waiting)
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "echo", caller.signal))
    await entered.wait()
    caller.abort("cancelled while policy decided")
    release.set()
    result = await pending
    assert result["isError"] is True
    assert result["error"] == expected
    assert result["content"] == [{"type": "text", "text": f"Error: {expected['message']}"}]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}preserves a tool-owned failure after the body observes cancellation"
)
async def test_a_tool_failure_after_cancellation_is_kept() -> None:
    ctx, tools = await tool_runtime()
    entered = asyncio.Event()

    async def execute(args: object, execution: Any) -> str:
        entered.set()
        await execution.signal.wait()
        raise HarnessError("tool failed", "TOOL_FAILURE")

    tools.register(echo(name="tool-failure", execute=execute))
    caller = AbortController()
    pending = asyncio.ensure_future(execute_one(tools, "tool-failure", caller.signal))
    await entered.wait()
    caller.abort("cancelled running body")
    result = await pending
    assert result["content"] == [{"type": "text", "text": "Error: tool failed"}]
    assert result["error"]["info"] == {"name": "HarnessError", "code": "TOOL_FAILURE"}
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}a pre-execute deny short-circuits before tools/execute (the seam never runs)"
)
async def test_a_denied_call_never_enters_tools_execute() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    entered: list[bool] = []
    ctx.on("tools/pre-execute", decide({"kind": "deny", "reason": "nope"}))

    def around(execution: object, next_: Callable[[], Any]) -> Any:
        entered.append(True)
        return next_()

    ctx.on("tools/execute", around)
    result = await execute_one(tools, "echo")
    assert result["content"] == [{"type": "text", "text": "Error: nope"}]
    assert entered == []
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}freezes core dispatch outcomes before around and post listeners can observe them"
)
async def test_dispatch_outcomes_are_frozen_for_listeners() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    mutations: list[bool] = []

    def attempt(result: Any, value: str) -> None:
        try:
            result["value"] = value
            mutations.append(True)
        except TypeError:
            mutations.append(False)

    async def around(execution: object, next_: Callable[[], Any]) -> Any:
        result = await next_()
        attempt(result, "around mutation")
        return result

    def post(execution: object, result: Any, next_: Callable[[], Any]) -> Any:
        attempt(result, "post mutation")
        return next_()

    ctx.on("tools/execute", around)
    ctx.on("tools/post-execute", post)
    result = await execute_one(tools, "echo", arguments='{"text":"original"}')
    assert mutations == [False, False]
    assert result["value"] == "original"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}a tools/execute listener can short-circuit dispatch by returning a result without "
    "next()"
)
async def test_a_wrapper_can_short_circuit_dispatch() -> None:
    ctx, tools = await tool_runtime()
    tool, dispatched = must_not_run()
    tools.register({**tool, "name": "never-runs"})
    ctx.on(
        "tools/execute",
        decide(
            {
                "content": [{"type": "text", "text": "ignored authored content"}],
                "isError": False,
                "value": "short-circuited",
            }
        ),
    )
    result = await execute_one(tools, "never-runs")
    assert dispatched == []
    assert result["content"] == [{"type": "text", "text": "short-circuited"}]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}revalidates a cached canonical result returned from a different dispatch"
)
async def test_a_borrowed_canonical_result_is_revalidated() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo(name="string-output", execute=returning("cached")))
    object_ran: list[bool] = []

    async def object_body(args: object, execution: object) -> Mapping[str, Any]:
        object_ran.append(True)
        return {"ok": True}

    tools.register(
        define_tool(
            name="object-output",
            description="Return one closed object.",
            parameters={},
            output={
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean", "required": True}},
                    "additionalProperties": False,
                },
                "render": lambda args, value: [{"type": "text", "text": str(value["ok"])}],
            },
            execute=object_body,
        )
    )
    cached: list[Any] = []

    async def around(execution: Any, next_: Callable[[], Any]) -> Any:
        if execution["name"] == "string-output":
            cached.append(await next_())
            return cached[0]
        return cached[0]

    ctx.on("tools/execute", around)
    first = await execute_one(tools, "string-output")
    second = await execute_one(tools, "object-output")
    assert first["value"] == "cached"
    assert object_ran == []
    assert second["isError"] is True
    assert second["error"]["info"] == OUTPUT_ERROR
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}preserves additionalContexts supplied by an around-dispatch result"
)
async def test_wrapper_supplied_contexts_are_kept() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    supplied = context("test", "from around dispatch")
    ctx.on(
        "tools/execute",
        decide(
            {
                "content": [{"type": "text", "text": "short-circuited with context"}],
                "isError": False,
                "value": "short-circuited with context",
                "additionalContexts": [supplied],
            }
        ),
    )
    assert (await execute_one(tools, "echo"))["additionalContexts"] == [supplied]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}returns an isError result when a tools/execute listener throws")
@pytest.mark.upstream(
    f"{RUNTIME}returns an isError result when a tools/pre-execute listener throws"
)
@pytest.mark.upstream(
    f"{RUNTIME}returns an isError result when a tools/post-execute listener throws"
)
@pytest.mark.parametrize(
    ("event", "message"),
    [
        ("tools/execute", "wrapper broke"),
        ("tools/pre-execute", "permission hook broke"),
        ("tools/post-execute", "post hook broke"),
    ],
)
async def test_throwing_listeners_become_error_results(event: str, message: str) -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    ctx.on(event, raising(RuntimeError(message)))
    assert await execute_one(tools, "echo", arguments='{"text":"hi"}') == {
        "content": [{"type": "text", "text": f"Error: {message}"}],
        "error": {"message": message},
        "isError": True,
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}preserves structured error info when a tools/pre-execute listener throws "
    "HarnessError"
)
async def test_a_throwing_policy_keeps_its_harness_error_info() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    ctx.on("tools/pre-execute", raising(HarnessError("denied", "DENIED")))
    result = await execute_one(tools, "echo")
    assert result["error"] == {
        "message": "denied",
        "info": {"name": "HarnessError", "code": "DENIED"},
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}settles accepted calls after the actual native executor fiber unloads"
)
async def test_accepted_calls_settle_after_the_executor_unloads() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)

    def executor_plugin(plugin_ctx: Context, config: object = None) -> None:
        install_native_tool_executor(plugin_ctx)

    provider = await ctx.plugin(executor_plugin)
    tools.register(echo())
    entered, release = asyncio.Event(), asyncio.Event()

    async def pre(execution: Any, next_: Callable[[], Any]) -> Any:
        if execution["callId"] == "first":
            entered.set()
            await release.wait()
        return await next_()

    ctx.on("tools/pre-execute", pre)
    calls = [
        {"id": id_, "name": "echo", "arguments": f'{{"text":"{id_}"}}'}
        for id_ in ("first", "second")
    ]
    batch = {"calls": calls, "signal": AbortController().signal}
    pending = asyncio.ensure_future(tools.execute(batch))
    await entered.wait()
    await provider.dispose()
    with pytest.raises(ToolExecutorUnavailableError, match="selected tool executor is unavailable"):
        await tools.execute(batch)
    release.set()
    result = await pending
    assert [(item["result"]["isError"], item["result"]["value"]) for item in result["results"]] == [
        (False, "first"),
        (False, "second"),
    ]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}preserves deferred loading through registry and prompt schema projection"
)
async def test_deferred_loading_reaches_schemas_and_the_prompt() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo(name="deferred", deferLoading=True))
    assert tools.schemas()[0]["deferLoading"] is True
    prompt = ctx.get("systemPrompt")
    assert isinstance(prompt, SystemPrompt)
    assert (await prompt.assemble({}))["tools"][0]["deferLoading"] is True
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}schemas() snapshots deeply nested parameters without using structured-clone "
    "recursion"
)
async def test_deep_parameter_schemas_are_projected_iteratively() -> None:
    ctx, tools = await tool_runtime()
    nested: dict[str, Any] = {"type": "string"}
    for _ in range(5_000):
        nested = {"oneOf": [nested, {"type": "null"}]}
    tools.register(
        echo(name="deep-schema", parameters={"type": "object", "properties": {"nested": nested}})
    )
    cursor = tools.schemas()[0]["parameters"]["properties"]["nested"]
    layers = 0
    while "oneOf" in cursor:
        cursor = cursor["oneOf"][0]
        layers += 1
    assert layers == 5_000
    assert cursor == {"type": "string"}
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::executes raw native calls without an Agent or Session and keeps input order"
)
async def test_raw_calls_run_without_an_agent_and_keep_order() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo())
    batch = await tools.execute(
        {
            "signal": AbortController().signal,
            "calls": [
                {"id": "raw", "name": "echo", "arguments": '{ "text": "raw value" }'},
                {"id": "malformed", "name": "echo", "arguments": "{invalid"},
                {"id": "empty", "name": "echo", "arguments": ""},
            ],
        }
    )
    results = batch["results"]
    assert [item["callId"] for item in results] == ["raw", "malformed", "empty"]
    assert (results[0]["result"]["isError"], results[0]["result"]["value"]) == (False, "raw value")
    assert results[1]["result"]["isError"] is True
    assert (results[2]["result"]["isError"], results[2]["result"]["value"]) == (False, "")
    assert batch["additionalContexts"] == []
    assert batch["concluded"] is False
    assert ctx.get("sessions") is None
    await ctx.fiber.dispose()
