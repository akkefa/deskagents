"""Ported from packages/core/tools/tests/activation.spec.ts and execution-mode.spec.ts.

Conditional registrations share step admission with every call; execution mode
classification fails closed to exclusive.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import SystemPrompt, ToolsService, define_tool
from deskagents.core.scope import Scope, create_scope
from deskagents.core.tools import (
    TOOL_ABORTED_BEFORE_DISPATCH,
    NativeToolExecutor,
    ResolvedToolSet,
    ToolActivationError,
)
from deskagents.util.abort import AbortController, AbortError

ACTIVATION = "packages/core/tools/tests/activation.spec.ts::"
MODE = "packages/core/tools/tests/execution-mode.spec.ts::ToolExecutionPipeline.executionMode > "
MODEL_SELECTION = {"provider": "test", "model": "test"}


class FakeAgent:
    """An identity scope key standing in for an Agent."""

    def __init__(self, id_: str) -> None:
        self.id = id_


async def tool_runtime() -> tuple[Context, ToolsService]:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    return ctx, tools


@dataclass
class Harness:
    ctx: Context
    tools: ToolsService
    agent: FakeAgent
    scope: Scope
    tool: dict[str, Any]
    controller: AbortController = field(default_factory=AbortController)
    calls: list[str] = field(default_factory=list)

    async def execute(
        self, *, agent: FakeAgent | None = None, tool_set: ResolvedToolSet | None = None
    ) -> Mapping[str, Any]:
        batch = await self.tools.execute(
            {
                "calls": [{"id": "call", "name": "conditional", "arguments": "{}"}],
                "signal": self.controller.signal,
                "agent": agent or self.agent,
                **({"toolSet": tool_set} if tool_set is not None else {}),
            }
        )
        result: Mapping[str, Any] = batch["results"][0]["result"]
        return result

    async def resolve(self) -> ResolvedToolSet:
        return await self.tools.resolve(
            {
                "agent": self.agent,
                "modelSelection": MODEL_SELECTION,
                "signal": self.controller.signal,
            }
        )


async def setup() -> Harness:
    ctx, tools = await tool_runtime()
    agent = FakeAgent("activation")
    scopes: list[Scope] = []

    def host(inner: Context, config: object = None) -> None:
        scopes.append(create_scope(inner, agent))

    await ctx.plugin(host)
    calls: list[str] = []

    async def execute(args: object, execution: object) -> str:
        calls.append("conditional")
        return "done"

    tool = define_tool(
        name="conditional",
        description="A conditional operation.",
        parameters={},
        output={
            "schema": {"type": "string"},
            "render": lambda args, value: [{"type": "text", "text": value}],
        },
        execute=execute,
    )
    return Harness(ctx, tools, agent, scopes[0], tool, calls=calls)


def code(result: Mapping[str, Any]) -> object:
    return result.get("error", {}).get("info", {}).get("code")


def always(context: Mapping[str, Any]) -> bool:
    return True


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ACTIVATION}activates and deactivates a registered tool without allowing historical calls"
)
async def test_activation_follows_each_step_without_reviving_old_steps() -> None:
    h = await setup()
    enabled = False
    h.tools.register(h.tool, when=lambda context: enabled)
    unavailable = await h.resolve()
    assert h.tools.schemas(h.agent, unavailable) == []
    enabled = True
    assert code(await h.execute(tool_set=unavailable)) == "UNKNOWN_TOOL"
    assert code(await h.execute()) == "UNKNOWN_TOOL"
    available = await h.resolve()
    assert [schema["name"] for schema in h.tools.schemas(h.agent, available)] == ["conditional"]
    assert (await h.execute(tool_set=available))["isError"] is False
    enabled = False
    disabled = await h.resolve()
    assert code(await h.execute(tool_set=disabled)) == "UNKNOWN_TOOL"
    assert h.calls == ["conditional"]
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{ACTIVATION}applies a batch admission to the calls its executor runs")
async def test_a_batch_runs_only_what_its_step_admitted() -> None:
    h = await setup()
    h.tools.register(h.tool, when=always)
    tool_set = await h.resolve()
    assert (await h.execute(tool_set=tool_set))["isError"] is False
    assert code(await h.execute()) == "UNKNOWN_TOOL"
    assert h.calls == ["conditional"]
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{ACTIVATION}keeps admissions specific to the requesting agent")
async def test_admission_belongs_to_the_requesting_agent() -> None:
    h = await setup()
    h.tools.register(h.tool, when=always)
    tool_set = await h.resolve()
    assert code(await h.execute(agent=FakeAgent("other"), tool_set=tool_set)) == "UNKNOWN_TOOL"
    assert h.calls == []
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ACTIVATION}applies live restrictions and closes step admission permanently"
)
async def test_restrictions_apply_live_and_disposal_is_final() -> None:
    h = await setup()
    h.tools.register(h.tool, when=always)
    tool_set = await h.resolve()
    lift = h.tools.restrict(deny=["conditional"], ctx=h.scope.ctx)
    assert code(await h.execute(tool_set=tool_set)) == "UNKNOWN_TOOL"
    lift()
    assert (await h.execute(tool_set=tool_set))["isError"] is False
    tool_set.dispose()
    assert h.tools.schemas(h.agent, tool_set) == []
    assert code(await h.execute(tool_set=tool_set)) == "UNKNOWN_TOOL"
    assert h.calls == ["conditional"]
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ACTIVATION}reports a wrapper result completed at cancellation as cancelled rather than "
    "unknown"
)
async def test_a_wrapper_result_at_cancellation_is_aborted_not_unknown() -> None:
    h = await setup()
    h.tools.register(h.tool, when=always)
    tool_set = await h.resolve()

    async def around(execution: object, next_: Callable[[], Any]) -> Mapping[str, Any]:
        h.controller.abort()
        return {"isError": False, "value": "wrapped", "content": [{"type": "text", "text": "x"}]}

    h.ctx.on("tools/execute", around)
    assert code(await h.execute(tool_set=tool_set)) == TOOL_ABORTED_BEFORE_DISPATCH
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{ACTIVATION}does not substitute a same-name registration after admission")
async def test_a_re_registration_is_not_admitted_by_an_older_step() -> None:
    h = await setup()
    unregister = h.tools.register(h.tool, when=always)
    old = await h.resolve()
    unregister()
    h.tools.register(h.tool, when=always)
    assert code(await h.execute(tool_set=old)) == "UNKNOWN_TOOL"
    assert (await h.execute(tool_set=await h.resolve()))["isError"] is False
    assert h.calls == ["conditional"]
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ACTIVATION}does not restore a registration removed during its asynchronous condition"
)
async def test_a_late_condition_cannot_restore_a_removed_registration() -> None:
    h = await setup()
    condition: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    entered = asyncio.Event()

    def when(context: Mapping[str, Any]) -> asyncio.Future[bool]:
        entered.set()
        return condition

    unregister = h.tools.register(h.tool, when=when)
    pending = asyncio.ensure_future(h.resolve())
    await entered.wait()
    unregister()
    condition.set_result(True)
    tool_set = await pending
    assert h.tools.schemas(h.agent, tool_set) == []
    assert code(await h.execute(tool_set=tool_set)) == "UNKNOWN_TOOL"
    assert h.calls == []
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ACTIVATION}propagates cancellation after an asynchronous condition settles"
)
async def test_cancellation_wins_after_a_condition_settles() -> None:
    h = await setup()
    condition: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    seen: list[Mapping[str, Any]] = []
    entered = asyncio.Event()

    def when(context: Mapping[str, Any]) -> asyncio.Future[bool]:
        seen.append(context)
        entered.set()
        return condition

    h.tools.register(h.tool, when=when)
    pending = asyncio.ensure_future(h.resolve())
    await entered.wait()
    h.controller.abort()
    condition.set_result(True)
    with pytest.raises(AbortError):
        await pending
    assert seen[0]["agent"] is h.agent
    assert seen[0]["modelSelection"] is MODEL_SELECTION
    assert seen[0]["signal"] is h.controller.signal
    assert h.calls == []
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ACTIVATION}reports condition failures without treating them as disabled tools"
)
async def test_a_failing_condition_is_an_activation_error() -> None:
    h = await setup()
    cause = RuntimeError("credential storage unavailable")

    def when(context: Mapping[str, Any]) -> bool:
        raise cause

    h.tools.register(h.tool, when=when)
    with pytest.raises(ToolActivationError) as caught:
        await h.resolve()
    assert caught.value.code == "TOOL_ACTIVATION_FAILED"
    assert caught.value.cause is cause
    assert h.calls == []
    await h.ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_a_condition_failing_after_cancellation_reports_the_cancellation() -> None:
    h = await setup()
    entered, release = asyncio.Event(), asyncio.Event()

    async def when(context: Mapping[str, Any]) -> bool:
        entered.set()
        await release.wait()
        raise RuntimeError("condition failed late")

    h.tools.register(h.tool, when=when)
    pending = asyncio.ensure_future(h.resolve())
    await entered.wait()
    h.controller.abort()
    release.set()
    with pytest.raises(AbortError):
        await pending
    await h.ctx.fiber.dispose()


# --- execution mode ------------------------------------------------------------------------


def mode(tools: ToolsService, name: str, args: object) -> str:
    """The native executor's per-call classifier (upstream ``pipeline.executionMode``)."""
    executor = tools._select_executor(None)
    assert isinstance(executor, NativeToolExecutor)
    block = {"id": "c1", "name": name, "arguments": json.dumps(args)}
    return executor._execution_mode(block, {"signal": AbortController().signal})


def content_tool(name: str, parameters: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """Upstream's ``defineContentToolFixture``: the canonical value is the content."""

    async def execute(args: object, execution: object) -> list[Any]:
        return []

    return define_tool(
        name=name,
        description=name,
        parameters=parameters,
        output={"schema": {"type": "json"}, "render": lambda args, value: value},
        execute=execute,
        **extra,
    )


def raw_tool(name: str, classifier: Callable[[Any], object]) -> dict[str, Any]:
    async def execute(args: object, execution: object) -> None:
        return None

    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
        "output": {"schema": {"type": "null"}, "render": lambda args, value: []},
        "isConcurrencySafe": classifier,
        "execute": execute,
    }


@pytest.mark.asyncio
@pytest.mark.upstream(f"{MODE}returns parallel only for an explicit true classifier")
@pytest.mark.upstream(
    f"{MODE}defaults to exclusive for a tool with no isConcurrencySafe declaration"
)
@pytest.mark.upstream(f"{MODE}returns exclusive for an unknown tool")
async def test_only_an_explicit_true_runs_in_parallel() -> None:
    ctx, tools = await tool_runtime()
    tools.register(content_tool("safe", {}, isConcurrencySafe=lambda args: True))
    tools.register(content_tool("plain", {}))
    assert mode(tools, "safe", {}) == "parallel"
    assert mode(tools, "plain", {}) == "exclusive"
    assert mode(tools, "nonexistent", {}) == "exclusive"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{MODE}returns exclusive when the classifier returns false for these args")
async def test_the_classifier_decides_per_call() -> None:
    ctx, tools = await tool_runtime()
    tools.register(
        content_tool(
            "rw",
            {"mode": {"type": "string", "required": True}},
            isConcurrencySafe=lambda args: args["mode"] == "read",
        )
    )
    assert mode(tools, "rw", {"mode": "read"}) == "parallel"
    assert mode(tools, "rw", {"mode": "write"}) == "exclusive"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{MODE}classifies invalid defineContentToolFixture arguments as exclusive without throwing"
)
async def test_invalid_arguments_classify_as_exclusive() -> None:
    ctx, tools = await tool_runtime()
    tools.register(
        content_tool(
            "needs-mode",
            {"mode": {"type": "string", "required": True}},
            isConcurrencySafe=lambda args: True,
        )
    )
    assert mode(tools, "needs-mode", {}) == "exclusive"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{MODE}treats a throwing raw classifier as exclusive")
@pytest.mark.upstream(f"{MODE}treats a truthy non-boolean raw result as exclusive")
async def test_raw_classifiers_fail_closed() -> None:
    ctx, tools = await tool_runtime()

    def throwing(args: object) -> bool:
        raise RuntimeError("boom")

    tools.register(raw_tool("thrower", throwing))
    tools.register(raw_tool("truthy", lambda args: "yes"))
    assert mode(tools, "thrower", {}) == "exclusive"
    assert mode(tools, "truthy", {}) == "exclusive"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{MODE}passes parsed arguments directly to a raw definition")
async def test_raw_classifiers_receive_the_parsed_arguments() -> None:
    ctx, tools = await tool_runtime()
    seen: list[object] = []

    def classifier(args: object) -> bool:
        seen.append(args)
        return True

    tools.register(raw_tool("raw-safe", classifier))
    assert mode(tools, "raw-safe", {"anything": 1}) == "parallel"
    assert seen == [{"anything": 1}]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{MODE}isConcurrencySafe never reaches the model-facing schemas() projection"
)
async def test_the_classifier_never_reaches_the_model() -> None:
    ctx, tools = await tool_runtime()
    tools.register(
        content_tool(
            "safe", {"x": {"type": "string", "required": True}}, isConcurrencySafe=lambda a: True
        )
    )
    assert sorted(tools.schemas()[0]) == ["description", "name", "parameters"]
    await ctx.fiber.dispose()
