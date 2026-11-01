"""Ported from packages/core/agent-loop/tests/tool-calls.spec.ts: the tool-call scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import AgentLoop, AgentRegistry, SystemPrompt, ToolsService, define_tool
from deskagents.core.session import SessionStore
from deskagents.core.tools import TOOL_ABORTED_BEFORE_DISPATCH, NativeToolExecutor
from deskagents.llm import LlmRuntime, create_user_message
from deskagents.testing import ScriptedAdapter

SPEC = "packages/core/agent-loop/tests/tool-calls.spec.ts"
ABORT = f"{SPEC}::tool-call scheduler: abort handling > "
ABORTED_BEFORE_DISPATCH = {"name": "AbortError", "code": TOOL_ABORTED_BEFORE_DISPATCH}


async def harness(scripts: Sequence[Any]) -> tuple[Context, AgentRegistry, ToolsService]:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    await ctx.plugin(SystemPrompt, {"includeHarnessIdentity": False, "personaPrefix": "P"})
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(AgentLoop)
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    llm.register_adapter(["test"], ScriptedAdapter(scripts))
    agents = ctx.get("agents")
    assert isinstance(agents, AgentRegistry)
    return ctx, agents, tools


async def create(agents: AgentRegistry, max_parallel: int | None = None) -> Any:
    options = {"provider": "test", "model": "m"}
    if max_parallel is not None:
        options["maxParallelToolCalls"] = max_parallel  # type: ignore[assignment]
    return await agents.create({"sessionId": "a1", "agentOptions": options})


def user(text: str) -> Mapping[str, Any]:
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})


def multi_call(calls: Sequence[tuple[str, str, str]]) -> list[Mapping[str, Any]]:
    chunks: list[Mapping[str, Any]] = []
    for index, (call_id, name, arg) in enumerate(calls):
        block = {"type": "tool-call", "id": call_id, "name": name, "arguments": f'{{"id":"{arg}"}}'}
        chunks.append({"type": "block-end", "index": index, "block": block})
    chunks.append({"type": "finish", "reason": {"kind": "stop"}})
    return chunks


def answer(text: str) -> list[Mapping[str, Any]]:
    return [
        {"type": "text-delta", "index": 0, "text": text},
        {"type": "finish", "reason": {"kind": "stop"}},
    ]


class Gated:
    """A tool whose calls block until the test releases them by id."""

    def __init__(self, name: str, *, parallel: bool) -> None:
        self.started: list[str] = []
        self._gates: dict[str, asyncio.Future[None]] = {}

        async def execute(args: Mapping[str, Any], execution: object) -> str:
            del execution
            self.started.append(args["id"])
            gate = asyncio.get_running_loop().create_future()
            self._gates[args["id"]] = gate
            await gate
            return f"done-{args['id']}"

        self.tool = define_tool(
            name=name,
            description=f"gated {name}",
            parameters={"id": {"type": "string", "required": True}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
            **({"isConcurrencySafe": lambda args: True} if parallel else {}),
        )

    def release(self, id_: str) -> None:
        gate = self._gates.pop(id_, None)
        if gate is not None and not gate.done():
            gate.set_result(None)


async def until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


def of_type(agent: Any, kind: str) -> list[Mapping[str, Any]]:
    return [event for event in agent.session.snapshot_events() if event["type"] == kind]


def result_summary(agent: Any) -> list[tuple[str, bool, object]]:
    return [
        (
            event["data"]["message"]["source"]["callId"],
            event["data"]["message"]["isError"],
            event["data"].get("error"),
        )
        for event in of_type(agent, "tool/result")
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ABORT}starts no calls when the signal is already aborted before a parallel group"
)
async def test_abort_before_a_parallel_group_starts_no_calls() -> None:
    ctx, agents, tools = await harness(
        [multi_call([("c1", "p", "1"), ("c2", "p", "2")]), answer("never")]
    )
    gated = Gated("p", parallel=True)
    tools.register(gated.tool)
    handle = await create(agents)
    agent = handle.agent

    def cancel_on_message(session: Any, event: Mapping[str, Any]) -> None:
        if session is agent.session and event["type"] == "assistant/message":
            agent.cancel({"kind": "user"})

    ctx.on("session/event", cancel_on_message)
    agent.followup(user("go"))
    await agent.when_idle()
    assert gated.started == []
    assert [event["data"]["callId"] for event in of_type(agent, "tool/call")] == ["c1", "c2"]
    assert result_summary(agent) == [
        ("c1", True, ABORTED_BEFORE_DISPATCH),
        ("c2", True, ABORTED_BEFORE_DISPATCH),
    ]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ABORT}skips dispatch and stops starting siblings when abort fires during ordered pre-execute"
)
async def test_abort_during_pre_execute_skips_dispatch_and_siblings() -> None:
    ctx, agents, tools = await harness(
        [multi_call([("c1", "p", "1"), ("c2", "p", "2")]), answer("never")]
    )
    gated = Gated("p", parallel=True)
    tools.register(gated.tool)
    handle = await create(agents)
    agent = handle.agent

    def pre_execute(execution: Mapping[str, Any], next_: Callable[[], object]) -> object:
        if execution["callId"] == "c1":
            agent.cancel({"kind": "user"})
        return next_()

    ctx.on("tools/pre-execute", pre_execute)
    agent.followup(user("go"))
    await agent.when_idle()
    assert gated.started == []
    assert [event["data"]["callId"] for event in of_type(agent, "tool/call")] == ["c1", "c2"]
    assert result_summary(agent) == [
        ("c1", True, ABORTED_BEFORE_DISPATCH),
        ("c2", True, ABORTED_BEFORE_DISPATCH),
    ]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{ABORT}stops replenishing after abort, commits started results, and parks accepted "
    "additional contexts"
)
async def test_abort_stops_the_pool_and_parks_committed_contexts() -> None:
    ctx, agents, tools = await harness(
        [
            multi_call([(f"c{n}", "p", str(n)) for n in (1, 2, 3, 4)]),
            answer("after wake"),
        ]
    )
    gated = Gated("p", parallel=True)
    tools.register(gated.tool)

    def post_execute(
        execution: Mapping[str, Any], result: Mapping[str, Any], next_: Callable[[], Any]
    ) -> Mapping[str, Any]:
        context = create_user_message(
            content=[{"type": "text", "text": f"ctx-{execution['callId']}"}],
            source={"kind": "p"},
        )
        return {**next_(), "additionalContexts": [context]}

    ctx.on("tools/post-execute", post_execute)
    handle = await create(agents, max_parallel=2)
    agent = handle.agent
    agent.followup(user("go"))
    await until(lambda: len(gated.started) == 2)
    agent.cancel({"kind": "user"})
    gated.release("1")
    gated.release("2")
    await agent.when_idle()

    assert gated.started == ["1", "2"]
    call_ids = ["c1", "c2", "c3", "c4"]
    assert [event["data"]["callId"] for event in of_type(agent, "tool/call")] == call_ids
    assert [summary[0] for summary in result_summary(agent)] == call_ids
    assert result_summary(agent)[2:] == [
        ("c3", True, ABORTED_BEFORE_DISPATCH),
        ("c4", True, ABORTED_BEFORE_DISPATCH),
    ]
    assert [message["content"][0]["text"] for message in agent.inbox.next_step] == [
        "ctx-c1",
        "ctx-c2",
    ]

    agent.followup(user("wake"))
    await agent.when_idle()
    delivered = [
        event["data"]["content"][0]["text"]
        for event in of_type(agent, "user/message")
        if event["data"]["source"]["kind"] == "p"
    ]
    assert delivered == ["ctx-c1", "ctx-c2"]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{ABORT}does not run an exclusive barrier after a parallel group aborts")
async def test_abort_skips_the_exclusive_barrier_after_a_parallel_group() -> None:
    ctx, agents, tools = await harness(
        [multi_call([("c1", "p", "1"), ("c2", "p", "2"), ("c3", "x", "3")]), answer("never")]
    )
    gated = Gated("p", parallel=True)
    exclusive: list[str] = []

    async def run_exclusive(args: Mapping[str, Any], execution: object) -> str:
        exclusive.append(args["id"])
        return "x"

    tools.register(gated.tool)
    tools.register(
        define_tool(
            name="x",
            description="exclusive",
            parameters={"id": {"type": "string", "required": True}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=run_exclusive,
        )
    )
    handle = await create(agents, max_parallel=2)
    agent = handle.agent
    agent.followup(user("go"))
    await until(lambda: len(gated.started) == 2)
    agent.cancel({"kind": "user"})
    gated.release("1")
    gated.release("2")
    await agent.when_idle()

    assert exclusive == []
    assert [event["data"]["callId"] for event in of_type(agent, "tool/call")] == [
        "c1",
        "c2",
        "c3",
    ]
    last = of_type(agent, "tool/result")[-1]["data"]
    assert last["message"]["source"]["callId"] == "c3"
    assert last["message"]["content"] == [
        {"type": "text", "text": "Error: tool call aborted before dispatch"}
    ]
    assert last["error"] == ABORTED_BEFORE_DISPATCH
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::tool-call scheduler: model-order results despite out-of-order settlement > "
    "commits tool/result in model order even when a later call settles first"
)
async def test_results_commit_in_model_order() -> None:
    ctx, agents, tools = await harness(
        [multi_call([("c1", "p", "1"), ("c2", "p", "2")]), answer("done")]
    )
    gated = Gated("p", parallel=True)
    tools.register(gated.tool)
    handle = await create(agents)
    agent = handle.agent
    agent.followup(user("go"))
    await until(lambda: len(gated.started) == 2)
    gated.release("2")
    await asyncio.sleep(0)
    assert of_type(agent, "tool/result") == []
    gated.release("1")
    await agent.when_idle()
    assert [summary[0] for summary in result_summary(agent)] == ["c1", "c2"]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::tool-call scheduler: rolling pool honors maxParallelToolCalls > "
    "starts at most the cap, replenishing as calls settle"
)
async def test_rolling_pool_starts_at_most_the_cap() -> None:
    ctx, agents, tools = await harness(
        [multi_call([(f"c{n}", "p", str(n)) for n in (1, 2, 3)]), answer("done")]
    )
    gated = Gated("p", parallel=True)
    tools.register(gated.tool)
    handle = await create(agents, max_parallel=2)
    agent = handle.agent
    agent.followup(user("go"))
    await until(lambda: len(gated.started) == 2)
    await asyncio.sleep(0)
    assert gated.started == ["1", "2"]
    gated.release("1")
    await until(lambda: len(gated.started) == 3)
    gated.release("2")
    gated.release("3")
    await agent.when_idle()
    assert [summary[0] for summary in result_summary(agent)] == ["c1", "c2", "c3"]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::tool-call scheduler: failure quiescence > "
    "pairs outstanding requests after %s fails and preserves committed results"
)
@pytest.mark.parametrize("phase", ["execution-mode", "prepare", "complete"])
async def test_scheduler_failure_pairs_outstanding_requests(phase: str) -> None:
    ctx, agents, tools = await harness(
        [multi_call([("c1", "p", "1"), ("c2", "p", "2"), ("c3", "p", "3")]), answer("continued")]
    )
    executed: list[str] = []

    async def execute(args: Mapping[str, Any], execution: Any) -> str:
        executed.append(args["id"])
        if args["id"] == "1":
            execution.defer_context(
                create_user_message(
                    content=[{"type": "text", "text": "committed context"}],
                    source={"kind": "p"},
                )
            )
        return f"done-{args['id']}"

    tools.register(
        define_tool(
            name="p",
            description="exclusive",
            parameters={"id": {"type": "string", "required": True}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
        )
    )
    executor = tools._select_executor(None)
    assert isinstance(executor, NativeToolExecutor)
    failure = RuntimeError(f"{phase} failed")
    stage = {"execution-mode": "_execution_mode", "prepare": "_prepare", "complete": "_complete"}
    original: Callable[..., Any] = getattr(executor, stage[phase])

    def call_id_of(subject: Any) -> str:
        scheduled_call = getattr(subject, "call", None)
        return str(subject["id"] if scheduled_call is None else scheduled_call.block["id"])

    def failing(subject: Any, *args: Any) -> Any:
        if call_id_of(subject) == "c2":
            raise failure
        return original(subject, *args)

    setattr(executor, stage[phase], failing)
    handle = await create(agents)
    agent = handle.agent
    agent.followup(user("go"))
    await agent.when_idle()

    events = agent.session.snapshot_events()
    calls = [event for event in events if event["type"] == "tool/call"]
    results = [event for event in events if event["type"] == "tool/result"]
    assert executed == (["1", "2"] if phase == "complete" else ["1"])
    unknown = "TOOL_NOT_STARTED" if phase == "execution-mode" else "TOOL_OUTCOME_UNKNOWN"
    assert [
        (event["data"]["message"]["toolCallId"], (event["data"].get("error") or {}).get("code"))
        for event in results
    ] == [("c1", None), ("c2", unknown), ("c3", "TOOL_NOT_STARTED")]
    assert results[0]["data"]["message"]["content"] == [{"type": "text", "text": "done-1"}]
    assert [event["data"]["message"]["isError"] for event in results[1:]] == [True, True]
    assert [event.get("sourceEventSeqs") for event in results] == [
        [calls[0]["seq"]],
        None if phase == "execution-mode" else [calls[1]["seq"]],
        None,
    ]
    assert [event["type"] for event in events[-3:]] == ["tool/result", "step/end", "turn/end"]
    assert events[-1]["data"]["reason"] == {
        "kind": "error",
        "error": {"message": f"{phase} failed", "code": "UNKNOWN"},
    }

    setattr(executor, stage[phase], original)
    agent.followup(user("continue"))
    await agent.when_idle()
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    adapter = llm._get_registration("test").adapter
    assert isinstance(adapter, ScriptedAdapter)
    flattened: list[str] = []
    for message in adapter.requests[1]["messages"]:
        if message["role"] == "tool":
            flattened.append(f"result:{message['toolCallId']}")
            continue
        for block in message["content"]:
            if block["type"] == "tool-call":
                flattened.append(f"call:{block['id']}")
            elif message["role"] == "user" and block["type"] == "text":
                flattened.append(f"user:{block['text']}")
    assert flattened == [
        "user:go",
        "call:c1",
        "call:c2",
        "call:c3",
        "result:c1",
        "result:c2",
        "result:c3",
        "user:committed context",
        "user:continue",
    ]
    assert of_type(agent, "turn/end")[-1]["data"]["reason"]["kind"] == "completed"
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_a_concluding_call_does_not_drop_its_siblings() -> None:
    ctx, agents, tools = await harness(
        [multi_call([("c1", "finish", "1"), ("c2", "echo", "2")]), answer("next turn")]
    )
    ran: list[str] = []

    async def finish(args: Mapping[str, Any], execution: Any) -> str:
        execution.conclude_turn()
        return "finished"

    async def echo(args: Mapping[str, Any], execution: object) -> str:
        ran.append(args["id"])
        return "echoed"

    for name, execute in (("finish", finish), ("echo", echo)):
        tools.register(
            define_tool(
                name=name,
                description=name,
                parameters={"id": {"type": "string", "required": True}},
                output={
                    "schema": {"type": "string"},
                    "render": lambda args, value: [{"type": "text", "text": value}],
                },
                execute=execute,
            )
        )
    handle = await create(agents)
    agent = handle.agent
    agent.followup(user("go"))
    await agent.when_idle()
    assert ran == ["2"]
    assert [summary[:2] for summary in result_summary(agent)] == [("c1", False), ("c2", False)]
    assert of_type(agent, "turn/end")[-1]["data"]["reason"] == {"kind": "completed"}
    await handle.dispose()
    await ctx.fiber.dispose()
