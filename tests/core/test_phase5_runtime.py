from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from deepseek_cordis import Context, CordisError

from deskagents.core import AgentLoop, AgentRegistry, SystemPrompt, ToolsService, define_tool
from deskagents.core.session import CreateSessionOptions, SessionStore
from deskagents.core.tools import NativeToolExecutor
from deskagents.llm import (
    HarnessError,
    LlmAdapter,
    LlmRuntime,
    create_user_message,
    expand_assistant_stream,
)
from deskagents.session import SessionTitleService, persistence_jsonl
from deskagents.session.persistence import SessionHandle, SessionPersistence
from deskagents.testing import ScriptedAdapter


async def runtime(
    scripts: list[list[Mapping[str, Any]]],
) -> tuple[Context, Any, ScriptedAdapter]:
    ctx = Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    await ctx.plugin(SystemPrompt, {"includeHarnessIdentity": False, "personaPrefix": "Helpful"})
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(AgentLoop)
    adapter = ScriptedAdapter(scripts)
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    llm.register_adapter(["test"], adapter)
    return ctx, ctx.get("agents"), adapter


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/system-prompt/tests/system-prompt.spec.ts::"
    "assembles sections in order with context-resolved text and collected tools"
)
async def test_prompt_assembly() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt, {"includeHarnessIdentity": False, "personaPrefix": "A"})
    prompt = ctx.get("systemPrompt")
    assert isinstance(prompt, SystemPrompt)
    prompt.section({"name": "b", "order": 20, "text": lambda context: context["value"]})
    prompt.section({"name": "a", "order": 10, "text": "first"})
    prompt.variable("who", lambda context: context["who"])
    prompt.section({"name": "variable", "order": 30, "text": "Hi {{who}}"})
    prompt.tools(lambda context: {"schemas": [{"name": "z", "description": "Z", "parameters": {}}]})
    assembly = await prompt.assemble({"value": "second", "who": "Sol"})
    assert [item["name"] for item in assembly["sections"]] == [
        "deployment:persona-prefix",
        "a",
        "b",
        "variable",
        "deployment:persona-suffix",
    ]
    assert assembly["tools"][0]["name"] == "z"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/tools.spec.ts::"
    "registry round-trips a defineTool definition (register→schemas→execute)"
)
async def test_native_tool_round_trip() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)

    async def execute(args: Mapping[str, Any], execution: object) -> object:
        del execution
        return {"answer": args["value"] + 1}

    definition = define_tool(
        name="increment",
        description="Increment",
        parameters={"value": {"type": "integer", "required": True}},
        output={
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            "render": lambda args, value: [{"type": "text", "text": str(value["answer"])}],
        },
        execute=execute,
    )
    tools.register(definition)
    signal = type("Signal", (), {"aborted": False})()
    result = await tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "call-1",
                    "name": "increment",
                    "arguments": '{"value":4}',
                }
            ],
            "signal": signal,
        }
    )
    assert result["results"][0]["result"]["content"] == [{"type": "text", "text": "5"}]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_scripted_agent_no_tool_turn() -> None:
    ctx, agents, adapter = await runtime(
        [
            [
                {"type": "text-delta", "index": 0, "text": "hello"},
                {"type": "finish", "reason": {"kind": "stop"}},
            ]
        ]
    )
    handle = await agents.create(
        {
            "sessionId": "phase5-no-tool",
            "agentOptions": {"provider": "test", "model": "scripted"},
        }
    )
    handle.agent.followup(
        create_user_message(content=[{"type": "text", "text": "hi"}], source={"kind": "user"})
    )
    await handle.agent.when_idle()
    types = [event["type"] for event in handle.agent.session.snapshot_events()]
    assert types[-5:] == [
        "request/header",
        "request/context",
        "assistant/message",
        "step/end",
        "turn/end",
    ]
    assert adapter.requests[0]["messages"][-1]["role"] == "user"
    await handle.dispose()
    await ctx.fiber.dispose()


def tool_call(call_id: str = "call-1") -> list[Mapping[str, Any]]:
    return [
        {
            "type": "block-end",
            "index": 0,
            "block": {
                "type": "tool-call",
                "id": call_id,
                "name": "echo",
                "arguments": '{"value":"ok"}',
            },
        },
        {"type": "finish", "reason": {"kind": "stop"}},
    ]


def answer(text: str = "done") -> list[Mapping[str, Any]]:
    return [
        {"type": "text-delta", "index": 0, "text": text},
        {"type": "finish", "reason": {"kind": "stop"}},
    ]


async def register_echo(ctx: Context) -> None:
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)

    async def execute(args: Mapping[str, Any], execution: object) -> object:
        del execution
        return args["value"]

    tools.register(
        define_tool(
            name="echo",
            description="Echo",
            parameters={"value": {"type": "string", "required": True}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
        )
    )


@pytest.mark.asyncio
async def test_scripted_agent_single_tool_turn() -> None:
    ctx, agents, adapter = await runtime([tool_call(), answer()])
    await register_echo(ctx)
    handle = await agents.create(
        {
            "sessionId": "phase5-tool",
            "agentOptions": {"provider": "test", "model": "scripted"},
        }
    )
    handle.agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "use echo"}],
            source={"kind": "user"},
        )
    )
    await handle.agent.when_idle()
    types = [event["type"] for event in handle.agent.session.snapshot_events()]
    assert types.count("assistant/message") == 2
    assert types.count("tool/call") == types.count("tool/result") == 1
    assert len(adapter.requests) == 2
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_scripted_agent_recursive_tool_turn() -> None:
    ctx, agents, adapter = await runtime([tool_call("call-1"), tool_call("call-2"), answer()])
    await register_echo(ctx)
    handle = await agents.create(
        {
            "sessionId": "phase5-recursive",
            "agentOptions": {"provider": "test", "model": "scripted"},
        }
    )
    handle.agent.followup(
        create_user_message(content=[{"type": "text", "text": "twice"}], source={"kind": "user"})
    )
    await handle.agent.when_idle()
    types = [event["type"] for event in handle.agent.session.snapshot_events()]
    assert types.count("tool/result") == 2
    assert len(adapter.requests) == 3
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_scripted_agent_failed_step_is_balanced() -> None:
    ctx, agents, _ = await runtime(
        [
            [
                {
                    "type": "finish",
                    "reason": {"kind": "error", "failure": {"message": "boom", "code": "FAILED"}},
                }
            ]
        ]
    )
    handle = await agents.create(
        {
            "sessionId": "phase5-failed",
            "agentOptions": {"provider": "test", "model": "scripted"},
        }
    )
    handle.agent.followup(
        create_user_message(content=[{"type": "text", "text": "fail"}], source={"kind": "user"})
    )
    await handle.agent.when_idle()
    events = handle.agent.session.snapshot_events()
    assert events[-2]["type"] == "step/end"
    assert events[-1]["type"] == "turn/end"
    assert events[-1]["data"]["reason"]["kind"] == "error"
    await handle.dispose()
    await ctx.fiber.dispose()


class WaitingAdapter(LlmAdapter):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        async def generate() -> AsyncIterable[Mapping[str, Any]]:
            self.started.set()
            signal = options["signal"]
            while not signal.aborted:  # noqa: ASYNC110 - adapter observes the runtime signal
                await asyncio.sleep(0)
            yield {
                "type": "finish",
                "reason": {
                    "kind": "aborted",
                    "failure": {"message": "aborted", "code": "ABORTED"},
                },
            }

        return generate()


class PartialWaitingAdapter(LlmAdapter):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        async def generate() -> AsyncIterable[Mapping[str, Any]]:
            yield {"type": "text-delta", "index": 0, "text": "partial"}
            self.started.set()
            signal = options["signal"]
            while not signal.aborted:  # noqa: ASYNC110 - adapter observes the runtime signal
                await asyncio.sleep(0)

        return generate()


@pytest.mark.asyncio
async def test_scripted_agent_cancelled_turn_is_balanced() -> None:
    ctx, agents, _ = await runtime([])
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    waiting = WaitingAdapter()
    llm.register_adapter(["wait"], waiting)
    handle = await agents.create(
        {
            "sessionId": "phase5-cancelled",
            "agentOptions": {"provider": "wait", "model": "model"},
        }
    )
    handle.agent.followup(
        create_user_message(content=[{"type": "text", "text": "wait"}], source={"kind": "user"})
    )
    await waiting.started.wait()
    handle.agent.cancel({"kind": "user"}, keep_inbox=True)
    await handle.agent.when_idle()
    events = handle.agent.session.snapshot_events()
    assert events[-2]["type"] == "step/end"
    assert events[-1]["type"] == "turn/end"
    assert events[-1]["data"]["reason"] == {
        "kind": "aborted",
        "reason": {"kind": "user"},
    }
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_concurrency_safe_tools_use_bounded_parallel_pool() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    active = 0
    peak = 0
    release = asyncio.Event()
    pool_filled = asyncio.Event()

    async def execute(args: Mapping[str, Any], execution: object) -> object:
        nonlocal active, peak
        del args, execution
        active += 1
        peak = max(peak, active)
        if active == 2:
            pool_filled.set()
        await release.wait()
        active -= 1
        return "ok"

    tools.register(
        define_tool(
            name="parallel",
            description="Parallel",
            parameters={},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
            isConcurrencySafe=lambda args: True,
        )
    )
    signal = type("Signal", (), {"aborted": False})()
    task = asyncio.create_task(
        tools.execute(
            {
                "calls": [
                    {"id": f"call-{index}", "name": "parallel", "arguments": "{}"}
                    for index in range(3)
                ],
                "signal": signal,
                "maxParallelToolCalls": 2,
            }
        )
    )
    await pool_filled.wait()
    release.set()
    result = await task
    assert peak == 2
    assert [item["callId"] for item in result["results"]] == [
        "call-0",
        "call-1",
        "call-2",
    ]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_agent_can_resume_persisted_session(tmp_path: Path) -> None:
    ctx, agents, _ = await runtime([[*answer("first")], [*answer("second")]])
    await ctx.plugin(
        persistence_jsonl,
        {"root": str(tmp_path), "compression": "none"},
    )
    created = await agents.create(
        {
            "sessionId": "phase5-resume",
            "agentOptions": {"provider": "test", "model": "scripted"},
        }
    )
    created.agent.followup(
        create_user_message(content=[{"type": "text", "text": "one"}], source={"kind": "user"})
    )
    await created.agent.when_idle()
    await created.dispose()
    resumed = await agents.resume(
        {
            "resumeSessionId": "phase5-resume",
            "agentOptions": {"provider": "test", "model": "scripted"},
        }
    )
    resumed.agent.followup(
        create_user_message(content=[{"type": "text", "text": "two"}], source={"kind": "user"})
    )
    await resumed.agent.when_idle()
    assert (
        len(
            [
                event
                for event in resumed.agent.session.snapshot_events()
                if event["type"] == "assistant/message"
            ]
        )
        == 2
    )
    await resumed.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/agent.spec.ts::"
    "Agent > contains a throwing status listener on both transitions"
)
async def test_throwing_status_listener_is_contained(caplog: pytest.LogCaptureFixture) -> None:
    ctx, agents, _ = await runtime([answer("ok")])
    handle = await agents.create(
        {"sessionId": "status-listener", "agentOptions": {"provider": "test", "model": "scripted"}}
    )
    statuses: list[str] = []

    def broken(payload: Mapping[str, Any]) -> None:
        raise RuntimeError(f"bad {payload['status']} listener")

    ctx.on("agent/status", broken)
    ctx.on("agent/status", lambda payload: statuses.append(payload["status"]))
    with caplog.at_level(logging.WARNING, logger="deskagents.core.agent"):
        handle.agent.followup(
            create_user_message(content=[{"type": "text", "text": "go"}], source={"kind": "user"})
        )
        await handle.agent.when_idle()
    assert handle.agent.status == "idle"
    assert statuses == ["running", "idle"]
    assert handle.agent.session.snapshot_events()[-1]["data"]["reason"] == {"kind": "completed"}
    assert [record.getMessage() for record in caplog.records] == [
        'agent event "agent/status" listener threw: bad running listener',
        'agent event "agent/status" listener threw: bad idle listener',
    ]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/scoped.spec.ts::scoped execution dispatch > "
    "notifies every tools/result observer with the frozen final outcome and contains failures"
)
async def test_tools_result_observer_failures_are_contained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    await register_echo(ctx)
    seen: list[bool] = []

    def broken(execution: object, result: Mapping[str, Any]) -> None:
        raise RuntimeError("sync observer failure")

    async def rejected(execution: object, result: Mapping[str, Any]) -> None:
        raise RuntimeError("async observer failure")

    ctx.on("tools/result", lambda execution, result: seen.append(result["isError"]))
    ctx.on("tools/result", broken)
    ctx.on("tools/result", rejected)
    ctx.on("tools/result", lambda execution, result: seen.append(result["isError"]))
    signal = type("Signal", (), {"aborted": False})()
    with caplog.at_level(logging.WARNING, logger="deskagents.core.tools"):
        result = await tools.execute(
            {
                "calls": [{"id": "final", "name": "echo", "arguments": '{"value":"ok"}'}],
                "signal": signal,
            }
        )
        await asyncio.sleep(0)
    assert result["results"][0]["result"]["content"] == [{"type": "text", "text": "ok"}]
    assert seen == [False, False]
    assert [record.getMessage() for record in caplog.records] == [
        'tool "echo" (final): tools/result observer threw: sync observer failure',
        'tool "echo" (final): tools/result observer rejected: async observer failure',
    ]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/resume.spec.ts::"
    "the session-persistence Agent Note: AgentLoop factory create/resume > "
    "a seed append failure closes the fresh write handle and rethrows"
)
async def test_seed_append_failure_closes_the_write_handle(tmp_path: Path) -> None:
    ctx, agents, _ = await runtime([])
    await ctx.plugin(persistence_jsonl, {"root": str(tmp_path), "compression": "none"})
    persistence = ctx.get("sessionPersistence")
    assert isinstance(persistence, SessionPersistence)
    original_create = persistence.create
    closed: list[str] = []

    async def create(*args: Any, **kwargs: Any) -> SessionHandle:
        handle = await original_create(*args, **kwargs)
        close = handle.close

        async def failing_append(events: object) -> None:
            raise RuntimeError("seed append failed")

        async def spy_close() -> None:
            closed.append(handle.id)
            await close()

        handle.append = failing_append  # type: ignore[method-assign]
        handle.close = spy_close  # type: ignore[method-assign]
        return handle

    persistence.create = create  # type: ignore[method-assign]
    seed = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "turn/end",
            "seq": 1,
            "time": 2,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    with pytest.raises(RuntimeError, match="seed append failed"):
        await agents.create(
            {
                "sessionId": "seed-append-fail",
                "sessionOptions": CreateSessionOptions(seed=seed),
            }
        )
    assert closed == ["seed-append-fail"]
    assert agents.get("seed-append-fail") is None
    sessions = ctx.get("sessions")
    assert isinstance(sessions, SessionStore)
    assert sessions.get("seed-append-fail") is None
    with pytest.raises(Exception, match="not found"):
        await persistence.open("seed-append-fail", "write")
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_create_rejects_an_owner_disposed_during_storage_creation(tmp_path: Path) -> None:
    ctx, agents, _ = await runtime([])
    await ctx.plugin(persistence_jsonl, {"root": str(tmp_path), "compression": "none"})
    persistence = ctx.get("sessionPersistence")
    assert isinstance(persistence, SessionPersistence)
    owner = await ctx.plugin(lambda _ctx, _config: None)
    original_create = persistence.create
    closed: list[str] = []

    async def create(*args: Any, **kwargs: Any) -> SessionHandle:
        handle = await original_create(*args, **kwargs)
        close = handle.close

        async def spy_close() -> None:
            closed.append(handle.id)
            await close()

        handle.close = spy_close  # type: ignore[method-assign]
        await owner.dispose()
        return handle

    persistence.create = create  # type: ignore[method-assign]
    with pytest.raises(CordisError, match="INACTIVE_EFFECT"):
        await agents.create({"sessionId": "owner-gone"}, ctx=owner.ctx)
    assert closed == ["owner-gone"]
    assert agents.get("owner-gone") is None
    await ctx.fiber.dispose()


def user(text: str) -> Mapping[str, Any]:
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})


def record_frames(ctx: Context, agent: object) -> list[Mapping[str, Any]]:
    frames: list[Mapping[str, Any]] = []
    ctx.on(
        "agent/assistant-stream",
        lambda payload: frames.append(payload["frame"]) if payload["agent"] is agent else None,
    )
    return frames


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/loop.spec.ts::agent loop > "
    "publishes one dense live attempt and commits one v2 message with the exact embedded stream"
)
async def test_assistant_stream_publishes_one_committed_attempt() -> None:
    ctx, agents, _ = await runtime([answer("live")])
    handle = await agents.create(
        {"sessionId": "live-attempt", "agentOptions": {"provider": "test", "model": "scripted"}}
    )
    agent = handle.agent
    frames = record_frames(ctx, agent)
    committed_after_message: list[bool] = []

    def on_frame(payload: Mapping[str, Any]) -> None:
        frame = payload["frame"]
        if frame["type"] == "end" and frame["outcome"]["kind"] == "committed":
            last = agent.session.snapshot_events()[-1]
            committed_after_message.append(last["type"] == "assistant/message")

    ctx.on("agent/assistant-stream", on_frame)
    agent.followup(user("stream this"))
    await agent.when_idle()
    assert frames[0]["type"] == "start"
    assert [frame["revision"] for frame in frames] == list(range(1, len(frames) + 1))
    assert len({frame["attemptId"] for frame in frames}) == 1
    chunks = [frame for frame in frames if frame["type"] == "chunk"]
    assert [frame["index"] for frame in chunks] == list(range(len(chunks)))
    message = [
        event for event in agent.session.snapshot_events() if event["type"] == "assistant/message"
    ][-1]
    embedded = [member["chunk"] for member in expand_assistant_stream(message["data"]["stream"])]
    assert embedded == [frame["chunk"] for frame in chunks]
    assert frames[-1] == {
        "type": "end",
        "index": len(chunks),
        "outcome": {"kind": "committed", "eventType": "assistant/message", "seq": message["seq"]},
        "attemptId": frames[0]["attemptId"],
        "revision": len(frames),
    }
    assert committed_after_message == [True]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/loop.spec.ts::agent loop > "
    "abandons the live row when durable Assistant settlement is rejected"
)
async def test_assistant_stream_abandons_a_rejected_settlement() -> None:
    ctx, agents, _ = await runtime([answer("cannot commit")])
    handle = await agents.create(
        {"sessionId": "abandoned", "agentOptions": {"provider": "test", "model": "scripted"}}
    )
    agent = handle.agent
    frames = record_frames(ctx, agent)
    append = agent.session.append

    def reject_settlement(event_type: str, *args: Any, **kwargs: Any) -> Any:
        if event_type == "assistant/message":
            raise RuntimeError("settlement rejected")
        return append(event_type, *args, **kwargs)

    agent.session.append = reject_settlement
    agent.followup(user("stream this"))
    await agent.when_idle()
    assert frames[-1]["type"] == "end"
    assert frames[-1]["outcome"] == {"kind": "abandoned"}
    events = agent.session.snapshot_events()
    assert not any(event["type"] == "assistant/message" for event in events)
    assert events[-1]["data"]["reason"]["kind"] == "error"
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_assistant_stream_commits_a_cancelled_empty_attempt() -> None:
    ctx, agents, _ = await runtime([])
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    waiting = WaitingAdapter()
    llm.register_adapter(["wait"], waiting)
    handle = await agents.create(
        {"sessionId": "cancelled-attempt", "agentOptions": {"provider": "wait", "model": "m"}}
    )
    frames = record_frames(ctx, handle.agent)
    handle.agent.followup(user("wait"))
    await waiting.started.wait()
    handle.agent.cancel({"kind": "user"})
    await handle.agent.when_idle()
    assert frames[0]["type"] == "start"
    assert frames[-1]["type"] == "end"
    assert frames[-1]["outcome"]["kind"] == "committed"
    assert frames[-1]["outcome"]["eventType"] == "assistant/attempt"
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/cancel.spec.ts::"
    "cancel mid-stream finalizes the streamed prefix onto the surface"
)
async def test_cancelled_stream_commits_its_visible_prefix() -> None:
    ctx, agents, _ = await runtime([])
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    waiting = PartialWaitingAdapter()
    llm.register_adapter(["partial"], waiting)
    handle = await agents.create(
        {
            "sessionId": "cancelled-prefix",
            "agentOptions": {"provider": "partial", "model": "m"},
        }
    )
    frames = record_frames(ctx, handle.agent)
    handle.agent.followup(user("wait"))
    await waiting.started.wait()
    handle.agent.cancel({"kind": "user"})
    await handle.agent.when_idle()
    messages = events_of(handle.agent, "assistant/message")
    assert messages[-1]["data"]["message"]["content"] == [{"type": "text", "text": "partial"}]
    assert messages[-1]["data"]["interrupted"] is True
    assert frames[-1]["outcome"] == {
        "kind": "committed",
        "eventType": "assistant/message",
        "seq": messages[-1]["seq"],
    }
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/loop.spec.ts::agent loop > "
    "agent/turn-stopping can steer another step (/loop pattern)"
)
async def test_turn_stopping_listener_can_steer_another_step() -> None:
    ctx, agents, adapter = await runtime([answer("step 1"), answer("step 2"), answer("step 3")])
    handle = await agents.create(
        {"sessionId": "loop-pattern", "agentOptions": {"provider": "test", "model": "scripted"}}
    )

    def step_count() -> int:
        events = handle.agent.session.snapshot_events()
        return sum(1 for event in events if event["type"] == "step/end")

    def continue_turn(payload: Mapping[str, Any]) -> None:
        if step_count() < 3:
            payload["agent"].steer(user("continue"))

    ctx.on("agent/turn-stopping", continue_turn)
    handle.agent.followup(user("go"))
    await handle.agent.when_idle()
    assert step_count() == 3
    assert len(adapter.requests) == 3
    turn_ends = [e for e in handle.agent.session.snapshot_events() if e["type"] == "turn/end"]
    assert len(turn_ends) == 1
    assert turn_ends[0]["data"]["reason"] == {"kind": "completed"}
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/contract-regressions.spec.ts::"
    "plugin exceptions are contained > "
    "a throwing agent/turn-stopping listener ends the turn with an error, loop survives"
)
async def test_throwing_turn_stopping_listener_ends_the_turn_with_an_error() -> None:
    ctx, agents, adapter = await runtime([answer("one"), answer("two")])
    handle = await agents.create(
        {"sessionId": "broken-stopping", "agentOptions": {"provider": "test", "model": "scripted"}}
    )
    threw: list[bool] = []

    async def broken(payload: Mapping[str, Any]) -> None:
        if not threw:
            threw.append(True)
            raise RuntimeError("broken continuation plugin")

    ctx.on("agent/turn-stopping", broken)
    handle.agent.followup(user("first"))
    await handle.agent.when_idle()
    reason = handle.agent.session.snapshot_events()[-1]["data"]["reason"]
    assert reason["kind"] == "error"
    assert reason["error"]["message"] == "broken continuation plugin"
    handle.agent.followup(user("second"))
    await handle.agent.when_idle()
    assert len(adapter.requests) == 2
    assert handle.agent.status == "idle"
    await handle.dispose()
    await ctx.fiber.dispose()


def events_of(agent: Any, type_: str) -> list[Mapping[str, Any]]:
    return [event for event in agent.session.snapshot_events() if event["type"] == type_]


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/request-reconstruction.spec.ts::"
    "request stability across the loop > "
    "starts a new request series only when the admitted step explicitly asks for one"
)
async def test_request_header_is_logged_once_unless_a_series_starts() -> None:
    ctx, agents, _ = await runtime([answer("one"), answer("two")])
    handle = await agents.create(
        {"sessionId": "series", "agentOptions": {"provider": "test", "model": "scripted"}}
    )

    async def second_turn_series(payload: Mapping[str, Any], next_: Any) -> Mapping[str, Any]:
        decision: Mapping[str, Any] = await next_()
        if decision["kind"] == "enter" and payload["turn"] == 2:
            return {**decision, "startsRequestSeries": True}
        return decision

    ctx.on("agent/pre-step", second_turn_series)
    for text in ("first", "second series"):
        handle.agent.followup(user(text))
        await handle.agent.when_idle()
    headers = events_of(handle.agent, "request/header")
    assert [event["data"]["reason"] for event in headers] == ["initial", "series"]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/agent-loop/tests/tool-updates.spec.ts::tool update emission > "
    "logs additions bound to the new header and removals without one, after the first header"
)
async def test_tool_registry_changes_append_developer_messages() -> None:
    ctx, agents, adapter = await runtime([answer("ok") for _ in range(4)])
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)

    def tool(name: str) -> Any:
        async def execute(args: Mapping[str, Any], execution: object) -> object:
            return "done"

        return tools.register(
            define_tool(
                name=name,
                description=f"{name} tool",
                parameters={},
                output={
                    "schema": {"type": "string"},
                    "render": lambda args, value: [{"type": "text", "text": value}],
                },
                execute=execute,
            )
        )

    handle = await agents.create(
        {"sessionId": "tool-updates", "agentOptions": {"provider": "test", "model": "scripted"}}
    )
    agent = handle.agent

    async def send(text: str) -> None:
        agent.followup(user(text))
        await agent.when_idle()

    dispose_search = tool("search")
    await send("first")
    assert events_of(agent, "developer/message") == []

    tool("fetch")
    await send("second")
    [addition] = events_of(agent, "developer/message")
    assert addition["surfaceOp"] == "append"
    assert addition["data"]["turn"] == 2 and addition["data"]["step"] == 1
    assert addition["data"]["headerSeq"] == events_of(agent, "request/header")[1]["seq"]
    assert addition["data"]["message"]["role"] == "developer"
    assert addition["data"]["message"]["source"] == {"kind": "tool-registry"}
    assert addition["data"]["message"]["content"] == [
        {"type": "tool-addition", "toolName": "fetch"}
    ]

    dispose_search()
    await send("third")
    removal = events_of(agent, "developer/message")[1]
    assert "headerSeq" not in removal["data"]
    assert removal["data"]["message"]["content"] == [{"type": "tool-removal", "toolName": "search"}]

    await send("fourth")
    assert len(events_of(agent, "developer/message")) == 2
    assert [len(request["toolHistory"]["updates"]) for request in adapter.requests] == [0, 1, 2, 2]
    for request in adapter.requests:
        assert not [message for message in request["messages"] if message["role"] == "developer"]
    assert [
        sorted(schema["name"] for schema in request["tools"]) for request in adapter.requests
    ] == [
        ["search"],
        ["fetch", "search"],
        ["fetch"],
        ["fetch"],
    ]
    await handle.dispose()
    await ctx.fiber.dispose()


async def tool_runtime() -> tuple[Context, ToolsService]:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    return ctx, tools


def register_failing(tools: ToolsService, error: Exception) -> None:
    async def execute(args: Mapping[str, Any], execution: object) -> object:
        raise error

    tools.register(
        define_tool(
            name="boom",
            description="Fails",
            parameters={},
            output={"schema": {"type": "null"}, "render": lambda args, value: []},
            execute=execute,
        )
    )


async def run_boom(tools: ToolsService) -> Mapping[str, Any]:
    signal = type("Signal", (), {"aborted": False})()
    batch = await tools.execute(
        {"calls": [{"id": "c1", "name": "boom", "arguments": "{}"}], "signal": signal}
    )
    result: Mapping[str, Any] = batch["results"][0]["result"]
    return result


async def run_tool(
    tools: ToolsService,
    name: str,
    *,
    agent: object = None,
) -> Mapping[str, Any]:
    signal = type("Signal", (), {"aborted": False})()
    batch = await tools.execute(
        {
            "calls": [{"id": "c1", "name": name, "arguments": "{}"}],
            "signal": signal,
            **({"agent": agent} if agent is not None else {}),
        }
    )
    result: Mapping[str, Any] = batch["results"][0]["result"]
    return result


def register_integer_tool(
    tools: ToolsService,
    *,
    execute: Any,
    render: Any,
    presentation_meta: Any = None,
) -> None:
    tools.register(
        define_tool(
            name="number",
            description="Number",
            parameters={},
            output={
                "schema": {"type": "integer"},
                "render": render,
                **(
                    {"presentationMeta": presentation_meta} if presentation_meta is not None else {}
                ),
            },
            execute=execute,
        )
    )


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/tools.spec.ts::ToolRuntime > "
    "keeps value/meta through content replacement and recomputes both projections "
    "after value replacement"
)
async def test_post_execute_value_replacement_reprojects_the_result() -> None:
    ctx, tools = await tool_runtime()
    rendered: list[int] = []

    async def execute(args: Mapping[str, Any], execution: object) -> int:
        del args, execution
        return 1

    def render(args: Mapping[str, Any], value: int) -> list[Mapping[str, Any]]:
        del args
        rendered.append(value)
        return [{"type": "text", "text": str(value)}]

    register_integer_tool(
        tools,
        execute=execute,
        render=render,
        presentation_meta=lambda args, value: {"projected": value},
    )
    ctx.on(
        "tools/post-execute",
        lambda execution, result, next_: {"kind": "accept", "value": 2},
    )
    result = await run_tool(tools, "number")
    assert result == {
        "isError": False,
        "value": 2,
        "content": [{"type": "text", "text": "2"}],
        "meta": {"projected": 2},
    }
    assert rendered == [1, 2]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/tools.spec.ts::ToolRuntime > "
    "preserves tool-deferred, execute-wrapper, and post-execute contexts in order"
)
async def test_tool_contexts_survive_each_pipeline_layer_in_order() -> None:
    ctx, tools = await tool_runtime()
    body = user("body")
    wrapper = user("wrapper")
    policy = user("policy")

    async def execute(args: Mapping[str, Any], execution: Any) -> int:
        del args
        execution.defer_context(body)
        return 1

    register_integer_tool(
        tools,
        execute=execute,
        render=lambda args, value: [{"type": "text", "text": str(value)}],
    )

    async def around(execution: object, next_: Any) -> Mapping[str, Any]:
        del execution
        result = await next_()
        return {
            **result,
            "additionalContexts": [*result.get("additionalContexts", ()), wrapper],
        }

    ctx.on("tools/execute", around)
    ctx.on(
        "tools/post-execute",
        lambda execution, result, next_: {
            "kind": "accept",
            "additionalContexts": [policy],
        },
    )
    result = await run_tool(tools, "number")
    assert result["additionalContexts"] == [body, wrapper, policy]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/tools.spec.ts::ToolRuntime > ask routing through ctx.approval > "
    "dispatches the tool when the answerer grants allowed-once, forwarding the ask fields"
)
async def test_ask_uses_the_optional_approval_service() -> None:
    ctx, tools = await tool_runtime()
    requests: list[Mapping[str, Any]] = []

    class Approval:
        async def request(self, request: Mapping[str, Any]) -> str:
            requests.append(request)
            return "allowed-once"

    ctx.provide("approval", Approval())

    async def execute(args: Mapping[str, Any], execution: object) -> int:
        del args, execution
        return 1

    register_integer_tool(
        tools,
        execute=execute,
        render=lambda args, value: [{"type": "text", "text": str(value)}],
    )
    ctx.on(
        "tools/pre-execute",
        lambda execution, next_: {
            "kind": "ask",
            "reason": "dangerous",
            "displayReason": {"en": "Run it?"},
        },
    )
    agent = object()
    result = await run_tool(tools, "number", agent=agent)
    assert result["isError"] is False
    assert requests == [
        {
            "agent": agent,
            "toolName": "number",
            "callId": "c1",
            "reason": "dangerous",
            "displayReason": {"en": "Run it?"},
            "signal": requests[0]["signal"],
        }
    ]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/tools.spec.ts::ToolRuntime > "
    "an ask decision with no reason degrades to deny with a default message"
)
async def test_ask_without_approval_degrades_to_denial() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: Mapping[str, Any], execution: object) -> int:
        del args, execution
        raise AssertionError("denied tool must not run")

    register_integer_tool(
        tools,
        execute=execute,
        render=lambda args, value: [{"type": "text", "text": str(value)}],
    )
    ctx.on("tools/pre-execute", lambda execution, next_: {"kind": "ask"})
    result = await run_tool(tools, "number", agent=object())
    assert result["isError"] is True
    assert result["error"]["message"] == 'tool "number" requires approval (not yet supported)'
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/tools.spec.ts::ToolRuntime > "
    "a thrown tool is normalized to an isError result BEFORE a tools/execute listener sees next()"
)
async def test_tools_execute_wrapper_sees_the_normalized_error() -> None:
    ctx, tools = await tool_runtime()
    register_failing(tools, HarnessError("kaboom", "BOOM"))
    seen: list[Mapping[str, Any]] = []

    async def around(execution: object, next_: Any) -> Mapping[str, Any]:
        result: Mapping[str, Any] = await next_()
        seen.append({"isError": result["isError"], "error": result.get("error")})
        return result

    ctx.on("tools/execute", around)
    result = await run_boom(tools)
    assert seen == [
        {
            "isError": True,
            "error": {"message": "kaboom", "info": {"name": "HarnessError", "code": "BOOM"}},
        }
    ]
    assert result["isError"] is True
    assert result["content"][0]["text"] == "Error: kaboom"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    "packages/core/tools/tests/tools.spec.ts::ToolRuntime > "
    "a thrown tool normalized inside tools/execute still reaches post-execute"
)
async def test_thrown_tool_still_reaches_post_execute() -> None:
    ctx, tools = await tool_runtime()
    register_failing(tools, RuntimeError("exploded"))
    post_saw: list[bool] = []

    def post(execution: object, result: Mapping[str, Any], next_: Any) -> object:
        post_saw.append(result["isError"])
        return next_()

    ctx.on("tools/execute", lambda execution, next_: next_())
    ctx.on("tools/post-execute", post)
    result = await run_boom(tools)
    assert post_saw == [True]
    assert result["isError"] is True
    assert result["content"][0]["text"] == "Error: exploded"
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_fallback_session_title_is_logged() -> None:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(
        SessionTitleService,
        {"fallbackMaxWords": 3, "fallbackMaxBytes": 30, "maxTitleBytes": 50},
    )
    sessions = ctx.get("sessions")
    assert isinstance(sessions, SessionStore)
    session = sessions.create("title")
    session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "  A useful title with extras  "}],
            source={"kind": "user"},
        ),
        surface_op="append",
    )
    await asyncio.sleep(0)
    title = ctx.get("sessionTitle")
    assert isinstance(title, SessionTitleService)
    snapshot = title.get(session)
    assert snapshot is not None
    assert snapshot["title"] == "A useful title"
    await ctx.fiber.dispose()
