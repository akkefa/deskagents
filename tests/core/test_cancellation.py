"""Ported from packages/core/agent-loop/tests/cancel.spec.ts: cooperative ``Agent.cancel()``."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterable, Mapping, Sequence
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core import AgentLoop, AgentRegistry, SystemPrompt, ToolsService, define_tool
from deskagents.core.session import SessionStore
from deskagents.core.tools import TOOL_ABORTED, TOOL_ABORTED_BEFORE_DISPATCH, NativeToolExecutor
from deskagents.llm import LlmAdapter, LlmRuntime, create_user_message

CANCEL = "packages/core/agent-loop/tests/cancel.spec.ts::Agent.cancel() > "
LOOP = "packages/core/agent-loop/tests/loop.spec.ts::agent loop > "

HANG = "hang"


class MockAdapter(LlmAdapter):
    """Scripted responses; ``"hang"`` streams nothing until the request is aborted."""

    def __init__(self, scripts: Sequence[object]) -> None:
        self._scripts: deque[object] = deque(scripts)
        self.requests: list[Mapping[str, Any]] = []
        self.hanging = asyncio.Event()

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        self.requests.append(options)
        script = self._scripts.popleft()

        async def generate() -> AsyncIterable[Mapping[str, Any]]:
            if script == HANG:
                self.hanging.set()
                await options["signal"].wait()
                return
            assert isinstance(script, list)
            for chunk in script:
                yield chunk

        return generate()


async def harness(adapter: LlmAdapter) -> tuple[Context, AgentRegistry, ToolsService]:
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
    llm.register_adapter(["mock"], adapter)
    agents = ctx.get("agents")
    assert isinstance(agents, AgentRegistry)
    return ctx, agents, tools


async def create(agents: AgentRegistry, session_id: str) -> Any:
    return await agents.create(
        {"sessionId": session_id, "agentOptions": {"provider": "mock", "model": "mock"}}
    )


def user(text: str) -> Mapping[str, Any]:
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})


def text_response(text: str, **finish: Any) -> list[Mapping[str, Any]]:
    return [
        {"type": "text-delta", "index": 0, "text": text},
        {"type": "finish", "reason": {"kind": "stop"}, **finish},
    ]


def tool_call(call_id: str, name: str) -> list[Mapping[str, Any]]:
    block = {"type": "tool-call", "id": call_id, "name": name, "arguments": "{}"}
    return [
        {"type": "block-end", "index": 0, "block": block},
        {"type": "finish", "reason": {"kind": "stop"}},
    ]


def string_tool(name: str, execute: Any) -> Mapping[str, Any]:
    return define_tool(
        name=name,
        description=name,
        parameters={},
        output={
            "schema": {"type": "string"},
            "render": lambda args, value: [{"type": "text", "text": value}],
        },
        execute=execute,
    )


def of_type(agent: Any, kind: str) -> list[Mapping[str, Any]]:
    return [event for event in agent.session.snapshot_events() if event["type"] == kind]


def reasons(agent: Any) -> list[Mapping[str, Any]]:
    return [event["data"]["reason"] for event in of_type(agent, "turn/end")]


def user_texts(agent: Any) -> list[str]:
    return [
        event["data"]["content"][0]["text"]
        for event in of_type(agent, "user/message")
        if event["data"]["source"]["kind"] == "user"
    ]


def unanswered(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    asked = [
        block["id"]
        for message in messages
        if message["role"] == "assistant"
        for block in message["content"]
        if block["type"] == "tool-call"
    ]
    answered = {message.get("toolCallId") for message in messages if message["role"] == "tool"}
    return [call_id for call_id in asked if call_id not in answered]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{CANCEL}cancel from an assistant/message observer skips execution but balances replay"
)
async def test_cancel_after_the_assistant_message_skips_tools_but_balances_replay() -> None:
    adapter = MockAdapter(
        [tool_call("c1", "danger"), text_response("recovered after cancellation")]
    )
    ctx, agents, tools = await harness(adapter)
    executions = 0

    async def danger(args: Mapping[str, Any], execution: object) -> str:
        nonlocal executions
        executions += 1
        return "ran"

    tools.register(string_tool("danger", danger))
    handle = await create(agents, "cancel-after-assistant-message")
    agent = handle.agent

    def cancel_on_message(session: Any, event: Mapping[str, Any]) -> None:
        if session is agent.session and event["type"] == "assistant/message":
            agent.cancel({"kind": "user"})

    dispose = ctx.on("session/event", cancel_on_message)
    agent.followup(user("go"))
    await agent.when_idle()
    dispose()

    assert executions == 0
    assert reasons(agent) == [{"kind": "aborted", "reason": {"kind": "user"}}]
    assert of_type(agent, "tool/call")[0]["data"]["callId"] == "c1"
    result = of_type(agent, "tool/result")[0]["data"]
    assert result["message"]["role"] == "tool"
    assert result["message"]["source"] == {"kind": "tool", "callId": "c1"}
    assert result["message"]["isError"] is True
    assert result["error"] == {"name": "AbortError", "code": TOOL_ABORTED_BEFORE_DISPATCH}

    agent.followup(user("continue safely"))
    await agent.when_idle()
    replayed = next(m for m in adapter.requests[1]["messages"] if m["role"] == "tool")
    assert replayed["toolCallId"] == "c1"
    assert replayed["isError"] is True
    assert reasons(agent) == [
        {"kind": "aborted", "reason": {"kind": "user"}},
        {"kind": "completed"},
    ]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{CANCEL}retains terminal replay state when cancellation races the final stream chunk"
)
async def test_listener_cancel_on_the_final_chunk_keeps_replay_state() -> None:
    replay_state = {"response": {"id": "response"}, "blocks": ["text-meta"]}
    adapter = MockAdapter([text_response("complete", replayState=replay_state)])
    ctx, agents, _ = await harness(adapter)
    handle = await create(agents, "terminal-cancel-replay")
    agent = handle.agent

    def on_frame(payload: Mapping[str, Any]) -> None:
        frame = payload["frame"]
        if payload["agent"] is agent and frame["type"] == "chunk":
            if frame["chunk"]["type"] == "finish":
                agent.cancel({"kind": "user"})

    ctx.on("agent/assistant-stream", on_frame)
    agent.followup(user("go"))
    # The agent cancels itself; waiting for it must not raise into this task.
    await agent.when_idle()
    message = of_type(agent, "assistant/message")[0]["data"]
    assert message["message"]["source"]["replayState"] == replay_state
    assert message["interrupted"] is True
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{CANCEL}cancel({{ keepInbox: true }}) parks queued work after an active turn aborts"
)
async def test_keep_inbox_cancel_parks_queued_work_until_the_next_wake() -> None:
    adapter = MockAdapter([HANG, text_response("preserved reply"), text_response("wake reply")])
    ctx, agents, _ = await harness(adapter)
    handle = await create(agents, "keep-after-abort")
    agent = handle.agent
    agent.followup(user("active"))
    await adapter.hanging.wait()
    agent.followup(user("preserved"))
    agent.cancel({"kind": "user"}, keep_inbox=True)
    await agent.when_idle()

    assert user_texts(agent) == ["active"]
    assert len(agent.inbox.next_turn) == 1
    assert len(adapter.requests) == 1

    agent.followup(user("wake it"))
    await agent.when_idle()
    assert user_texts(agent) == ["active", "preserved", "wake it"]
    assert len(adapter.requests) == 3
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{CANCEL}cancel({{ keepInbox: true }}) latches a waking send landing in the abort-to-idle "
    "window"
)
async def test_waking_send_after_abort_is_latched_and_replayed() -> None:
    adapter = MockAdapter([HANG, text_response("B reply")])
    ctx, agents, _ = await harness(adapter)
    handle = await create(agents, "latch-window")
    agent = handle.agent
    agent.followup(user("active"))
    await adapter.hanging.wait()
    agent.cancel({"kind": "user"}, keep_inbox=True)
    agent.followup(user("B"))
    await agent.when_idle()

    assert user_texts(agent) == ["active", "B"]
    assert len(adapter.requests) == 2
    assert len(agent.inbox.next_turn) == 0
    assert reasons(agent) == [
        {"kind": "aborted", "reason": {"kind": "user"}},
        {"kind": "completed"},
    ]
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{LOOP}a tool can conclude the turn despite owing a follow-up request")
async def test_a_tool_can_conclude_the_turn() -> None:
    adapter = MockAdapter([tool_call("c1", "echo")])
    ctx, agents, tools = await harness(adapter)

    async def echo(args: Mapping[str, Any], execution: Any) -> str:
        execution.conclude_turn()
        return "x"

    tools.register(string_tool("echo", echo))
    handle = await create(agents, "a1")
    agent = handle.agent
    agent.followup(user("go"))
    await agent.when_idle()
    assert len(adapter.requests) == 1
    assert of_type(agent, "tool/result")
    await handle.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_cancel_during_a_tool_lets_it_finish_and_records_aborted() -> None:
    adapter = MockAdapter([tool_call("c1", "slow"), text_response("after")])
    ctx, agents, tools = await harness(adapter)
    entered, state = asyncio.Event(), {}

    async def slow(args: Mapping[str, Any], execution: object) -> str:
        entered.set()
        await asyncio.sleep(0.01)
        state["completed"] = True
        return "finished"

    tools.register(string_tool("slow", slow))
    handle = await create(agents, "cancel-running-tool")
    agent = handle.agent
    agent.followup(user("go"))
    await entered.wait()
    agent.cancel({"kind": "user"})
    await agent.when_idle()

    assert state == {"completed": True}
    result = of_type(agent, "tool/result")[0]["data"]
    assert result["message"]["content"] == [{"type": "text", "text": "Error: tool call aborted"}]
    assert result["error"] == {"name": "AbortError", "code": TOOL_ABORTED}
    assert reasons(agent) == [{"kind": "aborted", "reason": {"kind": "user"}}]

    agent.followup(user("again"))
    await agent.when_idle()
    assert unanswered(adapter.requests[1]["messages"]) == []
    await handle.dispose()
    await ctx.fiber.dispose()
