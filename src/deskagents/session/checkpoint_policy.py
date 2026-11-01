"""Semantic durability checkpoints before model and tool side effects.

Upstream: packages/session/session-checkpoint-policy at ce7073d.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, cast

from deepseek_cordis import Context

from deskagents.core.session import SessionStore


def install_checkpoint_policy(ctx: Context, config: object = None) -> None:
    del config
    sessions = ctx.get("sessions")
    if not isinstance(sessions, SessionStore):
        raise RuntimeError("checkpoint policy requires SessionStore")

    def llm_stream(options: Mapping[str, Any], next_: Any) -> AsyncIterable[Mapping[str, Any]]:
        session = sessions.get(options["sessionId"]) if "sessionId" in options else None
        if session is None:
            return cast(AsyncIterable[Mapping[str, Any]], next_())

        async def checked() -> AsyncIterator[Mapping[str, Any]]:
            await sessions.flush(session)
            async for chunk in next_():
                yield chunk

        return checked()

    async def tools_execute(execution: Mapping[str, Any], next_: Any) -> object:
        agent = execution.get("agent")
        if agent is None or execution.get("parent") is not None:
            result = next_()
            return await result if inspect.isawaitable(result) else result
        await sessions.flush(agent.session)
        if getattr(execution["signal"], "aborted", False):
            return {
                "content": [{"type": "text", "text": "Error: tool call aborted before dispatch"}],
                "isError": True,
                "error": {
                    "message": "tool call aborted before dispatch",
                    "info": {
                        "name": "AbortError",
                        "code": "ABORTED_BEFORE_DISPATCH",
                    },
                },
            }
        result = next_()
        return await result if inspect.isawaitable(result) else result

    async def pre_step(payload: Mapping[str, Any], next_: Any) -> object:
        await sessions.flush(payload["agent"].session)
        result = next_()
        return await result if inspect.isawaitable(result) else result

    ctx.on("llm/stream", llm_stream)
    ctx.on("tools/execute", tools_execute)
    ctx.on("agent/pre-step", pre_step)


apply = install_checkpoint_policy

__all__ = ["apply", "install_checkpoint_policy"]
