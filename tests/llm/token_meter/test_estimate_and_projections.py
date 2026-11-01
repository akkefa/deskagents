from __future__ import annotations

from typing import Any, cast

import pytest

from deskagents.core.session import SessionEvent, SessionLogOffset
from deskagents.llm import create_system_message, create_user_message
from deskagents.llm.token_meter import (
    BREAKDOWN_STATE_SCHEMA,
    context_breakdown_projection_definition,
    context_pressure_projection_definition,
    estimate_content,
    estimate_message,
    estimate_system_message,
    estimate_tools_tokens,
    token_usage_projection_definition,
)

ESTIMATE_SPEC = "packages/llm/token-meter/tests/context-breakdown-projection.spec.ts"
USAGE_SPEC = "packages/llm/token-meter/tests/token-usage-projection.spec.ts"


def event(seq: int, kind: str, data: Any, **metadata: Any) -> SessionEvent:
    return {"seq": seq, "time": seq, "type": kind, "data": data, **metadata}


@pytest.mark.upstream(
    f"{ESTIMATE_SPEC}::prices every content-block shape under the fixed heuristic"
)
def test_fixed_estimator_prices_builtin_and_extended_blocks() -> None:
    assert estimate_content([{"type": "text", "text": "abcd"}]) == 5
    assert estimate_content([{"type": "reasoning", "text": "abcdefgh"}]) == 6
    assert (
        estimate_content([{"type": "tool-call", "id": "c", "name": "bash", "arguments": '{"a":1}'}])
        == 7
    )
    unknown = {"type": "mystery", "payload": "abc"}
    assert estimate_content([unknown]) > 4


@pytest.mark.upstream(
    f"{ESTIMATE_SPEC}::prices the system node without block overhead and an empty prompt to zero"
)
def test_system_and_tool_estimates() -> None:
    assert estimate_system_message(create_system_message("")) == 0
    assert estimate_message(create_system_message("abcdefgh")) == 6
    header = {
        "config": {"provider": "test", "model": "test"},
        "tools": [{"name": "bash", "description": "run", "parameters": {"type": "object"}}],
    }
    assert estimate_tools_tokens(None) == 0
    assert estimate_tools_tokens(header) > 4


def usage_chunk(seq: int, usage: dict[str, int], turn: int = 1, step: int = 1) -> SessionEvent:
    return event(
        seq,
        "assistant/attempt",
        {
            "turn": turn,
            "step": step,
            "stream": [{"type": "chunk", "time": seq, "chunk": {"type": "usage", "usage": usage}}],
        },
    )


@pytest.mark.upstream(
    f"{USAGE_SPEC}::accumulates retried attempts while replacing samples within each attempt"
)
def test_usage_projection_replaces_then_accumulates() -> None:
    definition = token_usage_projection_definition
    state = definition.init({}, SessionLogOffset(0))
    state = definition.apply(state, usage_chunk(0, {"inputTokens": 10, "outputTokens": 2}))
    state = definition.apply(
        state,
        event(
            1,
            "assistant/message",
            {
                "turn": 1,
                "step": 1,
                "stream": [],
                "usage": {"inputTokens": 14, "outputTokens": 5, "cacheReadTokens": 8},
                "message": {},
            },
        ),
    )
    state = definition.apply(
        state, event(2, "llm/retry-started", {"turn": 1, "step": 1, "retry": 1})
    )
    state = definition.apply(
        state, usage_chunk(3, {"inputTokens": 7, "outputTokens": 3, "cacheWriteTokens": 1})
    )
    assert definition.wire is not None
    assert definition.wire.view(state) == {
        "uncachedInputTokens": 21,
        "outputTokens": 8,
        "cacheReadTokens": 8,
        "cacheWriteTokens": 1,
    }


@pytest.mark.upstream(
    f"{USAGE_SPEC}::carries the sample forward over surface growth and a compaction"
)
def test_context_pressure_projection_tracks_surface_growth() -> None:
    definition = context_pressure_projection_definition
    state = definition.init({}, SessionLogOffset(0))
    user = create_user_message(content=[{"type": "text", "text": "abcd"}], source={"kind": "user"})
    state = definition.apply(state, event(0, "user/message", user, surfaceOp="append"))
    state = definition.apply(
        state, usage_chunk(1, {"inputTokens": 100, "outputTokens": 200, "cacheReadTokens": 5})
    )
    state = definition.apply(
        state,
        event(
            2,
            "user/message",
            create_user_message(
                content=[{"type": "text", "text": "follow up"}], source={"kind": "user"}
            ),
            surfaceOp="append",
        ),
    )
    assert definition.wire is not None
    view = cast(dict[str, int], definition.wire.view(state))
    assert view["pressureTokens"] == 105
    assert view["projectedTokens"] > 105


@pytest.mark.upstream(
    f"{ESTIMATE_SPEC}::"
    "keeps the system node out of the message figure across appends and a system replacement"
)
def test_context_breakdown_and_type_adapter_checkpoint() -> None:
    definition = context_breakdown_projection_definition
    state = definition.init({}, SessionLogOffset(0))
    system = event(
        0,
        "system/message",
        {"message": create_system_message("You are terse."), "turn": 1, "step": 1},
        surfaceOp="append",
    )
    user = event(
        1,
        "user/message",
        create_user_message(content=[{"type": "text", "text": "abcd"}], source={"kind": "user"}),
        surfaceOp="append",
    )
    state = definition.apply(definition.apply(state, system), user)
    assert BREAKDOWN_STATE_SCHEMA.validate_python(state) == state
    assert definition.wire is not None
    assert definition.wire.view(state) == {
        "systemTokens": 8,
        "toolsTokens": 0,
        "messageTokens": 9,
    }
