from __future__ import annotations

from typing import Any

import pytest

from deskagents.core.session import SessionEvent
from deskagents.llm.token_meter import derive_turn_token_usage

SPEC = "packages/llm/token-meter/tests/turn-usage.spec.ts"


def event(seq: int, kind: str, data: Any) -> SessionEvent:
    return {"seq": seq, "time": seq, "type": kind, "data": data}


def message(
    seq: int,
    usage: dict[str, int] | None,
    *,
    step: int = 1,
    provider: str = "deepseek",
    model: str = "deepseek-chat",
) -> SessionEvent:
    data: dict[str, Any] = {
        "turn": 1,
        "step": step,
        "stream": [],
        "message": {
            "id": f"message-{seq}",
            "role": "assistant",
            "content": [],
            "source": {"kind": "model", "provider": provider, "model": model},
        },
    }
    if usage is not None:
        data["usage"] = usage
    return event(seq, "assistant/message", data)


def complete(*middle: SessionEvent) -> list[SessionEvent]:
    return [
        event(0, "turn/start", {"turn": 1}),
        event(1, "step/start", {"turn": 1, "step": 1}),
        *middle,
        event(90, "step/end", {"turn": 1, "step": 1}),
        event(91, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    ]


@pytest.mark.upstream(f"{SPEC}::preserves authoritative totals and explicit optional buckets")
def test_preserves_exact_usage_and_route() -> None:
    result = derive_turn_token_usage(
        complete(
            message(
                2,
                {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "totalTokens": 170,
                    "cacheReadTokens": 50,
                    "cacheWriteTokens": 0,
                    "reasoningTokens": 8,
                },
            )
        )
    )
    assert result == {
        "uncachedInputTokens": 100,
        "outputTokens": 20,
        "totalTokens": 170,
        "cacheReadTokens": 50,
        "cacheWriteTokens": 0,
        "reasoningTokens": 8,
        "routes": [{"provider": "deepseek", "model": "deepseek-chat"}],
    }


@pytest.mark.upstream(f"{SPEC}::derives an exact total only when both cache buckets are present")
def test_derives_total_only_from_complete_prompt_buckets() -> None:
    exact = derive_turn_token_usage(
        complete(
            message(
                2,
                {
                    "inputTokens": 10,
                    "outputTokens": 4,
                    "cacheReadTokens": 2,
                    "cacheWriteTokens": 1,
                },
            )
        )
    )
    assert exact is not None and exact["totalTokens"] == 17
    assert (
        derive_turn_token_usage(
            complete(
                message(
                    2,
                    {"inputTokens": 10, "outputTokens": 4, "cacheReadTokens": 2},
                )
            )
        )
        is None
    )


@pytest.mark.upstream(f"{SPEC}::counts an error-finished attempt once across its retry boundary")
def test_aggregates_retry_attempts() -> None:
    first = {"inputTokens": 100, "outputTokens": 20, "totalTokens": 170, "cacheReadTokens": 50}
    second = {"inputTokens": 40, "outputTokens": 10, "totalTokens": 70, "cacheReadTokens": 20}
    events = complete(
        event(
            2,
            "assistant/attempt",
            {
                "turn": 1,
                "step": 1,
                "stream": [{"type": "chunk", "chunk": {"type": "usage", "usage": first}}],
            },
        ),
        event(3, "llm/retry", {"turn": 1, "step": 1}),
        event(4, "llm/retry-started", {"turn": 1, "step": 1, "retry": 1}),
        message(5, second),
    )
    assert derive_turn_token_usage(events) == {
        "uncachedInputTokens": 140,
        "outputTokens": 30,
        "totalTokens": 240,
        "cacheReadTokens": 70,
    }


@pytest.mark.upstream(f"{SPEC}::fails closed for missing lifecycle or missing attempt usage")
@pytest.mark.parametrize(
    "events",
    [
        [event(0, "turn/start", {"turn": 1}), message(1, {"inputTokens": 1, "outputTokens": 1})],
        [
            event(0, "turn/start", {"turn": 1}),
            event(1, "step/start", {"turn": 1, "step": 1}),
            event(2, "step/end", {"turn": 1, "step": 1}),
            event(3, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        ],
    ],
)
def test_fails_closed_for_invalid_lifecycle(events: list[SessionEvent]) -> None:
    assert derive_turn_token_usage(events) is None
