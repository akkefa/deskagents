from __future__ import annotations

import pytest
from deepseek_cordis import Context

from deskagents.core.session import Session
from deskagents.llm import create_assistant_message, create_user_message
from deskagents.llm.token_meter import TokenMeter
from deskagents.session.projection import SessionProjectionRegistry

SPEC = "packages/llm/token-meter/tests/token-meter.spec.ts"


def meter() -> TokenMeter:
    ctx = Context()
    SessionProjectionRegistry(ctx)
    return TokenMeter(ctx, {})


@pytest.mark.upstream(f"{SPEC}::returns a detached deeply immutable empty measurement")
def test_empty_measurement() -> None:
    result = meter().measure(Session.create("empty"))
    assert result == {
        "logRevision": 0,
        "baseline": {"kind": "none", "tokens": 0},
        "surfaceDeltaTokens": 0,
        "totalTokens": 0,
        "surfaceTokens": 0,
        "nodes": [],
    }


@pytest.mark.upstream(f"{SPEC}::keeps an earlier unified snapshot detached from later replay")
def test_measure_replays_surface_and_detaches_results() -> None:
    service = meter()
    session = Session.create("detached")
    session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "first"}], source={"kind": "user"}),
        surface_op="append",
    )
    before = service.measure(session)
    session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "second"}], source={"kind": "user"}),
        surface_op="append",
    )
    after = service.measure(session)
    assert before["logRevision"] == 1
    assert len(before["nodes"]) == 1
    assert after["logRevision"] == 2
    assert len(after["nodes"]) == 2
    assert before["nodes"][0]["tokens"] == before["nodes"][0]["heuristicTokens"]


@pytest.mark.upstream(f"{SPEC}::uses disjoint provider usage and signed durable-output rewrites")
def test_measure_anchors_to_provider_usage() -> None:
    service = meter()
    session = Session.create("usage-anchor")
    session.append(
        "user/message",
        create_user_message(
            content=[{"type": "text", "text": "question"}], source={"kind": "user"}
        ),
        surface_op="append",
    )
    session.append("step/start", {"turn": 1, "step": 1})
    session.append(
        "request/header",
        {
            "reason": "initial",
            "header": {"config": {"provider": "mock", "model": "model"}},
        },
    )
    usage = {
        "inputTokens": 100,
        "outputTokens": 20,
        "cacheReadTokens": 5,
        "cacheWriteTokens": 2,
    }
    session.append(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "stream": [],
            "usage": usage,
            "message": create_assistant_message(
                content=[{"type": "text", "text": "answer"}],
                source={"provider": "mock", "model": "model"},
            ),
        },
        surface_op="append",
    )
    session.append("step/end", {"turn": 1, "step": 1})
    session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "later"}], source={"kind": "user"}),
        surface_op="append",
    )
    result = service.measure(session)
    assert result["baseline"] == {"kind": "usage", "tokens": 127, "usage": usage}
    assert result["surfaceDeltaTokens"] > 0
    assert result["totalTokens"] == 127 + result["surfaceDeltaTokens"]


@pytest.mark.upstream(f"{SPEC}::rejects an assistant without its step boundary transactionally")
def test_measure_repeats_boundary_failure_without_advancing() -> None:
    service = meter()
    session = Session.create("bad-step")
    session.append(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "stream": [],
            "message": create_assistant_message(
                content=[{"type": "text", "text": "bad"}],
                source={"provider": "mock", "model": "model"},
            ),
        },
        surface_op="append",
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="no matching step/start"):
            service.measure(session)
