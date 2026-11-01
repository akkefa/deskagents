"""Exact, fail-closed token accounting for one complete durable turn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from deskagents.core.session import SessionEvent
from deskagents.llm import TokenUsage
from deskagents.util.values import is_safe_integer

from .projections import usage_of
from .types import TurnTokenUsage, TurnTokenUsageRoute


def _count(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and is_safe_integer(value)
        and value >= 0
    )


def _safe_sum(values: Sequence[int]) -> int | None:
    total = sum(values)
    return total if is_safe_integer(total) else None


def _route(message: Mapping[str, Any]) -> TurnTokenUsageRoute | None:
    source = message.get("source")
    if not isinstance(source, Mapping):
        return None
    provider, model = source.get("provider"), source.get("model")
    if not isinstance(provider, str) or not provider or not isinstance(model, str) or not model:
        return None
    return {"provider": provider, "model": model}


def _normalize(
    usage: TokenUsage, route: TurnTokenUsageRoute | None = None
) -> dict[str, Any] | None:
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    if not _count(input_tokens) or not _count(output_tokens):
        return None
    assert isinstance(input_tokens, int) and isinstance(output_tokens, int)
    optional = ("cacheReadTokens", "cacheWriteTokens", "reasoningTokens")
    if any(name in usage and not _count(usage[name]) for name in optional):
        return None
    reasoning = usage.get("reasoningTokens")
    if isinstance(reasoning, int) and reasoning > output_tokens:
        return None
    read = usage.get("cacheReadTokens")
    write = usage.get("cacheWriteTokens")
    known = _safe_sum(
        [input_tokens]
        + ([] if read is None else [cast(int, read)])
        + ([] if write is None else [cast(int, write)])
    )
    if known is None:
        return None
    total = usage.get("totalTokens")
    if total is not None:
        if not _count(total):
            return None
        assert isinstance(total, int)
        exact_prompt = total - output_tokens
        if not _count(exact_prompt) or exact_prompt < known:
            return None
        if read is not None and write is not None and exact_prompt != known:
            return None
        exact_total = total
    else:
        if read is None or write is None:
            return None
        derived = _safe_sum([known, output_tokens])
        if derived is None:
            return None
        exact_total = derived
    result: dict[str, Any] = {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": exact_total,
    }
    for name in optional:
        if name in usage:
            result[name] = usage[name]
    if route is not None:
        result["route"] = route
    return result


def _aggregate(attempts: Sequence[dict[str, Any]]) -> TurnTokenUsage | None:
    if not attempts:
        return None
    inputs = _safe_sum([item["inputTokens"] for item in attempts])
    outputs = _safe_sum([item["outputTokens"] for item in attempts])
    totals = _safe_sum([item["totalTokens"] for item in attempts])
    if inputs is None or outputs is None or totals is None:
        return None
    result: TurnTokenUsage = {
        "uncachedInputTokens": inputs,
        "outputTokens": outputs,
        "totalTokens": totals,
    }
    for name in ("cacheReadTokens", "cacheWriteTokens", "reasoningTokens"):
        values = [item.get(name) for item in attempts]
        if all(_count(value) for value in values):
            summed = _safe_sum(cast(list[int], values))
            if summed is not None:
                result[name] = summed
    routes = [item.get("route") for item in attempts]
    if all(route is not None for route in routes):
        unique: dict[tuple[str, str], TurnTokenUsageRoute] = {}
        for route in cast(list[TurnTokenUsageRoute], routes):
            unique[(route["provider"], route["model"])] = route
        result["routes"] = list(unique.values())
    return result


def derive_turn_token_usage(events: Sequence[SessionEvent]) -> TurnTokenUsage | None:
    """Derive exact usage only when the complete attempt lifecycle proves it."""
    state: dict[str, Any] = {"kind": "idle"}
    attempts: list[dict[str, Any]] = []
    turn: int | None = None
    saw_end = False

    def same(data: Mapping[str, Any]) -> bool:
        return state.get("turn") == data.get("turn") and state.get("step") == data.get("step")

    def close(route: TurnTokenUsageRoute | None = None) -> bool:
        if state["kind"] != "open" or state.get("sample") is None:
            return False
        normalized = _normalize(state["sample"], route)
        if normalized is None:
            return False
        attempts.append(normalized)
        return True

    for event in events:
        kind, data = event["type"], event["data"]
        if kind == "turn/start":
            if turn is not None or state["kind"] != "idle":
                return None
            turn = data["turn"]
            continue
        if turn is None:
            return None
        if kind == "turn/end":
            if data["turn"] != turn or state["kind"] != "idle" or saw_end:
                return None
            saw_end = True
            continue
        if saw_end:
            return None
        if kind == "step/start":
            if data["turn"] != turn or state["kind"] != "idle":
                return None
            state = {"kind": "open", "turn": turn, "step": data["step"]}
        elif kind == "llm/retry-started":
            if state["kind"] != "settled" or state.get("by") != "retry" or not same(data):
                return None
            state = {"kind": "open", "turn": turn, "step": data["step"]}
        elif kind == "assistant/attempt":
            if state["kind"] != "open" or data["turn"] != turn or not same(data):
                return None
            sample = usage_of(event) or state.get("sample")
            if sample is not None:
                state["sample"] = sample
            if not close():
                return None
            state = {"kind": "finishClosed", "turn": turn, "step": data["step"]}
        elif kind == "assistant/message":
            if state["kind"] != "open" or data["turn"] != turn or not same(data):
                return None
            sample = usage_of(event)
            if sample is not None:
                state["sample"] = sample
            if not close(_route(data["message"])):
                return None
            state = {
                "kind": "settled",
                "turn": turn,
                "step": data["step"],
                "by": "message",
            }
        elif kind == "llm/retry":
            if state["kind"] == "idle" or data["turn"] != turn or not same(data):
                return None
            if state["kind"] == "settled" or (state["kind"] == "open" and not close()):
                return None
            state = {
                "kind": "settled",
                "turn": turn,
                "step": data["step"],
                "by": "retry",
            }
        elif kind == "step/end":
            if state["kind"] == "idle" or data["turn"] != turn or not same(data):
                return None
            if state["kind"] == "open" and not close():
                return None
            state = {"kind": "idle"}
    if not saw_end or state["kind"] != "idle":
        return None
    return _aggregate(attempts)
