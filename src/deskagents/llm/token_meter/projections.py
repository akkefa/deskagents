"""Pinned token usage, context pressure, and context breakdown projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NotRequired, cast

from pydantic import ConfigDict, Field, TypeAdapter, with_config
from typing_extensions import TypedDict

from deskagents.core.session import (
    SessionEvent,
    canonical_header,
    derive_event_message,
    is_surface_event,
)
from deskagents.llm import TokenUsage
from deskagents.session.projection import ProjectionDefinition, ProjectionWire

from .estimate import estimate_message, estimate_tools_tokens
from .surface import ShadowPriceClaim, fold_surface_projection
from .types import (
    ContextBreakdownProjection,
    ContextPressureProjection,
    TokenUsageProjection,
)

Count = Annotated[int, Field(strict=True, ge=0)]
PositiveCount = Annotated[int, Field(strict=True, gt=0)]


@with_config(ConfigDict(extra="forbid"))
class _UsageBuckets(TypedDict):
    uncachedInputTokens: Count
    outputTokens: Count
    cacheReadTokens: Count
    cacheWriteTokens: Count


@with_config(ConfigDict(extra="forbid"))
class _LastUsage(TypedDict):
    turn: Count
    step: Count
    buckets: _UsageBuckets


@with_config(ConfigDict(extra="forbid"))
class TokenUsageState(TypedDict):
    totals: _UsageBuckets
    last: _LastUsage | None


@with_config(ConfigDict(extra="forbid"))
class _PressureClaim(TypedDict):
    start: Count
    end: Count
    tokens: Count


@with_config(ConfigDict(extra="forbid"))
class ContextPressureState(TypedDict):
    surfaceTokens: Count
    contextWindow: NotRequired[PositiveCount]
    pressureTokens: NotRequired[Count]
    sampledSurfaceTokens: NotRequired[Count]
    claim: NotRequired[_PressureClaim]


@with_config(ConfigDict(extra="forbid"))
class _SystemNode(TypedDict):
    seq: Count
    tokens: Count


@with_config(ConfigDict(extra="forbid"))
class _BreakdownClaim(TypedDict):
    start: Count
    end: Count
    tokens: Count
    systemSeqs: list[Count]


@with_config(ConfigDict(extra="forbid"))
class ContextBreakdownState(TypedDict):
    systems: list[_SystemNode]
    surfaceTokens: Count
    breakdown: ContextBreakdownProjection
    claim: NotRequired[_BreakdownClaim]


USAGE_VIEW_SCHEMA: TypeAdapter[TokenUsageProjection] = TypeAdapter(TokenUsageProjection)
USAGE_STATE_SCHEMA: TypeAdapter[TokenUsageState] = TypeAdapter(TokenUsageState)
PRESSURE_VIEW_SCHEMA: TypeAdapter[ContextPressureProjection] = TypeAdapter(
    ContextPressureProjection
)
PRESSURE_STATE_SCHEMA: TypeAdapter[ContextPressureState] = TypeAdapter(ContextPressureState)
BREAKDOWN_VIEW_SCHEMA: TypeAdapter[ContextBreakdownProjection] = TypeAdapter(
    ContextBreakdownProjection
)
BREAKDOWN_STATE_SCHEMA: TypeAdapter[ContextBreakdownState] = TypeAdapter(ContextBreakdownState)


def _zero_buckets() -> _UsageBuckets:
    return {
        "uncachedInputTokens": 0,
        "outputTokens": 0,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
    }


def _last_stream_usage(stream: object) -> TokenUsage | None:
    if not isinstance(stream, Sequence) or isinstance(stream, str | bytes):
        return None
    found: TokenUsage | None = None
    for entry in stream:
        if not isinstance(entry, Mapping):
            continue
        chunk = entry.get("chunk", entry)
        if isinstance(chunk, Mapping) and chunk.get("type") == "usage":
            usage = chunk.get("usage")
            if isinstance(usage, Mapping):
                found = cast(TokenUsage, usage)
    return found


def usage_of(event: SessionEvent) -> TokenUsage | None:
    """Read the last provider usage carried by an assistant settlement."""
    if event["type"] not in ("assistant/message", "assistant/attempt"):
        return None
    data = event["data"]
    direct = data.get("usage")
    if event["type"] == "assistant/message" and isinstance(direct, Mapping):
        return cast(TokenUsage, direct)
    return _last_stream_usage(data.get("stream"))


def _usage_init(_header: object, _inherited: int) -> TokenUsageState:
    return {"totals": _zero_buckets(), "last": None}


def _usage_apply(state: TokenUsageState, event: SessionEvent) -> TokenUsageState:
    if event["type"] == "llm/retry-started":
        last = state["last"]
        data = event["data"]
        if last is not None and last["turn"] == data["turn"] and last["step"] == data["step"]:
            return {**state, "last": None}
        return state
    sample = usage_of(event)
    if sample is None:
        return state
    data = event["data"]
    buckets: _UsageBuckets = {
        "uncachedInputTokens": sample["inputTokens"],
        "outputTokens": sample["outputTokens"],
        "cacheReadTokens": sample.get("cacheReadTokens", 0),
        "cacheWriteTokens": sample.get("cacheWriteTokens", 0),
    }
    last = state["last"]
    previous = (
        last["buckets"]
        if last is not None and last["turn"] == data["turn"] and last["step"] == data["step"]
        else None
    )
    if previous == buckets:
        return state
    totals = state["totals"]
    return {
        "totals": {
            "uncachedInputTokens": totals["uncachedInputTokens"]
            - (0 if previous is None else previous["uncachedInputTokens"])
            + buckets["uncachedInputTokens"],
            "outputTokens": totals["outputTokens"]
            - (0 if previous is None else previous["outputTokens"])
            + buckets["outputTokens"],
            "cacheReadTokens": totals["cacheReadTokens"]
            - (0 if previous is None else previous["cacheReadTokens"])
            + buckets["cacheReadTokens"],
            "cacheWriteTokens": totals["cacheWriteTokens"]
            - (0 if previous is None else previous["cacheWriteTokens"])
            + buckets["cacheWriteTokens"],
        },
        "last": {"turn": data["turn"], "step": data["step"], "buckets": buckets},
    }


token_usage_projection_definition = ProjectionDefinition[TokenUsageState](
    key="tokenUsage",
    state_version=2,
    state_schema=USAGE_STATE_SCHEMA,
    init=_usage_init,
    apply=_usage_apply,
    wire=ProjectionWire(
        view_schema=USAGE_VIEW_SCHEMA,
        view=lambda state: state["totals"],
    ),
)


def _pressure_init(_header: object, _inherited: int) -> ContextPressureState:
    return {"surfaceTokens": 0}


def _pressure_apply(state: ContextPressureState, event: SessionEvent) -> ContextPressureState:
    claim = cast(ShadowPriceClaim | None, state.get("claim"))
    folded = fold_surface_projection(claim, event)
    next_state: ContextPressureState = state
    if event["type"] == "request/context":
        window = event["data"].get("contextWindow")
        if window != state.get("contextWindow"):
            next_state = cast(ContextPressureState, dict(next_state))
            if window is None:
                next_state.pop("contextWindow", None)
            else:
                next_state["contextWindow"] = window
    usage = usage_of(event)
    if usage is not None:
        pressure = (
            usage["inputTokens"]
            + usage.get("cacheReadTokens", 0)
            + usage.get("cacheWriteTokens", 0)
        )
        if (
            pressure != next_state.get("pressureTokens")
            or next_state.get("sampledSurfaceTokens") != next_state["surfaceTokens"]
        ):
            next_state = cast(
                ContextPressureState,
                {
                    **next_state,
                    "pressureTokens": pressure,
                    "sampledSurfaceTokens": next_state["surfaceTokens"],
                },
            )
    if folded["deltaTokens"]:
        next_state = cast(
            ContextPressureState,
            {
                **next_state,
                "surfaceTokens": next_state["surfaceTokens"] + folded["deltaTokens"],
            },
        )
    next_claim = folded["claim"]
    if "claim" not in state and next_claim is None:
        return next_state
    next_state = cast(ContextPressureState, dict(next_state))
    next_state.pop("claim", None)
    if next_claim is not None:
        next_state["claim"] = cast(_PressureClaim, next_claim)
    return next_state


def _pressure_view(state: ContextPressureState) -> ContextPressureProjection:
    view: ContextPressureProjection = {}
    if "contextWindow" in state:
        view["contextWindow"] = state["contextWindow"]
    if "pressureTokens" in state:
        view["pressureTokens"] = state["pressureTokens"]
        sampled = state.get("sampledSurfaceTokens")
        if sampled is not None:
            view["projectedTokens"] = max(
                0, state["pressureTokens"] + state["surfaceTokens"] - sampled
            )
    return view


context_pressure_projection_definition = ProjectionDefinition[ContextPressureState](
    key="contextPressure",
    state_version=5,
    state_schema=PRESSURE_STATE_SCHEMA,
    init=_pressure_init,
    apply=_pressure_apply,
    wire=ProjectionWire(view_schema=PRESSURE_VIEW_SCHEMA, view=_pressure_view),
)


def _breakdown_init(_header: object, _inherited: int) -> ContextBreakdownState:
    return {
        "systems": [],
        "surfaceTokens": 0,
        "breakdown": {"systemTokens": 0, "toolsTokens": 0, "messageTokens": 0},
    }


def _with_totals(
    state: ContextBreakdownState, systems: list[_SystemNode], surface_tokens: int
) -> ContextBreakdownState:
    surface_tokens = max(0, surface_tokens)
    system_tokens = next((node["tokens"] for node in reversed(systems) if node["tokens"] > 0), 0)
    message_tokens = max(0, surface_tokens - system_tokens)
    old = state["breakdown"]
    breakdown = (
        old
        if old["systemTokens"] == system_tokens and old["messageTokens"] == message_tokens
        else {
            "systemTokens": system_tokens,
            "toolsTokens": old["toolsTokens"],
            "messageTokens": message_tokens,
        }
    )
    return {
        "systems": systems,
        "surfaceTokens": surface_tokens,
        "breakdown": breakdown,
    }


def _breakdown_apply(state: ContextBreakdownState, event: SessionEvent) -> ContextBreakdownState:
    if event["type"] in ("compaction/summary", "compaction/prune"):
        data = event["data"]
        shadowed = set(data["shadowedSeqs"])
        span = data["shadowedRange"]
        return {
            **state,
            "claim": {
                "start": span["start"],
                "end": span["end"],
                "tokens": data["shadowedTokenCount"],
                "systemSeqs": [item["seq"] for item in state["systems"] if item["seq"] in shadowed],
            },
        }
    claim = state.get("claim")
    unclaimed = cast(
        ContextBreakdownState,
        {key: value for key, value in state.items() if key != "claim"},
    )
    if event["type"] == "request/header":
        tools = estimate_tools_tokens(canonical_header(event["data"]["header"]))
        if tools == unclaimed["breakdown"]["toolsTokens"]:
            return unclaimed
        return {
            **unclaimed,
            "breakdown": {**unclaimed["breakdown"], "toolsTokens": tools},
        }
    if not is_surface_event(event):
        return unclaimed
    message = derive_event_message(event)
    tokens = 0 if message is None else estimate_message(message)
    op: Any = event["surfaceOp"]
    node: _SystemNode = {"seq": event["seq"], "tokens": tokens}
    is_system = event["type"] == "system/message"
    if op == "append":
        systems = [*unclaimed["systems"], node] if is_system else unclaimed["systems"]
        return _with_totals(unclaimed, systems, unclaimed["surfaceTokens"] + tokens)
    if claim is not None:
        if claim["start"] != op["startSeq"] or claim["end"] != op["endSeq"]:
            raise ValueError(
                f"context breakdown: replace at seq {event['seq']} over range "
                f"{op['startSeq']}-{op['endSeq']} has no adjacent shadow price "
                f"(armed claim covers {claim['start']}-{claim['end']})"
            )
        if is_system:
            raise ValueError(
                f"context breakdown: replace at seq {event['seq']} puts a system message "
                "in place of a metered range; its order among the surviving system prompts "
                "is unknown"
            )
        removed = set(claim["systemSeqs"])
        systems = [item for item in unclaimed["systems"] if item["seq"] not in removed]
        return _with_totals(
            unclaimed, systems, unclaimed["surfaceTokens"] + tokens - claim["tokens"]
        )
    index = (
        next(
            (idx for idx, item in enumerate(unclaimed["systems"]) if item["seq"] == op["startSeq"]),
            -1,
        )
        if op["startSeq"] == op["endSeq"]
        else -1
    )
    if index < 0:
        return unclaimed
    replaced = unclaimed["systems"][index]
    systems = list(unclaimed["systems"])
    systems[index : index + 1] = [node] if is_system else []
    return _with_totals(
        unclaimed, systems, unclaimed["surfaceTokens"] + tokens - replaced["tokens"]
    )


context_breakdown_projection_definition = ProjectionDefinition[ContextBreakdownState](
    key="contextBreakdown",
    state_version=6,
    state_schema=BREAKDOWN_STATE_SCHEMA,
    init=_breakdown_init,
    apply=_breakdown_apply,
    wire=ProjectionWire(
        view_schema=BREAKDOWN_VIEW_SCHEMA,
        view=lambda state: state["breakdown"],
    ),
)
