"""Replay-aware token measurement service.

Pinned upstream route pricing depends on attachment providers that are external to the
core distribution. This port therefore keeps attachment blocks on the shared fixed
structural heuristic. Assistant anchors use the durable message rather than reassembling
provider streams; usage and all known session-event handling are preserved.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

from deepseek_cordis import Context, Service
from pydantic import ConfigDict, Field, TypeAdapter, with_config
from typing_extensions import TypedDict

from deskagents.core.session import (
    EpochHeader,
    Session,
    SessionEvent,
    SessionSeq,
    canonical_header,
    derive_event_message,
    header_equals,
    is_surface_event,
)
from deskagents.llm import Message
from deskagents.session.projection import (
    ProjectionDefinition,
    ProjectionHandle,
    SessionProjectionRegistry,
)
from deskagents.util.values import deep_freeze

from .estimate import estimate_message, estimate_tools_tokens, usage_tokens
from .projections import (
    ContextPressureState,
    context_breakdown_projection_definition,
    context_pressure_projection_definition,
    token_usage_projection_definition,
)
from .surface import MeterSurfaceNode, place_surface_node
from .types import ContextPressureProjection, TokenMeasurement, TokenSurfaceNode

Count = Annotated[int, Field(strict=True, ge=0)]


@with_config(ConfigDict(extra="forbid"))
class _Step(TypedDict):
    turn: int | float
    step: int | float


@with_config(ConfigDict(extra="forbid"))
class _Anchor(TypedDict):
    headerSeq: Count | None
    assistantTokens: int | float
    usage: dict[str, Any] | None


@with_config(ConfigDict(extra="forbid"))
class MeasurementState(TypedDict):
    headerSeq: Count | None
    stepStart: _Step | None
    anchor: _Anchor | None
    invalidSeq: Count | None


MEASUREMENT_STATE_SCHEMA: TypeAdapter[MeasurementState] = TypeAdapter(MeasurementState)


def _measurement_init(_header: object, _inherited: int) -> MeasurementState:
    return {"headerSeq": None, "stepStart": None, "anchor": None, "invalidSeq": None}


def advance_measurement(state: MeasurementState, event: SessionEvent) -> MeasurementState:
    """Advance fixed-size recovery facts, raising on invalid step boundaries."""
    kind = event["type"]
    data = event["data"]
    if kind == "request/header":
        canonical_header(data["header"])
        return {**state, "headerSeq": event["seq"]}
    if kind == "step/start":
        open_step = state["stepStart"]
        if open_step is not None:
            raise ValueError(
                f"token meter: step/start at seq {event['seq']} arrived before turn "
                f"{open_step['turn']}/step {open_step['step']} ended"
            )
        return {**state, "stepStart": {"turn": data["turn"], "step": data["step"]}}
    if kind == "step/end":
        open_step = state["stepStart"]
        if (
            open_step is None
            or open_step["turn"] != data["turn"]
            or open_step["step"] != data["step"]
        ):
            raise ValueError(
                f"token meter: step/end at seq {event['seq']} has no matching step/start event"
            )
        return {**state, "stepStart": None}
    if kind != "assistant/message":
        return state
    open_step = state["stepStart"]
    if open_step is None or open_step["turn"] != data["turn"] or open_step["step"] != data["step"]:
        raise ValueError(
            f"token meter: assistant/message at seq {event['seq']} has no matching step/start event"
        )
    message = derive_event_message(event)
    assistant_tokens = 0 if message is None else estimate_message(message)
    raw_usage = data.get("usage")
    usage = (
        dict(raw_usage)
        if isinstance(raw_usage, Mapping) and state["headerSeq"] is not None
        else None
    )
    return {
        **state,
        "anchor": {
            "headerSeq": state["headerSeq"],
            "assistantTokens": assistant_tokens,
            "usage": usage,
        },
    }


def _measurement_apply(state: MeasurementState, event: SessionEvent) -> MeasurementState:
    if state["invalidSeq"] is not None:
        return state
    try:
        return advance_measurement(state, event)
    except (KeyError, TypeError, ValueError):
        return {**state, "invalidSeq": event["seq"]}


measurement_projection_definition = ProjectionDefinition[MeasurementState](
    key="tokenMeasurement",
    state_version=2,
    state_schema=MEASUREMENT_STATE_SCHEMA,
    init=_measurement_init,
    apply=_measurement_apply,
)


@dataclass
class _ReplayState:
    consumed_events: int = 0
    surface: list[MeterSurfaceNode] = field(default_factory=list)
    header: tuple[SessionSeq, EpochHeader] | None = None
    anchor_header: tuple[SessionSeq, EpochHeader] | None = None
    anchor_nodes: tuple[MeterSurfaceNode, ...] = ()
    measurement: MeasurementState = field(default_factory=lambda: _measurement_init({}, 0))


class TokenMeter(Service):
    """One fixed estimator with isolated replay state per Session."""

    name = "tokenMeter"

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, config)
        if config is not None:
            if not isinstance(config, Mapping):
                raise TypeError("TokenMeterConfig must be a mapping")
            for key in config:
                raise ValueError(
                    f'TokenMeterConfig: unknown key "{key}" (no settings are supported)'
                )
        projections = ctx.get("sessionProjections")
        if not isinstance(projections, SessionProjectionRegistry):
            raise ValueError("TokenMeter requires sessionProjections")
        self._states: weakref.WeakKeyDictionary[Session, _ReplayState] = weakref.WeakKeyDictionary()
        self._measurement = projections.register(measurement_projection_definition, ctx=ctx)
        self._usage = projections.register(token_usage_projection_definition, ctx=ctx)
        self._pressure = cast(
            ProjectionHandle[ContextPressureState],
            projections.register(context_pressure_projection_definition, ctx=ctx),
        )
        self._breakdown = projections.register(context_breakdown_projection_definition, ctx=ctx)

    def estimate_message(self, message: Message) -> int:
        return estimate_message(message)

    def context_pressure(self, session: Session) -> ContextPressureProjection:
        """Return provider prompt pressure and newest known context capacity."""
        state: ContextPressureState = self._pressure.state_of(session)
        result: ContextPressureProjection = {}
        if "pressureTokens" in state:
            result["pressureTokens"] = state["pressureTokens"]
        if "contextWindow" in state:
            result["contextWindow"] = state["contextWindow"]
        return result

    @staticmethod
    def _header_at(state: _ReplayState, seq: int | None) -> EpochHeader | None:
        if seq is None:
            return None
        if state.header is not None and state.header[0] == seq:
            return state.header[1]
        if state.anchor_header is not None and state.anchor_header[0] == seq:
            return state.anchor_header[1]
        return None

    def _sync(self, session: Session) -> _ReplayState:
        state = self._states.get(session)
        if state is None:
            state = _ReplayState()
            self._states[session] = state
        while state.consumed_events < session.seq:
            event = session.event_at(state.consumed_events)
            assert event is not None
            next_measurement = advance_measurement(state.measurement, event)
            next_surface = (
                place_surface_node(state.surface, event)
                if is_surface_event(event)
                else state.surface
            )
            if event["type"] == "request/header":
                state.header = (event["seq"], canonical_header(event["data"]["header"]))
            if event["type"] == "assistant/message":
                state.anchor_nodes = tuple(state.surface)
                state.anchor_header = state.header
            state.measurement = next_measurement
            state.surface = next_surface
            state.consumed_events += 1
        return state

    def measure(
        self, session: Session, request_header: EpochHeader | None = None
    ) -> TokenMeasurement:
        """Measure the current durable surface using fixed heuristic attachment pricing."""
        state = self._sync(session)
        measurement = state.measurement
        header = (
            self._header_at(state, measurement["headerSeq"])
            if request_header is None
            else canonical_header(request_header)
        )
        public_nodes: list[TokenSurfaceNode] = [
            {
                "seq": node["seq"],
                "tokens": node["heuristicTokens"],
                "heuristicTokens": node["heuristicTokens"],
            }
            for node in state.surface
        ]
        surface_tokens = sum(node["tokens"] for node in public_nodes)
        anchor = measurement["anchor"]
        if anchor is not None and self._optional_header_equals(
            self._header_at(state, anchor["headerSeq"]), header
        ):
            anchor_surface = sum(node["heuristicTokens"] for node in state.anchor_nodes) + int(
                anchor["assistantTokens"]
            )
            estimated = estimate_tools_tokens(header) + anchor_surface
            usage = anchor["usage"]
            if usage is not None and usage_tokens(usage) >= estimated:
                baseline: dict[str, Any] = {
                    "kind": "usage",
                    "tokens": usage_tokens(usage),
                    "usage": usage,
                }
            else:
                baseline = {"kind": "estimated", "tokens": estimated}
            delta = surface_tokens - anchor_surface
        elif header is None and surface_tokens == 0:
            baseline = {"kind": "none", "tokens": 0}
            delta = 0
        else:
            baseline = {
                "kind": "estimated",
                "tokens": estimate_tools_tokens(header) + surface_tokens,
            }
            delta = 0
        result: TokenMeasurement = {
            "logRevision": state.consumed_events,
            "baseline": cast(Any, baseline),
            "surfaceDeltaTokens": delta,
            "totalTokens": max(0, baseline["tokens"] + delta),
            "surfaceTokens": surface_tokens,
            "nodes": public_nodes,
        }
        return deep_freeze(result)

    @staticmethod
    def _optional_header_equals(left: EpochHeader | None, right: EpochHeader | None) -> bool:
        if left is None or right is None:
            return left is right
        return header_equals(left, right)
