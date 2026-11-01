"""Pure surface pricing folds used by projections and ``TokenMeter.measure``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from deskagents.core.session import SessionEvent, SessionSeq, derive_event_message, is_surface_event

from .estimate import estimate_message


class MeterSurfaceNode(TypedDict):
    seq: SessionSeq
    heuristicTokens: int


class ShadowPriceClaim(TypedDict):
    start: SessionSeq
    end: SessionSeq
    tokens: int


class SurfaceTokensFold(TypedDict):
    deltaTokens: int
    claim: ShadowPriceClaim | None


def fold_surface_projection(
    claim: ShadowPriceClaim | None, event: SessionEvent
) -> SurfaceTokensFold:
    """Fold the bounded shadow-price protocol for one event."""
    if event["type"] in ("compaction/summary", "compaction/prune"):
        data = event["data"]
        shadowed = data["shadowedRange"]
        return {
            "deltaTokens": 0,
            "claim": {
                "start": SessionSeq(shadowed["start"]),
                "end": SessionSeq(shadowed["end"]),
                "tokens": data["shadowedTokenCount"],
            },
        }
    if not is_surface_event(event):
        return {"deltaTokens": 0, "claim": None}
    message = derive_event_message(event)
    tokens = 0 if message is None else estimate_message(message)
    op = event["surfaceOp"]
    if op == "append":
        return {"deltaTokens": tokens, "claim": None}
    if claim is None:
        return {"deltaTokens": 0, "claim": None}
    if claim["start"] != op["startSeq"] or claim["end"] != op["endSeq"]:
        raise ValueError(
            f"token surface: replace at seq {event['seq']} over range "
            f"{op['startSeq']}-{op['endSeq']} has no adjacent shadow price "
            f"(armed claim covers {claim['start']}-{claim['end']})"
        )
    return {"deltaTokens": tokens - claim["tokens"], "claim": None}


def analyze_surface_node(event: SessionEvent) -> MeterSurfaceNode:
    """Build the fixed-price node represented by one surface event."""
    message = derive_event_message(event)
    return {
        "seq": event["seq"],
        "heuristicTokens": 0 if message is None else estimate_message(message),
    }


def place_surface_node(
    nodes: Sequence[MeterSurfaceNode], event: SessionEvent
) -> list[MeterSurfaceNode]:
    """Validate and immutably apply one positional surface transition."""
    node = analyze_surface_node(event)
    op: Any = event["surfaceOp"]
    if op == "append":
        return [*nodes, node]
    assert isinstance(op, Mapping)
    start = next((index for index, item in enumerate(nodes) if item["seq"] == op["startSeq"]), -1)
    end = next((index for index, item in enumerate(nodes) if item["seq"] == op["endSeq"]), -1)
    if start < 0 or end < 0 or start > end:
        raise ValueError(
            f"token surface: replace at seq {event['seq']} has invalid current range "
            f"{op['startSeq']}-{op['endSeq']}"
        )
    return [*nodes[:start], node, *nodes[end + 1 :]]
