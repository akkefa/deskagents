"""Pending tool-result recovery shared by failed live steps, interrupted logs and fork seeds.

Upstream: packages/core/session/src/repair.ts
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from deskagents.util.values import deep_freeze

from .types import SessionEvent, SessionSeq

TOOL_NOT_STARTED = "TOOL_NOT_STARTED"
"""Recovery code for an assistant tool request that never reached a recorded call start."""

TOOL_OUTCOME_UNKNOWN = "TOOL_OUTCOME_UNKNOWN"
"""Recovery code for a recorded tool call whose completed outcome was not durably recorded."""

OpenTurnCloseKind = Literal["interrupted", "forked"]

_CLOSER_TEXT: dict[str, dict[str, str]] = {
    "interrupted": {
        "started": (
            "The tool call was interrupted after it was recorded, but no result was durably "
            "recorded. Its outcome is unknown. Decide whether to retry from the tool semantics: "
            "retry only if the operation is read-only or idempotent; if it may have side effects, "
            "first verify external state or ask the user. Do not retry blindly."
        ),
        "notStarted": (
            "The tool call was interrupted before the Harness recorded it as started. "
            "Retry it if it is still needed."
        ),
    },
    "forked": {
        "started": (
            "The history inherited by this branch records this tool call starting but does not "
            "include its result. The parent session may have completed it after the fork point. "
            "Decide whether to retry from the tool semantics: retry only if the operation is "
            "read-only or idempotent; if it may have side effects, first verify external state "
            "or ask the user. Do not retry blindly."
        ),
        "notStarted": (
            "The history inherited by this branch has no record of this tool call starting. "
            "The parent session may have executed it after the fork point. Decide whether to "
            "retry from the tool semantics: retry only if the operation is read-only or "
            "idempotent; if it may have side effects, first verify external state or ask the "
            "user. Do not retry blindly."
        ),
    },
}


def open_turn_closers(
    events: Sequence[SessionEvent], cause: OpenTurnCloseKind
) -> list[SessionEvent]:
    """Deterministic synthetic events that close an open tail turn.

    Unmatched calls receive error results first, then an open ``step/end`` and a
    cause-specific ``turn/end``. Sequences continue the log; timestamps reuse the last
    real event. A balanced or empty log returns no events.
    """
    open_turn: int | None = None
    open_step: int | None = None
    recovery = ToolCallRecovery(cause)
    for event in events:
        recovery.observe(event)
        kind, data = event["type"], event["data"]
        if kind == "turn/start":
            open_turn, open_step = data["turn"], None
        elif kind == "turn/end":
            open_turn, open_step = None, None
        elif kind == "step/start":
            open_step = data["step"]
        elif kind == "step/end":
            open_step = None
    if open_turn is None or not events:
        return []
    last = events[-1]
    closers: list[SessionEvent] = list(recovery.results())
    seq = last["seq"] + len(closers) + 1
    time = last["time"]
    if open_step is not None:
        closers.append(
            deep_freeze(
                {
                    "type": "step/end",
                    "seq": seq,
                    "time": time,
                    "data": {"turn": open_turn, "step": open_step},
                }
            )
        )
        seq += 1
    closers.append(
        deep_freeze(
            {
                "type": "turn/end",
                "seq": seq,
                "time": time,
                "data": {"turn": open_turn, "reason": {"kind": cause}},
            }
        )
    )
    return closers


@dataclass
class _Pending:
    turn: int
    step: int
    call_seq: SessionSeq | None = None


class ToolCallRecovery:
    """Track unanswered assistant tool requests from one Session's committed events.

    Observe from the start of the owned step or replay prefix, and recover before its
    step closes. Retains pending identities, not event history.
    """

    def __init__(self, cause: OpenTurnCloseKind = "interrupted") -> None:
        self._cause = cause
        self._pending: dict[str, _Pending] = {}
        self._last: tuple[int, int] | None = None

    def observe(self, event: SessionEvent) -> None:
        """Consume the next event; closed steps and turn boundaries discard pending requests."""
        self._last = (event["seq"], event["time"])
        kind, data = event["type"], event["data"]
        if kind in ("turn/start", "turn/end", "step/end"):
            self._pending.clear()
        elif kind == "assistant/message":
            for block in data["message"]["content"]:
                if block["type"] == "tool-call":
                    self._pending[block["id"]] = _Pending(data["turn"], data["step"])
        elif kind == "tool/call":
            entry = self._pending.get(data["callId"])
            if entry is not None:
                entry.call_seq = event["seq"]
        elif kind == "tool/result":
            call_id = data["message"]["source"]["callId"]
            entry = self._pending.get(call_id)
            if (
                event.get("surfaceOp") == "append"
                and entry is not None
                and entry.turn == data["turn"]
                and entry.step == data["step"]
            ):
                del self._pending[call_id]

    def results(self) -> list[SessionEvent]:
        """Conservative error results in assistant order, without changing tracked state."""
        if self._last is None:
            return []
        seq = self._last[0] + 1
        time = self._last[1]
        text = _CLOSER_TEXT[self._cause]
        results: list[SessionEvent] = []
        for call_id, pending in self._pending.items():
            started = pending.call_seq is not None
            message = {
                "id": f"{self._cause}-tool-result-{call_id}-{seq}",
                "role": "tool",
                "toolCallId": call_id,
                "isError": True,
                "source": {"kind": "tool", "callId": call_id},
                "content": [
                    {"type": "text", "text": text["started"] if started else text["notStarted"]}
                ],
            }
            event: dict[str, object] = {
                "type": "tool/result",
                "seq": seq,
                "time": time,
                "data": {
                    "turn": pending.turn,
                    "step": pending.step,
                    "message": message,
                    "error": (
                        {"name": "ToolOutcomeUnknownError", "code": TOOL_OUTCOME_UNKNOWN}
                        if started
                        else {"name": "ToolNotStartedError", "code": TOOL_NOT_STARTED}
                    ),
                },
                "surfaceOp": "append",
            }
            if started:
                event["sourceEventSeqs"] = [pending.call_seq]
            results.append(deep_freeze(event))
            seq += 1
        return results


def interrupted_turn_closers(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    """Crash recovery: synthetic ``interrupted`` closers that balance a persisted log."""
    return open_turn_closers(events, "interrupted")
