"""Session identity, positions, header and event vocabulary.

Upstream: packages/core/session/src/types.ts

Events and payloads are frozen JSON mappings. Each ``SessionEvent`` has ``type``,
``seq``, ``time`` and ``data``; message-producing events also carry ``surfaceOp``
and optionally ``sourceEventSeqs``; an unknown-but-skippable record carries
``ignorable: true``. Payload shapes per event type are documented on
:data:`SESSION_EVENT_DATA` and follow upstream's ``SessionEventMap`` field for field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NewType

from deskagents.util.values import is_safe_integer

__all__ = [
    "SESSION_EVENT_DATA",
    "SESSION_FORMAT_VERSION",
    "SURFACE_EVENT_TYPES",
    "CreateSessionOptions",
    "EpochHeader",
    "PluginRecord",
    "RequestContext",
    "RestoredSessionOptions",
    "SessionEvent",
    "SessionHeader",
    "SessionId",
    "SessionLogOffset",
    "SessionSeedEventState",
    "SessionSeq",
    "SurfaceOp",
    "session_id",
    "session_log_offset",
    "session_seq",
]

SessionId = NewType("SessionId", str)
"""Identifies one session in the store and its persistence artifacts."""

SessionSeq = NewType("SessionSeq", int)
"""Sequence number of one existing event in a Session log."""

SessionLogOffset = NewType("SessionLogOffset", int)
"""A log gap, prefix length or read offset, which may equal the event count."""


def session_id(value: str) -> SessionId:
    """Brand a string as a :data:`SessionId` (upstream ``SessionId(id)``)."""
    return SessionId(value)


def _non_negative(value: object, name: str) -> int:
    negative_zero = isinstance(value, float) and value == 0 and str(value).startswith("-")
    if not is_safe_integer(value) or negative_zero or value < 0:  # type: ignore[operator]
        raise TypeError(f"{name} must be a non-negative safe integer, got {value!r}")
    return int(value)  # type: ignore[call-overload,no-any-return]


def session_seq(value: object) -> SessionSeq:
    """Admit a value as an existing Session event position (upstream ``SessionSeq(n)``)."""
    return SessionSeq(_non_negative(value, "SessionSeq"))


def session_log_offset(value: object) -> SessionLogOffset:
    """Admit a value as a Session log offset (upstream ``SessionLogOffset(n)``)."""
    return SessionLogOffset(_non_negative(value, "SessionLogOffset"))


SESSION_FORMAT_VERSION = 4
"""Current logical Session format version, stamped into every new header."""

SessionHeader = Mapping[str, Any]
"""Immutable validated storage metadata kept outside the event log: ``version``, ``id``,
``createdAt``, ``isSeeded``, and ``cwd``, ``parentSession``, ``origin`` (``"subagent"``),
``delegationDepth``, ``agentPreset`` when present."""

SessionEvent = Mapping[str, Any]
"""One immutable log entry: ``type``, ``seq``, ``time``, ``data`` and, when applicable,
``sourceEventSeqs``, ``surfaceOp``, ``ignorable``."""

EpochHeader = Mapping[str, Any]
"""Logged request state outside derived history: ``config``, ``adapterDefaults``?, ``tools``?."""

RequestContext = Mapping[str, Any]
"""Route metadata: ``provider``, ``model``, ``contextWindow``?, ``systemPromptUpdate``?."""

SurfaceOp = Any
"""``"append"`` or ``{"op": "replace", "startSeq": n, "endSeq": m}``."""

SessionSeedEventState = Literal["detached", "shared-frozen"]
"""Aliasing state of an adoptable seed; ``shared-frozen`` permits deeply frozen aliases."""

SURFACE_EVENT_TYPES: frozenset[str] = frozenset(
    {"system/message", "developer/message", "user/message", "assistant/message", "tool/result"}
)
"""Event types whose events produce LLM messages and may join the ordered surface."""

SESSION_EVENT_DATA: Mapping[str, str] = {
    "turn/start": "{turn}",
    "turn/end": "{turn, reason: TurnEndReason}",
    "step/start": "{turn, step}",
    "step/end": "{turn, step}",
    "user/message": "UserMessage",
    "developer/message": "{turn, step, message: DeveloperMessage, headerSeq?}",
    "system/message": "{turn, step, message: SystemMessage}",
    "assistant/message": "{turn, step, message, stream, usage?, interrupted?: true}",
    "assistant/attempt": "{turn, step, stream}",
    "tool/call": "{turn, step, callId, name, arguments}",
    "tool/result": "{turn, step, message: ToolResultMessage, error?, meta?}",
    "request/header": "{header: EpochHeader, reason, startsSeries?: true}",
    "request/context": "RequestContext",
    "session/end-seed": "{inherited?: true}",
}
"""Payload shape of each core event type, as in upstream ``SessionEventMap``.

``TurnEndReason.kind`` is one of ``completed``, ``aborted`` (with ``reason``), ``blocked``,
``error`` (with ``error: LlmFailure``), ``max-tokens``, ``interrupted`` or ``forked``.
``request/header.reason`` is ``initial``, ``resume``, ``change`` or ``series``."""


@dataclass(frozen=True)
class CreateSessionMeta:
    """Caller-supplied storage fields folded into a :data:`SessionHeader`."""

    cwd: str | None = None
    parent_session: SessionId | str | None = None
    created_at: int | None = None
    is_seeded: bool | None = None
    origin: Literal["subagent"] | None = None
    delegation_depth: int | None = None
    agent_preset: str | None = None


@dataclass(frozen=True)
class CreateSessionOptions:
    """Options for creating a Session through the store.

    ``seed`` replays or forks an existing log; ``inherited_event_count`` is the exact
    fork-inherited prefix length when ``meta.is_seeded`` is true.
    """

    seed: Sequence[SessionEvent] | None = None
    inherited_event_count: SessionLogOffset | None = None
    meta: CreateSessionMeta | None = None


@dataclass(frozen=True)
class RestoredSessionOptions:
    """Adoptable storage values transferred to ``SessionStore.prepare`` from persistence."""

    seed: Sequence[SessionEvent]
    meta: SessionHeader
    inherited_event_count: SessionLogOffset
    event_state: SessionSeedEventState


@dataclass(frozen=True)
class PluginRecord:
    """One committed experimental plugin record, read back with ``plugin_record_of``."""

    type: str
    seq: SessionSeq
    time: int
    data: Any = field(compare=True)
