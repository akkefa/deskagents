"""Event-sourced Session, the in-memory store (``ctx.sessions``) and log validation.

Upstream: packages/core/session/src/index.ts
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from deepseek_cordis import Context, Service, current_context

from deskagents._cordis import contain
from deskagents.core.scope import ScopeCarrier, scope_of, scope_target
from deskagents.llm import Message, ToolHistory
from deskagents.util.values import (
    FrozenJsonObject,
    deep_freeze,
    is_json_value,
    is_safe_integer,
    snapshot_json_value,
)

from .fork import build_fork_seed
from .request_header import fold_request_header
from .surface import (
    SessionMessageProjection,
    SurfaceManager,
    validate_session_event_data,
    validate_surface_metadata,
)
from .tool_history import ToolHistoryProjection
from .types import (
    SESSION_FORMAT_VERSION,
    SURFACE_EVENT_TYPES,
    CreateSessionMeta,
    CreateSessionOptions,
    EpochHeader,
    PluginRecord,
    RequestContext,
    RestoredSessionOptions,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
    SessionSeedEventState,
    SessionSeq,
    session_log_offset,
)

logger = logging.getLogger(__name__)

_SURFACE_ROLES = {
    "system/message": "system",
    "developer/message": "developer",
    "user/message": "user",
    "assistant/message": "assistant",
    "tool/result": "tool",
}
_ENVELOPE_KEYS = frozenset(
    {"type", "seq", "time", "data", "surfaceOp", "sourceEventSeqs", "ignorable"}
)
_ADAPTER_DEFAULT_KEYS = frozenset({"reasoningEffort", "maxTokens"})
_PLUGIN_RECORD_TYPE = re.compile(r"plugin:[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*")
_PLAIN_RECORDS = (dict, FrozenJsonObject)


def _now() -> int:
    """``Date.now()``: Unix epoch milliseconds."""
    return time.time_ns() // 1_000_000


def _is_count(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and is_safe_integer(value)
        and value >= 0
    )


# --- header validation -------------------------------------------------------------


def _validate_session_header(session_id: SessionId, value: object) -> SessionHeader:
    if not isinstance(value, Mapping):
        raise ValueError("session header is not a plain JSON record")
    record = value
    if "seedLength" in record:
        raise ValueError('session header has invalid field "seedLength"')
    if record.get("version") != SESSION_FORMAT_VERSION or isinstance(record.get("version"), bool):
        raise ValueError(
            f"session header version must be {SESSION_FORMAT_VERSION}, got {record.get('version')}"
        )
    if record.get("id") != session_id:
        raise ValueError(
            f'session header id "{record.get("id")}" does not match session id "{session_id}"'
        )
    if not _is_count(record.get("createdAt")):
        raise ValueError("session header createdAt must be a non-negative safe integer")
    if "cwd" in record:
        cwd = record["cwd"]
        if not isinstance(cwd, str):
            raise ValueError("session header cwd must be a string")
        if not os.path.isabs(cwd):
            raise ValueError(f'session header cwd must be an absolute path, got "{cwd}"')
    if "parentSession" in record and not isinstance(record["parentSession"], str):
        raise ValueError("session header parentSession must be a string")
    if not isinstance(record.get("isSeeded"), bool):
        raise ValueError("session header isSeeded must be a boolean")
    if "origin" in record and record["origin"] != "subagent":
        raise ValueError('session header origin must be "subagent"')
    if "delegationDepth" in record and not _is_count(record["delegationDepth"]):
        raise ValueError("session header delegationDepth must be a non-negative safe integer")
    if "agentPreset" in record and not isinstance(record["agentPreset"], str):
        raise ValueError("session header agentPreset must be a string")
    return deep_freeze(dict(record)) if type(record) is not FrozenJsonObject else record


def _validate_restored_session_header(session_id: SessionId, value: object) -> SessionHeader:
    if isinstance(value, Mapping) and type(value) not in _PLAIN_RECORDS:
        raise ValueError("session header is not a plain JSON record")
    return _validate_session_header(session_id, value)


def _snapshot_session_header(session_id: SessionId, source: SessionHeader | None) -> SessionHeader:
    value: object = (
        {
            "version": SESSION_FORMAT_VERSION,
            "id": session_id,
            "createdAt": _now(),
            "isSeeded": False,
        }
        if source is None
        else source
    )
    snapshot = snapshot_json_value(value, freeze=True)
    if snapshot is None:
        raise ValueError("session header is not losslessly JSON-serializable")
    return _validate_session_header(session_id, snapshot)


# --- event validation --------------------------------------------------------------


def _has_provider_model(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    provider, model = value.get("provider"), value.get("model")
    return isinstance(provider, str) and bool(provider) and isinstance(model, str) and bool(model)


def _assert_message_event_shape(event: Mapping[str, Any], subject: str) -> None:
    kind = event.get("type")
    if kind not in SURFACE_EVENT_TYPES:
        return
    data = event.get("data")
    record = data if isinstance(data, Mapping) else None
    message = record if kind == "user/message" else (record or {}).get("message")
    if (
        not isinstance(message, Mapping)
        or not isinstance(message.get("id"), str)
        or not message["id"]
    ):
        raise ValueError(f"{subject} lacks an identified message")
    expected_role = _SURFACE_ROLES[kind]
    if message.get("role") != expected_role:
        raise ValueError(f'{subject} message must have role "{expected_role}"')
    source = message.get("source")
    if (
        not isinstance(source, Mapping)
        or not isinstance(source.get("kind"), str)
        or not source["kind"]
    ):
        raise ValueError(f"{subject} message has invalid source")
    if not isinstance(message.get("content"), list):
        raise ValueError(f"{subject} message has invalid content")
    if kind == "system/message":
        if source["kind"] != "system-prompt":
            raise ValueError(f"{subject} message must have system-prompt source")
        return
    if kind == "assistant/message":
        if source["kind"] != "model" or not _has_provider_model(source):
            raise ValueError(f"{subject} message must have model source")
        return
    if kind != "tool/result":
        return
    call_id = source.get("callId")
    if source["kind"] != "tool" or not isinstance(call_id, str) or not call_id:
        raise ValueError(f"{subject} message must have tool source")
    if message.get("toolCallId") != call_id:
        raise ValueError(f"{subject} message has mismatched tool call ids")


def _assert_assistant_settlement_shape(data: object, kind: str, index: int) -> None:
    record = data if isinstance(data, Mapping) else {}
    if (
        not _is_count(record.get("turn"))
        or not _is_count(record.get("step"))
        or not isinstance(record.get("stream"), list)
    ):
        raise ValueError(f"seed {kind} at index {index} has invalid settlement fields")


def _assert_adapter_defaults(value: object, config: Mapping[str, Any], index: int) -> None:
    if not isinstance(value, Mapping) or (
        any(key not in _ADAPTER_DEFAULT_KEYS for key in value)
        or any(marker is not True for marker in value.values())
        or (value.get("reasoningEffort") is True and "reasoningEffort" not in config)
        or (value.get("maxTokens") is True and "maxTokens" not in config)
    ):
        raise ValueError(f"seed request/header at index {index} has invalid adapterDefaults")


def _assert_current_llm_shape(event: Mapping[str, Any], index: int) -> None:
    data = event.get("data")
    record = data if isinstance(data, Mapping) else None
    kind = event.get("type")
    if kind == "request/header":
        header = (record or {}).get("header") or {}
        config = header.get("config")
        if not isinstance(config, Mapping) or not _has_provider_model(config):
            raise ValueError(f"seed request/header at index {index} lacks provider/model")
        effort = config.get("reasoningEffort")
        if "reasoningEffort" in config and (not isinstance(effort, str) or not effort):
            raise ValueError(f"seed request/header at index {index} has an invalid reasoningEffort")
        if "adapterDefaults" in header:
            _assert_adapter_defaults(header["adapterDefaults"], config, index)
        if (record or {}).get("reason") not in ("initial", "resume", "change", "series"):
            raise ValueError(f"seed request/header at index {index} has an invalid reason")
        starts = (record or {}).get("startsSeries")
        if "startsSeries" in (record or {}) and starts is not True:
            raise ValueError(
                f"seed request/header at index {index} has an invalid startsSeries marker"
            )
    if kind == "assistant/attempt":
        _assert_assistant_settlement_shape(record, kind, index)
        return
    if kind not in SURFACE_EVENT_TYPES:
        return
    _assert_message_event_shape(event, f"seed {kind} at index {index}")
    if kind == "assistant/message":
        _assert_assistant_settlement_shape(record, kind, index)


def _assert_session_event_envelope(value: object, index: int) -> Mapping[str, Any]:
    invalid = f"seed event at index {index} has an invalid event envelope"
    if not isinstance(value, Mapping) or any(key not in _ENVELOPE_KEYS for key in value):
        raise ValueError(invalid)
    seq, stamp = value.get("seq"), value.get("time")
    if (
        not isinstance(value.get("type"), str)
        or not _is_count(seq)
        or not (isinstance(stamp, int) and not isinstance(stamp, bool) and is_safe_integer(stamp))
        or "data" not in value
        or ("ignorable" in value and value["ignorable"] is not True)
    ):
        raise ValueError(invalid)
    validate_session_event_data(value, f"seed {value['type']} at index {index}")
    if value["type"] in (
        "request/header",
        "developer/message",
        "system/message",
        "user/message",
        "assistant/attempt",
        "assistant/message",
        "tool/result",
    ):
        _assert_current_llm_shape(value, index)
    return value


def adopt_session_event(event: SessionEvent) -> SessionEvent:
    """Validate one event and return it deeply frozen.

    Upstream validates an exclusively owned event and freezes its message in place.
    Python can't freeze in place, so the result is a frozen copy unless ``event`` is
    already frozen. History relations are not checked.
    """
    validate_session_event_data(event, f"session event at seq {event['seq']}")
    validate_surface_metadata(event)
    _assert_message_event_shape(event, f"session event at seq {event['seq']}")
    return deep_freeze(event) if not isinstance(event, FrozenJsonObject) else event


def snapshot_session_event(event: SessionEvent) -> SessionEvent:
    """Detach one event while preserving deep immutability for its message."""
    snapshot = snapshot_json_value(event, freeze=True)
    if snapshot is None:
        raise ValueError("session event is not losslessly JSON-serializable")
    return adopt_session_event(snapshot)


# --- plugin records ----------------------------------------------------------------


def is_plugin_record_type(type_: str) -> bool:
    """Whether a type name follows the ``plugin:`` record grammar."""
    return _PLUGIN_RECORD_TYPE.fullmatch(type_) is not None


def append_plugin_record(session: Session, type_: str, data: object) -> SessionSeq:
    """Append one experimental plugin record: an ``ignorable`` event outside the core vocabulary.

    Readers that don't recognize the type retain and skip it; it never enters the
    model-visible surface. Read records back with :func:`plugin_record_of`.
    """
    return session._append_record(type_, data)


def plugin_record_of(event: SessionEvent) -> PluginRecord | None:
    """Read a committed event as a plugin record, or ``None`` when it isn't one."""
    kind = event["type"]
    if event.get("ignorable") is not True or not is_plugin_record_type(kind):
        return None
    return PluginRecord(type=kind, seq=event["seq"], time=event["time"], data=event["data"])


# --- dispatch helpers --------------------------------------------------------------


def _collect_callbacks(
    ctx: Context, name: str, args: tuple[object, ...], carrier: ScopeCarrier[Session]
) -> list[Callable[..., object]]:
    return list(ctx.events.dispatch("emit", name, args, carrier))


def _invoke_contained(
    session_id: SessionId,
    name: str,
    args: tuple[object, ...],
    callbacks: Sequence[Callable[..., object]],
) -> None:
    describe = f'session "{session_id}": {name} listener'
    for callback in callbacks:
        try:
            contain(callback(*args), describe, logger)
        except Exception as error:
            logger.warning("%s threw: %s", describe, error)


# --- Session -----------------------------------------------------------------------

_Announcement = Literal["pending", "dispatching", "complete", "failed"]


@dataclass(eq=False)
class _Entry:
    id: SessionId
    session: Session
    carrier: ScopeCarrier[Session]
    emit_ctx: Context
    store: SessionStore
    announcement: _Announcement = "pending"
    appending: bool = False
    detach_requested: bool = False

    def detach(self) -> None:
        self.store._detach_entered(self)


# Store-owned Sessions map to None while unentered; absent keys are independent.
_attachments: weakref.WeakKeyDictionary[Session, _Entry | None] = weakref.WeakKeyDictionary()
_INDEPENDENT = object()


class Session:
    """An event-sourced session: an append-only log of frozen events.

    Create live instances with ``ctx.sessions.create()`` and detached ones with
    :meth:`create`. Seeding with an existing log replays or forks a session.
    """

    def __init__(
        self,
        session_id: SessionId,
        seed: Sequence[SessionEvent] | None,
        header: SessionHeader | None,
        mode: Literal["snapshot"] | SessionSeedEventState,
        supplied_inherited_event_count: int | None,
        projections: Sequence[SessionMessageProjection],
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCT:
            raise TypeError("use Session.create(), Session.from_restore() or ctx.sessions")
        self._log: list[SessionEvent] = []
        self._surface = SurfaceManager(self._log, SessionLogOffset(0), projections)
        self._events_snapshot: tuple[SessionEvent, ...] | None = None
        self._header_fold: EpochHeader | None = None
        self._header_fold_seq = 0
        self._context_fold: RequestContext | None = None
        self._context_fold_seq = 0
        self._tool_history = ToolHistoryProjection()
        self._tool_history_seq = 0
        self._derived: list[Message] = []
        self._derived_nodes = 0
        self._derived_generation = 0
        restored = (
            None if mode == "snapshot" else _validate_restored_session_header(session_id, header)
        )
        if seed is not None:
            for index, source in enumerate(seed):
                if mode == "snapshot":
                    snapshot = snapshot_json_value(source, freeze=True)
                    if snapshot is None:
                        raise ValueError(
                            f"seed event at index {index} is not losslessly JSON-serializable"
                        )
                else:
                    snapshot = deep_freeze(source) if isinstance(source, dict) else source
                event = _assert_session_event_envelope(snapshot, index)
                if event["seq"] != index:
                    raise ValueError(
                        f"seed event at index {index} has seq {event['seq']} (expected {index}); "
                        "seed must be contiguous from 0"
                    )
                try:
                    self._surface.validate_next(event)
                except ValueError as error:
                    raise ValueError(f"invalid seed event at index {index}: {error}") from error
                self._log.append(event)
        self.first_live_seq = SessionLogOffset(len(self._log))
        """The constructor seed length, before any marker appended during construction."""
        self.header: SessionHeader = restored or _snapshot_session_header(session_id, header)
        """Detached, deep-frozen creation metadata, kept outside the event log."""
        seeded = self.header["isSeeded"]
        if seeded and seed is None:
            raise ValueError("seeded session requires an explicit constructor seed")
        if seeded and supplied_inherited_event_count is None:
            raise ValueError("seeded session requires an inherited event count")
        inherited = session_log_offset(supplied_inherited_event_count or 0)
        if not seeded and inherited != 0:
            raise ValueError("unseeded session inherited event count must be 0")
        if inherited > len(self._log):
            raise ValueError("session inherited event count exceeds its event log")
        marker = self._log[inherited] if inherited < len(self._log) else None
        marked_seed = (
            marker is not None
            and marker["type"] == "session/end-seed"
            and marker["data"].get("inherited") is True
        )
        if mode == "snapshot" and seeded and inherited != len(self._log) and not marked_seed:
            raise ValueError(
                "seeded session constructor seed must equal its inherited prefix or mark its "
                "inherited cut"
            )
        if marked_seed and any(
            event["type"] == "session/end-seed" and event["data"].get("inherited") is True
            for event in self._log[inherited + 1 :]
        ):
            raise ValueError(
                "session inherited event count must identify the final inherited marker"
            )
        self.inherited_event_count = inherited
        """Number of leading events inherited from this Session's fork parent."""
        self.first_lifecycle_seq = (
            inherited if mode == "snapshot" and seeded else self.first_live_seq
        )
        """First event produced for this object lifecycle (not persisted)."""
        if seed is not None and mode == "snapshot" and seeded and not marked_seed:
            self.append("session/end-seed", {"inherited": True})
        elif (
            seed is not None
            and not (mode == "snapshot" and seeded)
            and (not self._log or self._log[-1]["type"] != "session/end-seed")
        ):
            self.append("session/end-seed", {})

    @classmethod
    def create(
        cls,
        session_id: SessionId | str,
        seed: Sequence[SessionEvent] | None = None,
        header: SessionHeader | None = None,
        inherited_event_count: int | None = None,
        projections: Sequence[SessionMessageProjection] = (),
    ) -> Session:
        """Create a detached session, validating and snapshotting borrowed seed and header."""
        return cls(
            SessionId(session_id),
            seed,
            header,
            "snapshot",
            inherited_event_count,
            projections,
            _CONSTRUCT,
        )

    @classmethod
    def from_restore(
        cls,
        session_id: SessionId | str,
        seed: Sequence[SessionEvent],
        header: SessionHeader,
        inherited_event_count: int,
        event_state: SessionSeedEventState,
        projections: Sequence[SessionMessageProjection] = (),
    ) -> Session:
        """Restore a detached session from independently owned or frozen storage values."""
        return cls(
            SessionId(session_id),
            seed,
            header,
            event_state,
            inherited_event_count,
            projections,
            _CONSTRUCT,
        )

    @property
    def id(self) -> SessionId:
        """The session identity, from its durable header."""
        return SessionId(self.header["id"])

    @property
    def surface(self) -> SurfaceManager:
        """The ordered surface over this session's event log."""
        return self._surface

    @property
    def seq(self) -> SessionLogOffset:
        """The next event's sequence number: always the log length."""
        return SessionLogOffset(len(self._log))

    def event_at(self, seq: int) -> SessionEvent | None:
        """The event at one exact sequence number (deprecated upstream for new calls)."""
        return self._log[seq] if 0 <= seq < len(self._log) else None

    def snapshot_events(
        self, from_seq: int = 0, to_seq_exclusive: int | None = None
    ) -> tuple[SessionEvent, ...]:
        """An immutable snapshot of a half-open event range; the full one is cached."""
        end = len(self._log) if to_seq_exclusive is None else to_seq_exclusive
        if from_seq == 0 and end == len(self._log):
            if self._events_snapshot is None:
                self._events_snapshot = tuple(self._log)
            return self._events_snapshot
        return tuple(self._log[from_seq:end])

    def own_events(self) -> tuple[SessionEvent, ...]:
        """This Session's events after its fork-inherited prefix."""
        return self.snapshot_events(self.inherited_event_count)

    def is_own_seq(self, seq: int) -> bool:
        """Whether one existing position is outside the fork-inherited prefix."""
        return self.inherited_event_count <= seq < len(self._log)

    def append(
        self,
        type_: str,
        data: object,
        *,
        surface_op: object = None,
        source_event_seqs: Sequence[SessionSeq] | None = None,
    ) -> SessionEvent:
        """Append one event to the log and synchronously notify ``session/event`` observers.

        Message-producing types require ``surface_op`` (``"append"`` or a replace
        mapping); other types reject it. ``data`` must be lossless JSON. Returns the logged
        event: its assigned ``seq``/``time`` plus the frozen snapshot of ``data``.
        """
        self._publication_entry()
        metadata: dict[str, object] = {}
        if source_event_seqs is not None:
            metadata["sourceEventSeqs"] = source_event_seqs
        if surface_op is not None:
            metadata["surfaceOp"] = surface_op
        data_snapshot = snapshot_json_value(data, freeze=True)
        if data_snapshot is None and not is_json_value(data):
            raise ValueError(f'session event "{type_}" carries non-JSON-serializable data')
        metadata_snapshot = snapshot_json_value(metadata, freeze=True)
        if metadata_snapshot is None:
            raise ValueError(
                f'session event "{type_}" carries non-JSON-serializable surface metadata'
            )
        entry = self._publication_entry()
        fields: dict[str, Any] = {
            "type": type_,
            "seq": len(self._log),
            "time": _now(),
            "data": data_snapshot,
            **metadata_snapshot,
        }
        event = FrozenJsonObject(fields)
        validate_session_event_data(event, f'session event "{type_}" at seq {event["seq"]}')
        self._commit(event, entry)
        return event

    def _append_record(self, type_: str, data: object) -> SessionSeq:
        self._publication_entry()
        data_snapshot = snapshot_json_value(data, freeze=True)
        if data_snapshot is None and not is_json_value(data):
            raise ValueError(f'plugin record "{type_}" carries non-JSON-serializable data')
        entry = self._publication_entry()
        fields: dict[str, Any] = {
            "type": type_,
            "seq": len(self._log),
            "time": _now(),
            "data": data_snapshot,
            "ignorable": True,
        }
        record = FrozenJsonObject(fields)
        if not is_plugin_record_type(type_):
            raise ValueError(
                f'plugin record type "{type_}" must be "plugin:" followed by lowercase '
                "slash-separated segments"
            )
        self._commit(record, entry)
        return SessionSeq(len(self._log) - 1)

    def _publication_entry(self) -> _Entry | None:
        entry = _attachments.get(self, _INDEPENDENT)
        if entry is None:
            raise ValueError(f'session "{self.id}" cannot append outside a SessionStore')
        if entry is _INDEPENDENT:
            return None
        assert isinstance(entry, _Entry)
        if entry.announcement != "complete":
            raise ValueError(f'session "{self.id}" cannot append before session/created completes')
        if entry.appending:
            raise ValueError(
                "session append cannot reenter while another append is being published"
            )
        return entry

    def _commit(self, event: SessionEvent, entry: _Entry | None) -> None:
        self._surface.validate_next(event)
        if entry is not None:
            entry.appending = True
        try:
            args = (self, event)
            callbacks = (
                None
                if entry is None
                else _collect_callbacks(entry.emit_ctx, "session/event", args, entry.carrier)
            )
            self._log.append(event)
            self._events_snapshot = None
            if entry is not None and callbacks is not None:
                _invoke_contained(entry.id, "session/event", args, callbacks)
        finally:
            if entry is not None:
                entry.appending = False
                if entry.detach_requested and entry.announcement != "dispatching":
                    entry.detach()

    def request_header(self) -> EpochHeader | None:
        """The header in force after the log's last ``request/header``, folded incrementally."""
        if self._header_fold_seq < len(self._log):
            folded = fold_request_header(self._log[self._header_fold_seq :], self._header_fold)
            self._header_fold = None if folded is None else deep_freeze(folded)
            self._header_fold_seq = len(self._log)
        return self._header_fold

    def request_context(self) -> RequestContext | None:
        """The latest ``request/context`` route metadata, folded incrementally."""
        if self._context_fold_seq < len(self._log):
            for event in self._log[self._context_fold_seq :]:
                if event["type"] == "request/context":
                    self._context_fold = event["data"]
            self._context_fold_seq = len(self._log)
        return self._context_fold

    def tool_history(self) -> ToolHistory:
        """Capability-independent tool history for request projection, folded incrementally."""
        for event in self._log[self._tool_history_seq :]:
            self._tool_history.apply(event)
        self._tool_history_seq = len(self._log)
        return self._tool_history.snapshot()

    def derive_messages(self) -> list[Message]:
        """Derive model history from the surface; cached across pure tail growth.

        Returns a fresh list per call; the frozen messages in it are shared.
        """
        nodes = self._surface.nodes
        generation = self._surface.content_generation
        if generation != self._derived_generation:
            self._derived = []
            self._derived_nodes = 0
            self._derived_generation = generation
        for seq in nodes[self._derived_nodes :]:
            message = self.derive_event_message(self._log[seq])
            if message is not None:
                self._derived.append(message)
        self._derived_nodes = len(nodes)
        return list(self._derived)

    def derive_event_message(self, event: SessionEvent) -> Message | None:
        """Project one event with all committed message projections applied."""
        return self._surface.derive_event_message(event)


_CONSTRUCT = object()


# --- SessionStore ------------------------------------------------------------------


SessionForkErrorCode = Literal[
    "SESSION_NOT_FOUND", "SESSION_NOT_LIVE", "SESSION_ALREADY_EXISTS", "INVALID_BOUNDARY"
]


class SessionForkError(Exception):
    """Typed error for session fork rejections."""

    def __init__(self, message: str, code: SessionForkErrorCode) -> None:
        super().__init__(message)
        self.code: SessionForkErrorCode = code


class SessionStore(Service):
    """The in-memory session store, provided as ``sessions``.

    Persistence is not implemented here: the agent lifecycle attaches a log writer to
    each published session.
    """

    name = "sessions"

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, config)
        self._store: dict[SessionId, _Entry] = {}
        self._counter = 0
        self._projections: list[SessionMessageProjection] = []

    @property
    def message_projections(self) -> Sequence[SessionMessageProjection]:
        """Borrowed definitions for detached replay."""
        return self._projections

    def caller(self, ctx: Context | None = None) -> Context:
        """The context a store operation acts for: upstream's ``this.ctx`` on a service call.

        Upstream Cordis hands a service method the caller's context, so effects (session
        entries, projection registrations) belong to the calling fiber and the caller's
        scope routes ``session/*`` events. deepseek-cordis services keep their own
        context, so the caller is ``ctx`` when given, else ``current_context()`` (set during
        plugin setup, effects and tasks they start), else the store's own context.
        """
        candidate = ctx if ctx is not None else current_context()
        if candidate is None or candidate.root is not self.ctx.root:
            return self.ctx
        return candidate

    def register_message_projection(
        self, projection: SessionMessageProjection, *, ctx: Context | None = None
    ) -> Callable[[], object]:
        """Register one event interpreter for live creation, restore and fork.

        The registration belongs to the caller's fiber (see :meth:`caller`).
        """
        if any(item.type == projection.type for item in self._projections):
            raise ValueError(
                f'session message projection "{projection.type}" is already registered'
            )

        def setup() -> Callable[[], None]:
            self._projections.append(projection)
            return lambda: self._projections.remove(projection)

        return self.caller(ctx).effect(setup, "sessions.registerMessageProjection()")

    def create(
        self,
        session_id: SessionId | str | None = None,
        options: CreateSessionOptions | None = None,
        *,
        ctx: Context | None = None,
    ) -> Session:
        """Create a session owned by the calling fiber (see :meth:`caller`), entered and
        announced. Unloading that fiber removes the session from the store."""
        owner = self.caller(ctx)
        session = self.prepare(session_id, options)

        def setup() -> Iterator[Callable[[], None]]:
            yield self.enter(session, ctx=owner)
            self.announce(session)

        owner.effect(setup, "sessions.create()")
        return session

    def prepare(
        self,
        session_id: SessionId | str | None = None,
        options: CreateSessionOptions | RestoredSessionOptions | None = None,
    ) -> Session:
        """Build a session without entering it; pairs with :meth:`enter` and :meth:`announce`."""
        if session_id is None:
            while True:
                self._counter += 1
                sid = SessionId(f"session-{self._counter}")
                if sid not in self._store:
                    break
        else:
            sid = SessionId(session_id)
        if sid in self._store:
            raise ValueError(f'session "{sid}" already exists')
        if isinstance(options, RestoredSessionOptions):
            session = Session.from_restore(
                sid,
                options.seed,
                options.meta,
                options.inherited_event_count,
                options.event_state,
                self._projections,
            )
            _attachments[session] = None
            return session
        meta = None if options is None else options.meta
        header: dict[str, object] = {
            "version": SESSION_FORMAT_VERSION,
            "id": sid,
            "createdAt": _now() if meta is None or meta.created_at is None else meta.created_at,
        }
        if meta is not None and meta.cwd is not None:
            header["cwd"] = meta.cwd
        if meta is not None and meta.parent_session is not None:
            header["parentSession"] = meta.parent_session
        header["isSeeded"] = False if meta is None or meta.is_seeded is None else meta.is_seeded
        if meta is not None and meta.origin is not None:
            header["origin"] = meta.origin
        if meta is not None and meta.delegation_depth is not None:
            header["delegationDepth"] = meta.delegation_depth
        if meta is not None and meta.agent_preset is not None:
            header["agentPreset"] = meta.agent_preset
        session = Session.create(
            sid,
            None if options is None else options.seed,
            header,
            None if options is None else options.inherited_event_count,
            self._projections,
        )
        _attachments[session] = None
        return session

    def enter(self, session: Session, *, ctx: Context | None = None) -> Callable[[], None]:
        """Enter a session into the store; returns the detach disposer.

        Writes stay disabled until :meth:`announce` completes. The caller's scope
        (see :meth:`caller`) routes this session's ``session/*`` events.
        """
        sid = session.id
        owner = self.caller(ctx)
        carrier = scope_target(session, scope_of(owner))
        if sid in self._store:
            raise ValueError(f'session "{sid}" already exists')
        if _attachments.get(session) is not None:
            raise ValueError(f'session "{sid}" is already attached to a store')
        entry = _Entry(sid, session, carrier, owner, self)
        self._store[sid] = entry
        _attachments[session] = entry
        entered = True

        def detach() -> None:
            nonlocal entered
            if not entered:
                return
            entered = False
            if entry.announcement == "dispatching" or entry.appending:
                entry.detach_requested = True
                return
            entry.detach()

        return detach

    def _detach_entered(self, entry: _Entry) -> None:
        entry.detach_requested = False
        if self._store.get(entry.id) is not entry:
            return
        del self._store[entry.id]
        _attachments[entry.session] = None
        if entry.announcement != "pending":
            self._emit_disposed(entry)

    def announce(self, session: Session) -> None:
        """Emit ``session/created`` once for an entered session; success enables appends."""
        entry = self._live_entry_for(session)
        if entry.announcement != "pending":
            raise ValueError(f'session "{entry.id}" was already announced')
        args = (session,)
        entry.announcement = "dispatching"
        try:
            callbacks = _collect_callbacks(self.ctx, "session/created", args, entry.carrier)
            for callback in callbacks:
                # A synchronous throw vetoes publication; an async rejection is logged.
                contain(callback(*args), f'session "{entry.id}": session/created listener', logger)
            entry.announcement = "complete"
        except BaseException:
            entry.announcement = "failed"
            raise
        finally:
            if entry.detach_requested:
                entry.detach()

    def _emit_disposed(self, entry: _Entry) -> None:
        args = (entry.session,)
        try:
            callbacks = _collect_callbacks(self.ctx, "session/disposed", args, entry.carrier)
            _invoke_contained(entry.id, "session/disposed", args, callbacks)
        except Exception as error:
            logger.warning('session "%s": session/disposed dispatch threw: %s', entry.id, error)

    async def flush(self, session: Session) -> bool:
        """Dispatch the awaited ``session/flush`` durability checkpoint for ``session``.

        Returns whether any listener participated, after all settle; raises the first
        registered listener's failure.
        """
        entry = self._live_entry_for(session)
        args = (session,)
        callbacks = _collect_callbacks(self.ctx, "session/flush", args, entry.carrier)

        async def run(callback: Callable[..., object]) -> object:
            result = callback(*args)
            if inspect.isawaitable(result):
                return await result
            return result

        results = await asyncio.gather(*(run(cb) for cb in callbacks), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return len(callbacks) > 0

    def _live_entry_for(self, session: Session) -> _Entry:
        entry = _attachments.get(session)
        if entry is None or self._store.get(entry.id) is not entry:
            raise ValueError(f'session "{session.id}" is not live in this store')
        return entry

    def get(self, session_id: SessionId | str) -> Session | None:
        """Look up a live session."""
        entry = self._store.get(SessionId(session_id))
        return None if entry is None else entry.session

    def fork(
        self,
        source: Session | SessionId | str,
        boundary: int | None = None,
        child_session_id: SessionId | str | None = None,
        *,
        ctx: Context | None = None,
    ) -> Session:
        """Create an independent live session from an exact prefix of a live source.

        ``boundary`` is an inclusive source seq (default: the last event). An open tail
        receives synthetic tool results and step/turn closers with the forked cause.
        The child belongs to the calling fiber resolved by :meth:`caller`.
        """
        if child_session_id is not None and self.get(child_session_id) is not None:
            raise SessionForkError(
                f'session "{child_session_id}" already exists', "SESSION_ALREADY_EXISTS"
            )
        live = self._resolve_fork_source(source)
        events = live.snapshot_events()
        resolved = self._fork_boundary(live.id, events, boundary)
        seed = [] if resolved is None else build_fork_seed(events, SessionSeq(resolved))
        return self.create(
            child_session_id,
            CreateSessionOptions(
                seed=seed,
                inherited_event_count=SessionLogOffset(0 if resolved is None else resolved + 1),
                meta=CreateSessionMeta(cwd=live.header.get("cwd"), is_seeded=True),
            ),
            ctx=ctx,
        )

    def _fork_boundary(
        self, sid: SessionId, events: Sequence[SessionEvent], requested: int | None
    ) -> int | None:
        last = events[-1] if events else None
        if requested is None:
            if last is None:
                return None
            boundary: object = last["seq"]
        else:
            boundary = requested
        if not _is_count(boundary):
            raise SessionForkError(
                f'fork boundary for session "{sid}" must be a non-negative safe integer, '
                f"got {boundary}",
                "INVALID_BOUNDARY",
            )
        assert isinstance(boundary, int)
        if boundary >= len(events):
            last_seq = "none" if last is None else last["seq"]
            raise SessionForkError(
                f'fork boundary {boundary} does not exist in session "{sid}" '
                f"(last seq: {last_seq})",
                "INVALID_BOUNDARY",
            )
        if events[boundary]["seq"] != boundary:
            raise SessionForkError(
                f"fork boundary {boundary} does not match a contiguous event seq in "
                f'session "{sid}"',
                "INVALID_BOUNDARY",
            )
        return boundary

    def _resolve_fork_source(self, source: Session | SessionId | str) -> Session:
        if isinstance(source, str):
            found = self.get(source)
            if found is None:
                raise SessionForkError(f'session "{source}" not found', "SESSION_NOT_FOUND")
            return found
        live = self.get(source.id)
        if live is None:
            raise SessionForkError(f'session "{source.id}" not found', "SESSION_NOT_FOUND")
        if live is not source:
            raise SessionForkError(
                f'session "{source.id}" is not the live store instance', "SESSION_NOT_LIVE"
            )
        return source

    def list(self) -> list[Session]:
        """All live sessions, in creation order."""
        return [entry.session for entry in self._store.values()]
