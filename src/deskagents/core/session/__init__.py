"""Event-sourced session service: append-only log, in-memory store and derived history.

Upstream: packages/core/session (``@deepseek-ai/dsh-session``) at ce7073d.

Python differences:

- Events, headers and messages are frozen JSON mappings with upstream's camelCase field
  names, because they are the persisted and model-visible JSON. Upstream freezes in
  place; Python returns frozen copies (already-frozen values are reused).
- ``append(type, data, *, surface_op=..., source_event_seqs=...)`` takes the surface
  intent as keyword arguments; upstream's compile-time requirement becomes the runtime
  surface check upstream also performs.
- ``session/*`` events are dispatched with a scope filter (``filter_=``) instead of a
  Cordis ``thisArg`` carrier. Listener callbacks are collected through
  ``ctx.events.dispatch``, as upstream does.
- Upstream Cordis gives a service method the caller's context, so a session or message
  projection belongs to the fiber that created it, and that caller's scope routes its
  ``session/*`` events. deepseek-cordis services keep their own context, so the store
  resolves the caller as an explicit ``ctx=`` argument, else ``current_context()`` (set
  during plugin setup, effects and the tasks they start), else itself
  (:meth:`SessionStore.caller`). Code calling the store from outside a plugin passes
  ``ctx=`` or runs inside ``with ctx.scope():``.
- ``ctx.logger`` warnings go to this module's :mod:`logging` logger.
- The typed-RPC (typert) lookup registration is out of scope and omitted.
- ``Symbol.dispose`` on :class:`SessionPreparation` becomes ``dispose()`` and ``with``.
"""

from .fork import build_fork_seed
from .known_event_types import KNOWN_SESSION_EVENT_TYPES, MESSAGE_PROJECTION_EVENT_TYPES
from .preparation import SessionPreparation
from .repair import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    ToolCallRecovery,
    interrupted_turn_closers,
    open_turn_closers,
)
from .request_header import canonical_header, fold_request_header, header_equals
from .seq_ranges import decode_seq_ranges, encode_seq_ranges
from .session import (
    Session,
    SessionForkError,
    SessionStore,
    adopt_session_event,
    append_plugin_record,
    is_plugin_record_type,
    plugin_record_of,
    snapshot_session_event,
)
from .surface import (
    SessionMessageProjection,
    SessionMessageProjectionContext,
    SurfaceFoldReplacement,
    SurfaceFoldResult,
    SurfaceManager,
    derive_event_message,
    fold_surface,
    is_append_surface_event,
    is_replacement_surface_event,
    is_surface_eligible_type,
    is_surface_event,
    validate_session_event_data,
    validate_surface_metadata,
)
from .types import (
    SESSION_EVENT_DATA,
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
    SurfaceOp,
    session_id,
    session_log_offset,
    session_seq,
)

__all__ = [
    "KNOWN_SESSION_EVENT_TYPES",
    "MESSAGE_PROJECTION_EVENT_TYPES",
    "SESSION_EVENT_DATA",
    "SESSION_FORMAT_VERSION",
    "SURFACE_EVENT_TYPES",
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "CreateSessionMeta",
    "CreateSessionOptions",
    "EpochHeader",
    "PluginRecord",
    "RequestContext",
    "RestoredSessionOptions",
    "Session",
    "SessionEvent",
    "SessionForkError",
    "SessionHeader",
    "SessionId",
    "SessionLogOffset",
    "SessionMessageProjection",
    "SessionMessageProjectionContext",
    "SessionPreparation",
    "SessionSeedEventState",
    "SessionSeq",
    "SessionStore",
    "SurfaceFoldReplacement",
    "SurfaceFoldResult",
    "SurfaceManager",
    "SurfaceOp",
    "ToolCallRecovery",
    "adopt_session_event",
    "append_plugin_record",
    "build_fork_seed",
    "canonical_header",
    "decode_seq_ranges",
    "derive_event_message",
    "encode_seq_ranges",
    "fold_request_header",
    "fold_surface",
    "header_equals",
    "interrupted_turn_closers",
    "is_append_surface_event",
    "is_plugin_record_type",
    "is_replacement_surface_event",
    "is_surface_eligible_type",
    "is_surface_event",
    "open_turn_closers",
    "plugin_record_of",
    "session_id",
    "session_log_offset",
    "session_seq",
    "snapshot_session_event",
    "validate_session_event_data",
    "validate_surface_metadata",
]
