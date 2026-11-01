"""Session-projection seam: projection units, their eager drive and the read faces.

Upstream: packages/session/session-projection (``@deepseek-ai/dsh-session-projection``) at ce7073d.

Python differences:

- The declaration-merged type table (``SessionProjectionMap``,
  ``SessionProjectionSummaryExcluded`` and the view/summary value types) is type-only
  upstream and has no Python counterpart: keys are plain ``str``, and the declared
  ``wire.summary`` stays the only runtime fact of summary membership.
- zod schemas become any object with ``validate_python(value)``
  (:class:`ProjectionSchema`), such as a pydantic ``TypeAdapter``; a failure raises the
  validator's own error.
- Definitions are frozen keyword-only dataclasses (:class:`ProjectionDefinition`,
  :class:`ProjectionWire`) with snake_case fields; ``wire=None`` is a host-only unit and
  ``summary`` defaults to ``True``. Snapshots and restore results are dataclasses with
  ``as_of_seq`` / ``values`` and ``snapshot`` / ``checkpoint``; checkpoint rows keep the
  persisted ``{ver, seq, val}`` shape.
- ``Object.is`` comparisons are identity for objects and value equality for numbers and
  strings (Python does not intern every equal ``int`` or ``str``).
- Upstream Cordis hands service methods the caller's context. Here ``register`` and
  ``on_changed`` take an optional ``ctx=`` and otherwise use ``current_context()``, else the
  registry's own context (:meth:`SessionProjectionRegistry.caller`, the same rule as
  ``SessionStore.caller``).
- ``register()`` returns a :class:`ProjectionRegistration`. On success it is a
  deepseek-cordis ``Effect`` (so ``return``, ``ctx.effect(...)`` and synchronous generator
  ``yield`` adopt it into a composing effect) that is also the handle: awaiting it yields
  itself after the handoff, which runs on the next event-loop turn (``loop.call_soon``,
  upstream a microtask) and fails with deepseek-cordis ``CordisError("INACTIVE_EFFECT")``
  (via ``Fiber.assert_active``; upstream "inactive context") once the owner was
  disposed. The handle methods are reachable before the await. Calling it is the effect
  disposer and returns ``Awaitable[None] | None`` like every deepseek-cordis disposer
  (upstream always a Promise). A registration that fails before its effect exists returns
  a callable no-op whose await raises.
- There is no thenable assimilation: an ``async`` plugin returning the registration, or an
  async generator yielding it, adopts it as cleanup instead of receiving the handle, and
  there is no ``then``/``catch``/``finally`` chaining.
- A rejection counts as observed once the registration is awaited; an unobserved one is
  checked on a later loop turn (upstream ``setImmediate``) and logged at ``ERROR``. With no
  running loop the check runs immediately.
- ``ctx.logger`` becomes this package's :mod:`logging` loggers; the deprecation notice of
  ``state_of(session, key)`` is a ``logging`` warning, not a :mod:`warnings` warning.
- ``checkpoint`` detaches states with ``thaw_json`` (a plain mutable JSON copy) instead of
  ``structuredClone``.
- Units and change listeners removed while an event is being driven are skipped, as
  upstream's live ``Map``/``Set`` iteration skips deleted entries; additions made during a
  drive take effect from the next event.
"""

from .registry import SessionProjectionRegistry
from .types import (
    ProjectionChangeListener,
    ProjectionCheckpoint,
    ProjectionCheckpointRow,
    ProjectionDefinition,
    ProjectionHandle,
    ProjectionKeySelection,
    ProjectionRegistration,
    ProjectionRestoreResult,
    ProjectionSchema,
    ProjectionSnapshot,
    ProjectionWire,
    SessionSeqCursor,
)

__all__ = [
    "ProjectionChangeListener",
    "ProjectionCheckpoint",
    "ProjectionCheckpointRow",
    "ProjectionDefinition",
    "ProjectionHandle",
    "ProjectionKeySelection",
    "ProjectionRegistration",
    "ProjectionRestoreResult",
    "ProjectionSchema",
    "ProjectionSnapshot",
    "ProjectionWire",
    "SessionProjectionRegistry",
    "SessionSeqCursor",
]
