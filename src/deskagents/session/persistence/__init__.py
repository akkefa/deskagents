"""Durable session-persistence Service Definition (``sessionPersistence``).

Upstream: packages/session/session-persistence (``@deepseek-ai/dsh-session-persistence``)
at ce7073d.

Backends store session events as the event-sourced log and keep non-replayable header
metadata separately; callers address one stored session through a
:class:`SessionHandle` from :meth:`SessionPersistence.create` or ``open``.

Python differences:

- Methods are coroutines; upstream's ``AbortSignal`` options become ordinary asyncio
  task cancellation, so there is no ``signal`` parameter.
- ``Symbol.asyncDispose`` on handles becomes ``async with``.
- ``identity`` is a per-instance ``object()`` (upstream: a ``Symbol``).
- ``flush()`` failures raise an :class:`ExceptionGroup` naming each failed session
  (upstream: ``AggregateError``).
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NewType

from deepseek_cordis import Context, Service

from deskagents.core.session import SessionEvent, SessionHeader, SessionId, SessionLogOffset

from .errors import (
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionFormatUnsupportedError,
    SessionHandleClosedError,
    SessionLocation,
    SessionOwnershipLostError,
    SessionPersistenceCorruptionError,
    SessionPersistenceNotFoundError,
    SessionReadOnlyError,
    session_format_version_refusal,
)
from .handle import SessionAccess, SessionHandle, SessionHandleReadResult
from .storage_contract import (
    assert_contiguous,
    assert_stored_id,
    assert_version,
    materialize_append_batch,
    materialize_create_header,
    validate_stored_events,
)

__all__ = [
    "SessionAccess",
    "SessionAlreadyExistsError",
    "SessionAlreadyOwnedError",
    "SessionFormatUnsupportedError",
    "SessionHandle",
    "SessionHandleClosedError",
    "SessionHandleReadResult",
    "SessionInspection",
    "SessionLocation",
    "SessionOwnershipLostError",
    "SessionPersistence",
    "SessionPersistenceCorruptionError",
    "SessionPersistenceNotFoundError",
    "SessionPersistenceRevision",
    "SessionPersistenceSnapshot",
    "SessionReadOnlyError",
    "SessionStorageMetadata",
    "assert_contiguous",
    "assert_stored_id",
    "assert_version",
    "materialize_append_batch",
    "materialize_create_header",
    "session_format_version_refusal",
    "validate_stored_events",
]

SessionPersistenceRevision = NewType("SessionPersistenceRevision", str)
"""Opaque backend token identifying one storage source and one revision of a session log."""


@dataclass(frozen=True)
class SessionPersistenceSnapshot:
    """Lightweight stored-session observation from ``stat``/``list``, without the event log."""

    header: SessionHeader
    revision: SessionPersistenceRevision
    """Opaque change token; equal revisions from one instance and id mean an unchanged log."""
    format_status: Literal["current", "migration-required"] | None = None
    event_count: int | None = None
    """Logical event count, when cheaply available from metadata."""
    size_bytes: int | None = None
    """Physical artifact byte size, when cheaply available (JSONL)."""
    location: SessionLocation | None = None
    """The selected per-Session artifact, when the backend materializes one."""


@dataclass(frozen=True)
class SessionStorageMetadata:
    """A logical header paired with its exact inherited cut."""

    meta: SessionHeader
    inherited_event_count: SessionLogOffset


@dataclass(frozen=True)
class SessionInspection(SessionStorageMetadata):
    """Immutable logical session read: storage metadata plus the complete validated log."""

    events: Sequence[SessionEvent] = ()


class SessionPersistence(Service, abc.ABC):
    """Durable append-only session storage addressed through per-session handles.

    Shared semantics: events are contiguous from seq 0 and never rewritten; a torn
    physical tail is never returned and is truncated before the first append; reads
    validate current-format records only and refuse unknown vocabulary. ``append``
    persists best-effort; ``flush`` (per handle or service-wide) is the durability
    barrier. A created session is visible through ``stat``/``list``/``open`` in this
    process at once; other processes see it once it materializes.
    """

    name = "sessionPersistence"

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, config)
        self.identity = object()
        """Process-local instance identity, distinct after service replacement."""

    @abc.abstractmethod
    async def create(
        self, header: SessionHeader, *, inherited_event_count: int | None = None
    ) -> SessionHandle:
        """Create a new stored session and take its write ownership.

        ``inherited_event_count`` is required when ``header.isSeeded`` is true and must
        be omitted or ``0`` otherwise. Raises ``SessionAlreadyExistsError``.
        """

    @abc.abstractmethod
    async def open(self, session_id: SessionId | str, access: SessionAccess) -> SessionHandle:
        """Open an existing session; ``write`` claims single-writer ownership.

        Raises ``SessionPersistenceNotFoundError`` or, for ``write``,
        ``SessionAlreadyOwnedError``.
        """

    @abc.abstractmethod
    async def flush(self) -> None:
        """Flush every active write handle of this instance in one durability barrier."""

    @abc.abstractmethod
    async def stat(self, session_id: SessionId | str) -> SessionPersistenceSnapshot | None:
        """Observe one stored session without reading its log or taking ownership."""

    @abc.abstractmethod
    async def list(self) -> Sequence[SessionPersistenceSnapshot]:
        """Every stored session visible to this process, in no promised order."""
