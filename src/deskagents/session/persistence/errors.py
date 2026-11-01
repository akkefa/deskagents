"""Stable failures of the session-persistence service and its handles.

Upstream: packages/session/session-persistence/src/errors.ts
"""

from __future__ import annotations

from dataclasses import dataclass

from deskagents.core.session import SESSION_FORMAT_VERSION, SessionId


class SessionPersistenceNotFoundError(Exception):
    """The requested Session identity has no durable log visible to this caller."""

    def __init__(self, session_id: SessionId | str) -> None:
        super().__init__(f'session "{session_id}" not found')
        self.session_id = SessionId(session_id)


class SessionAlreadyExistsError(Exception):
    """``create`` targeted a Session identity that already exists in this backend."""

    def __init__(self, session_id: SessionId | str) -> None:
        super().__init__(f'session "{session_id}" already exists')
        self.session_id = SessionId(session_id)


class SessionAlreadyOwnedError(Exception):
    """A write open found the session already bound to an active write handle."""

    def __init__(self, session_id: SessionId | str) -> None:
        super().__init__(f'session "{session_id}" is already owned by an active write handle')
        self.session_id = SessionId(session_id)


class SessionReadOnlyError(Exception):
    """A mutation (``append``/``flush``) was called on a read handle."""

    def __init__(self, session_id: SessionId | str, operation: str) -> None:
        super().__init__(f'session "{session_id}": {operation} is not available on a read handle')
        self.session_id = SessionId(session_id)


class SessionOwnershipLostError(Exception):
    """A write handle's ownership is permanently gone; close it and reopen for write."""

    def __init__(self, session_id: SessionId | str) -> None:
        super().__init__(
            f'session "{session_id}": write ownership was lost; close this handle and reopen'
        )
        self.session_id = SessionId(session_id)


class SessionHandleClosedError(Exception):
    """An operation was called on a handle after ``close()``."""

    def __init__(self, session_id: SessionId | str, operation: str) -> None:
        super().__init__(f'session "{session_id}": {operation} on a closed handle')
        self.session_id = SessionId(session_id)


@dataclass(frozen=True)
class SessionLocation:
    """A backend-resolved per-Session artifact path, for diagnostics and export only."""

    kind: str
    """Backend-specific artifact kind, for example ``jsonl``."""
    path: str
    """Absolute path to this session's backend-owned artifact."""


class SessionPersistenceCorruptionError(Exception):
    """Durable session contents failed validation after a successful backend read.

    The original validation failure is the exception's ``__cause__``.
    """


class SessionFormatUnsupportedError(Exception):
    """The stored log is intact but this runtime cannot faithfully interpret it.

    Nothing is damaged; the raw log stays readable at :attr:`location` when the backend
    keeps one artifact per session.
    """

    def __init__(self, message: str, location: SessionLocation | None = None) -> None:
        super().__init__(message)
        self.location = location


def session_format_version_refusal(session_id: str, version: int) -> str:
    """Direction-aware refusal text for a stored format version this build doesn't read."""
    if version > SESSION_FORMAT_VERSION:
        return (
            f'session "{session_id}" uses log format v{version}, but this harness reads only '
            f"v{SESSION_FORMAT_VERSION}: the log was written by a newer harness — upgrade the "
            "harness to open it"
        )
    return (
        f'session "{session_id}" uses log format v{version}, older than the supported '
        f"v{SESSION_FORMAT_VERSION}, and this build ships no upgrade path for it"
    )
