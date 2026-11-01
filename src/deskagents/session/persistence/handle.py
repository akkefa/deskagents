"""The per-session storage handle returned by ``SessionPersistence.create``/``open``.

Upstream: packages/session/session-persistence/src/handle.ts
"""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol

from deskagents.core.session import (
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
    SessionSeedEventState,
)

SessionAccess = Literal["read", "write"]
"""``write`` is the session's single mutator (it also reads); ``read`` only observes."""


@dataclass(frozen=True)
class SessionHandleReadResult:
    """One persistence event slice returned by :meth:`SessionHandle.read`."""

    event_state: SessionSeedEventState
    """Whether event values are exclusively owned or shared only after deep freezing."""
    events: Sequence[SessionEvent]
    """Event values in a caller-owned outer sequence."""


class SessionHandle(Protocol):
    """One open channel onto a stored session.

    Single-owner state: ``read`` never backtracks below what this handle observed, a
    write handle reads its own appends, and :meth:`close` is the one teardown
    (idempotent, uncancellable; ``async with`` delegates to it). Every operation on a
    closed handle raises ``SessionHandleClosedError``.

    Once an ``append`` or ``flush`` returns on a write handle, every read started
    afterwards on the same backend instance observes at least that prefix.
    """

    @property
    def id(self) -> SessionId: ...

    @property
    def header(self) -> SessionHeader:
        """The immutable stored header, fixed at create/open."""
        ...

    @property
    def inherited_event_count(self) -> SessionLogOffset:
        """Exact fork-inherited prefix length; ``0`` when the header is not seeded."""
        ...

    @property
    def access(self) -> SessionAccess: ...

    async def read(self, offset: int = 0, length: int | None = None) -> SessionHandleReadResult:
        """Read a slice of the valid contiguous logical log; a torn tail is never returned."""
        ...

    async def append(self, events: Sequence[SessionEvent]) -> None:
        """Append a contiguous batch continuing the current end (best-effort durability)."""
        ...

    async def flush(self) -> None:
        """The durability barrier: every acknowledged append is durable on return."""
        ...

    def close(self) -> Awaitable[None]:
        """Release the handle; a write handle completes pending durability first.

        Idempotent; awaiting it again joins the same teardown.
        """
        ...

    async def __aenter__(self) -> SessionHandle: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
