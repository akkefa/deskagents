"""Ownership of one unpublished Session before registry publication.

Upstream: packages/core/session/src/preparation.ts. ``Symbol.dispose`` becomes
:meth:`SessionPreparation.dispose` and the context-manager protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Session


class SessionPreparation:
    """One exact unpublished Session and the provider state that keeps it usable.

    Disposal is synchronous and idempotent. Providers decide whether ``release``
    returns the Session to a cache or discards it.
    """

    def __init__(self, session: Session, release: Callable[[], None] | None) -> None:
        self.session = session
        """The exact Session to use for setup and publication."""
        self._release = release
        self._released = False

    @classmethod
    def create(
        cls, session: Session, *, release: Callable[[], None] | None = None
    ) -> SessionPreparation:
        """Wrap an unpublished Session in one preparation lifetime."""
        return cls(session, release)

    def dispose(self) -> None:
        """Release provider state once when this preparation leaves its caller."""
        if self._released:
            return
        self._released = True
        if self._release is not None:
            self._release()

    def __enter__(self) -> SessionPreparation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.dispose()
