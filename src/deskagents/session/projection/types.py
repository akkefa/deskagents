"""Public contracts of the session-projection seam: units, handles, reads and checkpoints."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypedDict, TypeVar

from deskagents.core.session import SessionEvent, SessionHeader, SessionLogOffset, SessionSeq

if TYPE_CHECKING:
    from deskagents.core.session import Session

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
    "SessionSeqCursor",
]

S = TypeVar("S")
T_co = TypeVar("T_co", covariant=True)
S_co = TypeVar("S_co", covariant=True)

SessionSeqCursor: TypeAlias = int
"""An inclusive event cursor: the seq of an existing event, or ``-1`` for an empty log."""


class ProjectionSchema(Protocol[T_co]):
    """A validator for persisted state or a wire payload (upstream: a zod schema).

    A pydantic ``TypeAdapter`` satisfies it. ``validate_python`` returns the validated
    value and raises when the value does not match.
    """

    def validate_python(self, value: Any, /) -> T_co: ...


@dataclass(frozen=True, kw_only=True)
class ProjectionWire(Generic[S]):
    """A unit's client view. Omit it (``wire=None``) for host-only units."""

    view_schema: ProjectionSchema[Any]
    """Validates the wire payload before it leaves the host."""
    view: Callable[[S], object]
    """State to wire payload. The live drive keeps the two latest raw results and compares
    them by identity; an object-valued view must reuse its result to suppress publication
    across internal-only state changes."""
    summary: bool = True
    """Whether a Session summary carries this unit; ``False`` leaves it out."""


@dataclass(frozen=True, kw_only=True)
class ProjectionDefinition(Generic[S]):
    """One domain's state-driven computation unit: a pure synchronous fold.

    The framework drives ``apply`` on every committed session event. All functions must
    be synchronous and the state must be plain JSON (the persisted-cache precondition).
    """

    key: str
    """The projection key this unit owns."""
    state_schema: ProjectionSchema[S]
    """Validates persisted state before it seeds a fold."""
    init: Callable[[SessionHeader, SessionLogOffset], S]
    """State for the empty log, from the Session header and its exact fork-inherited
    prefix length."""
    apply: Callable[[S, SessionEvent], S]
    """Pure transition. A unit uninterested in an event must return the same state object:
    an unchanged state produces zero downstream work."""
    state_version: int
    """Persisted-cache invalidation version; bump when the state fields or fold semantics
    change. Non-negative integer."""
    wire: ProjectionWire[S] | None = field(default=None)
    """Client view; ``None`` for host-only units."""


class ProjectionHandle(Protocol[S_co]):
    """An effect-owned registration with a reader retained independently of registration."""

    def state_of(self, session: Session) -> S_co:
        """Read the current state, including after unregistering, by folding new Session
        events with this handle's original unit. Do not mutate the live value."""
        ...

    def reader_for(self, session: Session) -> Callable[[], S_co]:
        """A reader for one exact Session that neither reads state nor prepares it."""
        ...

    async def dispose(self) -> None:
        """Unregister from global drives and snapshots without invalidating this reader."""
        ...


class ProjectionRegistration(Protocol[S_co]):
    """What ``register()`` returns: awaiting it yields the ready handle; calling it releases
    the registration through the original Cordis effect disposer (deprecated)."""

    def __call__(self) -> Awaitable[None] | None: ...

    def __await__(self) -> Generator[Any, None, ProjectionHandle[S_co]]: ...


ProjectionChangeListener: TypeAlias = Callable[["Session", str, Any, SessionSeq], object]
"""Change-feed listener ``(session, key, value, seq)``: one unit's raw view changed for one
session. ``value`` is the schema-validated output; ``seq`` is the causing event's seq."""

ProjectionKeySelection: TypeAlias = Sequence[str] | AbstractSet[str]
"""One read's client-visible key selection: a key list, or a derived set such as
``summary_keys()``. A set is used as given; only a list is copied into one."""


@dataclass(frozen=True)
class ProjectionSnapshot:
    """One consistent read cut over the registered client-visible units for one session."""

    as_of_seq: SessionSeqCursor
    """Seq of the last event the values reflect; ``-1`` for an empty log."""
    values: dict[str, Any]
    """Whole current client value per registered key."""


class ProjectionCheckpointRow(TypedDict):
    """One unit's checkpoint: the persisted ``(sessionId, key, ver, seq, val)`` row minus
    its two outer keys. A row is never authoritative, only a fold shortcut."""

    ver: int
    """The registering unit's state version at fold time."""
    seq: SessionSeqCursor
    """Seq of the last event folded into ``val``; ``-1`` for the empty log."""
    val: Any
    """The unit's internal state, plain JSON per the unit contract."""


ProjectionCheckpoint: TypeAlias = dict[str, ProjectionCheckpointRow]
"""Checkpoint rows keyed by projection key (one session's persisted cache value)."""


@dataclass(frozen=True)
class ProjectionRestoreResult:
    """A cold read's snapshot plus the refreshed checkpoint rows at the same cut."""

    snapshot: ProjectionSnapshot
    checkpoint: ProjectionCheckpoint
