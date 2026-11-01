"""``ctx.sessionProjections``: the projection unit table and its eager drive."""

from __future__ import annotations

import asyncio
import logging
import math
import weakref
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from deepseek_cordis import Context, Effect, Fiber, FiberState, Service, current_context

from deskagents.core.session import (
    Session,
    SessionEvent,
    SessionHeader,
    SessionLogOffset,
    SessionSeq,
)
from deskagents.util.values import format_json_number, is_safe_integer, stringify_json, thaw_json

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
    SessionSeqCursor,
)

__all__ = ["SessionProjectionRegistry"]

logger = logging.getLogger(__name__)

S = TypeVar("S")

_MISSING: Any = object()
"""An absent checkpoint row (upstream ``undefined``)."""
_NO_VIEW: Any = object()
"""An empty live-drive view slot: no cached comparison (upstream ``undefined``)."""


def _js_string(value: object) -> str:
    """``String(value)`` for the numbers and booleans named in diagnostics."""
    if value is _MISSING:
        return "undefined"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        if isinstance(value, float) and math.isnan(value):
            return "NaN"
        if isinstance(value, float) and math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return format_json_number(value)
    return str(value)


def _quote(key: str) -> str:
    """``JSON.stringify(key)``."""
    return stringify_json(key)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _same_value(left: object, right: object) -> bool:
    """``Object.is``: identity for objects, value equality for numbers and strings."""
    if left is right:
        return True
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if _is_number(left) and _is_number(right):
        assert isinstance(left, int | float) and isinstance(right, int | float)
        if left != left and right != right:
            return True
        if left == 0 and right == 0:
            return math.copysign(1.0, left) == math.copysign(1.0, right)
        return left == right
    return False


def _same_version(row_version: object, state_version: int) -> bool:
    """``row.ver === def.stateVersion`` (strict: a boolean never equals a number)."""
    return _is_number(row_version) and row_version == state_version


def _cursor_before(offset: int) -> SessionSeqCursor:
    """The inclusive cursor immediately before one log offset."""
    return -1 if offset == 0 else offset - 1


def _selected_keys(keys: ProjectionKeySelection | None) -> AbstractSet[str] | None:
    """Narrow one key selection to a set; a set is served without a copy."""
    if keys is None:
        return None
    return keys if isinstance(keys, AbstractSet) else set(keys)


@dataclass(eq=False)
class _Erased:
    """Type-erased unit view the drive machinery works with."""

    key: str
    state_schema: ProjectionSchema[Any]
    init: Callable[[SessionHeader, SessionLogOffset], Any]
    apply: Callable[[Any, SessionEvent], Any]
    view_schema: ProjectionSchema[Any] | None
    view: Callable[[Any], object] | None
    summary: bool | None
    """Whether a summary carries this unit; ``None`` for a host-only unit (upstream
    ``wire?.summary`` is ``undefined``)."""
    state_version: int

    @property
    def has_wire(self) -> bool:
        return self.view is not None

    def wire_value(self, state: object) -> object:
        """One schema-validated wire value."""
        if self.view is None or self.view_schema is None:
            raise ValueError(f"session projection {_quote(self.key)} has no wire view")
        return self.view_schema.validate_python(self.view(state))


@dataclass(eq=False)
class _Cell:
    """Per-session per-unit watermark and fixed live-drive view buffer."""

    state: Any
    observed_seq: SessionSeqCursor
    """Seq of the last event passed through ``apply`` (regardless of change)."""
    views: list[Any] = field(default_factory=lambda: [_NO_VIEW, _NO_VIEW])
    """``[previous_view, current_view]``; ``_NO_VIEW`` slots mean no cached comparison."""


@dataclass(eq=False)
class _Unit:
    """One shared unit and its per-session cells, retained by acquired readers after the
    final registration leaves the live table. ``refs`` counts live registrants."""

    definition: _Erased
    cells: weakref.WeakKeyDictionary[Session, _Cell] = field(
        default_factory=weakref.WeakKeyDictionary
    )
    refs: int = 0


def _watch_rejection(observed: Callable[[], bool], error: BaseException, owner: Fiber) -> None:
    """Log a registration failure nobody awaited, once the loop has had a turn to observe
    it; an owner that is no longer loading or active cannot use the handle, so drop it."""

    def report() -> None:
        if observed():
            return
        if owner.state is not FiberState.LOADING and owner.state is not FiberState.ACTIVE:
            return
        logger.error("%s", error, exc_info=error)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        report()
        return
    loop.call_soon(report)


class _Registration(Effect, Generic[S]):
    """A live registration: the Cordis effect disposer that is also the retained handle.

    Being an :class:`Effect`, a composing effect (``return``, ``ctx.effect(...)`` or a
    synchronous generator ``yield``) adopts it. Awaiting it yields itself as the handle.
    """

    def __init__(
        self, registry: SessionProjectionRegistry, owner: Context, definition: _Erased
    ) -> None:
        self.observed = False
        self._projection_registry = registry
        self._projection_owner = owner
        self._projection_unit: _Unit | None = None
        self._projection_failure: BaseException | None = None
        self._projection_ready: asyncio.Future[None] | None = None

        def install() -> Callable[[], None]:
            unit = registry._acquire(definition)
            self._projection_unit = unit
            return lambda: registry._release(unit)

        super().__init__(owner.fiber, install, "sessionProjections.register()")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._handoff()
            return
        self._projection_ready = loop.create_future()
        # Observe disposal during the asynchronous handoff before exposing a reader.
        loop.call_soon(self._handoff)

    def _handoff(self) -> None:
        try:
            self._projection_owner.fiber.assert_active()
        except Exception as error:
            self._projection_failure = error
            _watch_rejection(lambda: self.observed, error, self._projection_owner.fiber)
        ready = self._projection_ready
        if ready is not None and not ready.done():
            ready.set_result(None)

    async def _projection_wait(self) -> _Registration[S]:
        ready = self._projection_ready
        if ready is not None and not ready.done():
            await asyncio.shield(ready)
        if self._projection_failure is not None:
            raise self._projection_failure
        return self

    def __await__(self) -> Generator[Any, None, _Registration[S]]:
        self.observed = True
        return self._projection_wait().__await__()

    def state_of(self, session: Session) -> S:
        unit = self._projection_unit
        assert unit is not None
        state: S = self._projection_registry._cell_for(unit, session).state
        return state

    def reader_for(self, session: Session) -> Callable[[], S]:
        return lambda: self.state_of(session)

    async def dispose(self) -> None:
        result = self()
        if result is not None:
            await result


class _FailedRegistration(Generic[S]):
    """A registration that failed before its effect existed: awaiting it raises; calling it
    is a harmless no-op disposer."""

    def __init__(self, error: BaseException, owner: Fiber) -> None:
        self.observed = False
        self._error = error
        _watch_rejection(lambda: self.observed, error, owner)

    def __call__(self) -> None:
        return None

    async def _fail(self) -> ProjectionHandle[S]:
        raise self._error

    def __await__(self) -> Generator[Any, None, ProjectionHandle[S]]:
        self.observed = True
        return self._fail().__await__()


class SessionProjectionRegistry(Service):
    """``ctx.sessionProjections``: the projection unit table and its drive.

    The service subscribes to ``session/event`` once; every committed event passes every
    registered unit's ``apply`` (eager drive). A changed state computes the next client
    view; the change feed is notified only when its raw result changes by identity. Cells
    build lazily: a unit registered after events flowed, or a session older than the
    registry, folds ``init`` over the in-memory log on first touch. Registrants sharing a
    key share one unit and are counted; the last release removes the unit from live
    drives, checkpoints and client snapshots, while retained handles still fold new
    events on demand using the original unit.
    """

    name = "sessionProjections"

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, config)
        self._registrations: dict[str, _Unit] = {}
        self._summary_key_set: AbstractSet[str] | None = None
        """Stable summary key set; dropped whenever the key table changes."""
        self._listeners: dict[ProjectionChangeListener, None] = {}
        self._deprecated_state_of_keys: set[str] = set()
        """Keys whose deprecated registry-level read has already been reported."""
        ctx.on("session/created", self._on_created)
        ctx.on("session/event", self._drive)

    def caller(self, ctx: Context | None = None) -> Context:
        """The context a registry operation acts for: upstream's ``this.ctx`` on a service
        call. ``ctx`` when given, else ``current_context()``, else the registry's own
        context (the same rule as ``SessionStore.caller``)."""
        candidate = ctx if ctx is not None else current_context()
        if candidate is None or candidate.root is not self.ctx.root:
            return self.ctx
        return candidate

    def _on_created(self, session: Session) -> None:
        if session.seq != 0:
            return
        for unit in tuple(self._registrations.values()):
            if session in unit.cells:
                continue
            definition = unit.definition
            unit.cells[session] = _Cell(
                definition.init(session.header, session.inherited_event_count), -1
            )

    # --- registration ---------------------------------------------------------------

    def register(
        self, definition: ProjectionDefinition[S], *, ctx: Context | None = None
    ) -> ProjectionRegistration[S]:
        """Register one domain's unit on the calling context's fiber (see :meth:`caller`).

        Fiber or handle disposal releases one reference; the final release removes live
        participation, while retained handles remain readable. Await the result for the
        ready handle before publishing consumers. Calling the result releases the
        registration (deprecated; use ``handle.dispose()``).
        """
        owner = self.caller(ctx)
        wire = definition.wire
        erased = _Erased(
            key=definition.key,
            state_schema=definition.state_schema,
            init=definition.init,
            apply=definition.apply,
            view_schema=None if wire is None else wire.view_schema,
            view=None if wire is None else wire.view,
            summary=None if wire is None else wire.summary is not False,
            state_version=definition.state_version,
        )
        try:
            version = definition.state_version
            if not is_safe_integer(version) or version < 0:
                raise ValueError(
                    f"session projection {_quote(definition.key)} stateVersion must be a "
                    f"non-negative integer, got {_js_string(version)}"
                )
            return _Registration(self, owner, erased)
        except Exception as error:
            return _FailedRegistration(error, owner.fiber)

    def _acquire(self, definition: _Erased) -> _Unit:
        """Effect setup: install the unit, or join the registrants sharing its key."""
        key = definition.key
        existing = self._registrations.get(key)
        if existing is None:
            unit = _Unit(definition, refs=1)
            self._registrations[key] = unit
            self._summary_key_set = None
            return unit
        if existing.definition.state_version != definition.state_version:
            raise ValueError(
                f"session projection key {_quote(key)} is already registered at stateVersion "
                f"{_js_string(existing.definition.state_version)}; refusing to share it with "
                f"stateVersion {_js_string(definition.state_version)}"
            )
        if existing.definition.summary is not definition.summary:
            raise ValueError(
                f"session projection key {_quote(key)} is already registered with "
                f"summary={_js_string(_or_missing(existing.definition.summary))}; refusing to "
                f"share it with summary={_js_string(_or_missing(definition.summary))}"
            )
        existing.refs += 1
        return existing

    def _release(self, unit: _Unit) -> None:
        """Effect cleanup: release one reference; the last one out removes the key."""
        unit.refs -= 1
        if unit.refs == 0:
            del self._registrations[unit.definition.key]
            self._summary_key_set = None

    # --- reads ----------------------------------------------------------------------

    def state_of(self, session: Session, key: str) -> object:
        """Read a registered unit after materializing every registered unit at the Session
        cursor; unrelated wire views are not produced. ``None`` when not registered.

        Deprecated: use an awaited registration's ``handle.state_of(session)``. The first
        read of each key logs one deprecation warning per registry.
        """
        if key not in self._deprecated_state_of_keys:
            self._deprecated_state_of_keys.add(key)
            logger.warning(
                "sessionProjections.stateOf(session, %s) is deprecated; read "
                "handle.stateOf(session) from an awaited registration, or use the owning "
                "domain service",
                _quote(key),
            )
        unit = self._registrations.get(key)
        if unit is None:
            return None
        self._materialize_cells(session)
        return self._cell_for(unit, session).state

    def on_changed(
        self, listener: ProjectionChangeListener, *, ctx: Context | None = None
    ) -> Callable[[], None]:
        """Subscribe to the change feed as an effect on the caller's fiber (see
        :meth:`caller`). Returns the disposer that unsubscribes."""

        def setup() -> Callable[[], None]:
            self._listeners[listener] = None

            def remove() -> None:
                self._listeners.pop(listener, None)

            return remove

        dispose = self.caller(ctx).effect(setup, "sessionProjections.onChanged()")

        def stop() -> None:
            dispose()

        return stop

    def snapshot(
        self, session: Session, keys: ProjectionKeySelection | None = None
    ) -> ProjectionSnapshot:
        """One consistent cut over every registered client-visible unit for one session.

        Missing cells fold lazily over the in-memory log; state materialization stays
        complete when ``keys`` selects outputs. Each value passes its ``view_schema``.
        """
        values: dict[str, Any] = {}
        selected = _selected_keys(keys)
        self._materialize_cells(session)
        for unit in self._units():
            definition = unit.definition
            if not definition.has_wire:
                continue
            if selected is not None and definition.key not in selected:
                continue
            cell = self._cell_for(unit, session)
            values[definition.key] = definition.wire_value(cell.state)
        return ProjectionSnapshot(_cursor_before(session.seq), values)

    def cached_snapshot(
        self, session: Session, keys: ProjectionKeySelection | None = None
    ) -> ProjectionSnapshot | None:
        """Read only already-materialized client-visible cells without folding history.

        Values may trail the live Session (hints, not a baseline); missing cells are
        omitted. Returns the lowest common cached cut, or ``None`` when no wire cell exists.
        """
        values: dict[str, Any] = {}
        as_of_seq: SessionSeqCursor | None = None
        selected = _selected_keys(keys)
        for unit in self._units():
            definition = unit.definition
            if not definition.has_wire:
                continue
            if selected is not None and definition.key not in selected:
                continue
            cell = unit.cells.get(session)
            if cell is None:
                continue
            values[definition.key] = definition.wire_value(cell.state)
            if as_of_seq is None or cell.observed_seq < as_of_seq:
                as_of_seq = cell.observed_seq
        return None if as_of_seq is None else ProjectionSnapshot(as_of_seq, values)

    def summary_keys(self) -> AbstractSet[str]:
        """The keys a Session summary carries: every registered client-visible key whose
        unit does not declare ``summary=False``.

        The same read-only set is returned until a registration or its disposal changes
        the key table; pass it to ``cached_snapshot`` or ``view_checkpoint`` as is.
        """
        if self._summary_key_set is None:
            self._summary_key_set = self._derive_summary_keys()
        return self._summary_key_set

    def _derive_summary_keys(self) -> AbstractSet[str]:
        keys: dict[str, None] = {}
        for unit in self._registrations.values():
            if not unit.definition.has_wire or not unit.definition.summary:
                continue
            keys[unit.definition.key] = None
        return keys.keys()

    def checkpoint(self, session: Session) -> ProjectionCheckpoint:
        """State-level checkpoint of every persisted unit for one session.

        Missing cells fold lazily. Every ``val`` is a detached plain copy, never the live
        cell state.
        """
        rows: ProjectionCheckpoint = {}
        for unit in self._units():
            cell = self._cell_for(unit, session)
            rows[unit.definition.key] = {
                "ver": unit.definition.state_version,
                "seq": cell.observed_seq,
                "val": thaw_json(cell.state),
            }
        return rows

    def restore_floor(
        self, checkpoint: Mapping[str, ProjectionCheckpointRow]
    ) -> SessionLogOffset | None:
        """The stored offset a :meth:`restore` tail read over ``checkpoint`` must start at.

        One event below the lowest usable watermark (a row is usable when its ``ver``
        matches); an absent or mismatched row pulls the floor to ``0``. The one-below
        anchor lets :meth:`restore` detect a log that shrank below a row's watermark.
        ``None`` when no unit is registered.
        """
        floor: int | None = None
        for unit in self._registrations.values():
            definition = unit.definition
            row = checkpoint.get(definition.key, _MISSING)
            need = (
                max(row["seq"] + 1, 0)
                if row is not _MISSING
                and _same_version(row.get("ver", _MISSING), definition.state_version)
                else 0
            )
            floor = need if floor is None else min(floor, need)
        return None if floor is None else SessionLogOffset(max(floor - 1, 0))

    def view_checkpoint(
        self,
        checkpoint: Mapping[str, ProjectionCheckpointRow],
        keys: ProjectionKeySelection | None = None,
    ) -> dict[str, Any]:
        """View a checkpoint's rows without any log read.

        Every registered client-visible unit whose row's ``ver`` matches serves the
        validated view of its validated stored state; mismatched, malformed or absent rows
        leave their key absent.
        """
        values: dict[str, Any] = {}
        selected = _selected_keys(keys)
        for unit in self._units():
            definition = unit.definition
            if not definition.has_wire:
                continue
            if selected is not None and definition.key not in selected:
                continue
            row = checkpoint.get(definition.key, _MISSING)
            if row is _MISSING or not _same_version(
                row.get("ver", _MISSING), definition.state_version
            ):
                continue
            try:
                state = definition.state_schema.validate_python(row["val"])
            except Exception:
                continue
            values[definition.key] = definition.wire_value(state)
        return values

    def restore(
        self,
        checkpoint: Mapping[str, ProjectionCheckpointRow],
        events: Sequence[SessionEvent],
        base_seq: SessionLogOffset,
        header: SessionHeader,
        inherited_event_count: SessionLogOffset,
    ) -> ProjectionRestoreResult:
        """Cold read: fold every persisted unit over a stored log suffix.

        Call with the stored events at or past ``restore_floor(checkpoint)`` and that floor
        as ``base_seq``. A row is usable iff its ``ver`` matches, it does not predate
        ``base_seq`` and it does not claim events past the supplied end; an unusable row
        refolds from ``init``, which is only sound over the full log, so an unusable row
        with ``base_seq > 0`` raises (re-read from seq 0).
        """
        before_base = _cursor_before(base_seq)
        end_seq: SessionSeqCursor = events[-1]["seq"] if events else before_base
        values: dict[str, Any] = {}
        refreshed: ProjectionCheckpoint = {}
        for unit in self._units():
            definition = unit.definition
            row = checkpoint.get(definition.key, _MISSING)
            usable = (
                row is not _MISSING
                and _same_version(row.get("ver", _MISSING), definition.state_version)
                and row["seq"] >= before_base
                and row["seq"] <= end_seq
            )
            if not usable and base_seq > 0:
                raise ValueError(
                    f"session projection {_quote(definition.key)} cannot restore from seq "
                    f"{base_seq}: its checkpoint row is missing, version-mismatched, or beyond "
                    "the supplied log end; re-read from seq 0"
                )
            state = (
                definition.state_schema.validate_python(row["val"])
                if usable
                else definition.init(header, inherited_event_count)
            )
            start = row["seq"] if usable else before_base
            for index in range(start - base_seq + 1, len(events)):
                event = events[index] if index >= 0 else None
                expected = base_seq + index
                if event is None or event["seq"] != expected:
                    raise ValueError(
                        f"session projection {_quote(definition.key)} cannot restore across "
                        f"missing seq {expected}"
                    )
                state = definition.apply(state, event)
            if definition.has_wire:
                values[definition.key] = definition.wire_value(state)
            refreshed[definition.key] = {
                "ver": definition.state_version,
                "seq": end_seq,
                "val": state,
            }
        return ProjectionRestoreResult(ProjectionSnapshot(end_seq, values), refreshed)

    def hydrate(
        self,
        session: Session,
        checkpoint: Mapping[str, ProjectionCheckpointRow],
        events: Sequence[SessionEvent],
        base_seq: SessionLogOffset,
    ) -> ProjectionSnapshot:
        """Restore an exact cut and install its states on the supplied prepared Session.

        A later publication reuses these cells; live reads and the event drive advance any
        constructor-owned suffix exactly once. Returns all projection values at the cut.
        """
        end_seq: SessionSeqCursor = events[-1]["seq"] if events else _cursor_before(base_seq)
        complete = True
        for unit in self._registrations.values():
            current = unit.cells.get(session)
            if current is None or current.observed_seq != end_seq:
                complete = False
                break
        if complete:
            values: dict[str, Any] = {}
            for unit in self._units():
                if not unit.definition.has_wire:
                    continue
                cell = unit.cells[session]
                values[unit.definition.key] = unit.definition.wire_value(cell.state)
            return ProjectionSnapshot(end_seq, values)
        restored = self.restore(
            checkpoint, events, base_seq, session.header, session.inherited_event_count
        )
        for unit in self._units():
            row = restored.checkpoint.get(unit.definition.key)
            if row is None:
                continue
            current = unit.cells.get(session)
            if current is not None and current.observed_seq > row["seq"]:
                continue
            unit.cells[session] = _Cell(row["val"], row["seq"])
        return restored.snapshot

    # --- drive ----------------------------------------------------------------------

    def _units(self) -> Iterable[_Unit]:
        """Iterate the live table; a unit removed mid-iteration is skipped (as upstream's
        live ``Map`` iteration skips deleted entries)."""
        for key, unit in tuple(self._registrations.items()):
            if self._registrations.get(key) is unit:
                yield unit

    def _materialize_cells(self, session: Session) -> None:
        for unit in self._units():
            self._cell_for(unit, session)

    @staticmethod
    def _build_cell(
        definition: _Erased,
        header: SessionHeader,
        inherited_event_count: SessionLogOffset,
        events: Sequence[SessionEvent],
    ) -> _Cell:
        """Fold one unit from ``init`` over ``events``."""
        state = definition.init(header, inherited_event_count)
        for event in events:
            state = definition.apply(state, event)
        return _Cell(state, events[-1]["seq"] if events else -1)

    def _cell_for(self, unit: _Unit, session: Session) -> _Cell:
        """Read (or lazily build, folding the full in-memory log) one unit's cell."""
        cell = unit.cells.get(session)
        if cell is None:
            cell = self._build_cell(
                unit.definition,
                session.header,
                session.inherited_event_count,
                session.snapshot_events(),
            )
            unit.cells[session] = cell
        else:
            self._advance_cell(unit.definition, cell, session, _cursor_before(session.seq))
        return cell

    @staticmethod
    def _advance_cell(
        definition: _Erased, cell: _Cell, session: Session, through_seq: SessionSeqCursor
    ) -> None:
        """Advance one existing cell through a contiguous Session prefix."""
        if cell.observed_seq >= through_seq:
            return
        for seq in range(cell.observed_seq + 1, through_seq + 1):
            event = session.event_at(seq)
            if event is None or event["seq"] != seq:
                raise ValueError(
                    f"session projection {_quote(definition.key)} cannot advance across "
                    f"missing seq {seq}"
                )
            following = definition.apply(cell.state, event)
            if not _same_value(following, cell.state):
                cell.views[0] = cell.views[1]
                cell.views[1] = _NO_VIEW
            cell.state = following
            cell.observed_seq = SessionSeq(seq)

    def _drive(self, session: Session, event: SessionEvent) -> None:
        """Pass one committed event through every unit; notify on changed raw views."""
        seq: SessionSeq = event["seq"]
        for unit in self._units():
            definition = unit.definition
            cell = unit.cells.get(session)
            if cell is not None and cell.observed_seq >= seq:
                continue
            if cell is None:
                # Late build mid-stream: fold history before this event (seq = log index,
                # so the prefix slice is exact), then take the normal gate.
                cell = self._build_cell(
                    definition,
                    session.header,
                    session.inherited_event_count,
                    session.snapshot_events(0, seq),
                )
                unit.cells[session] = cell
            else:
                self._advance_cell(definition, cell, session, seq - 1)
            previous = cell.state
            following = definition.apply(previous, event)
            changed = not _same_value(following, previous)
            cell.state = following
            cell.observed_seq = seq
            if changed and definition.view is not None:
                views = cell.views
                views[0] = views[1]
                if self._listeners:
                    views[1] = definition.view(following)
                    if not _same_value(views[0], views[1]):
                        assert definition.view_schema is not None
                        value = definition.view_schema.validate_python(views[1])
                        for listener in tuple(self._listeners):
                            if listener in self._listeners:
                                listener(session, definition.key, value, seq)
                else:
                    views[1] = _NO_VIEW
            # An unchanged state keeps its current view as the valid comparison value for
            # the next state change.


def _or_missing(value: bool | None) -> object:
    return _MISSING if value is None else value
