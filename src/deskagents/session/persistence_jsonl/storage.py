"""JSONL session handles, live write batching and in-process ownership tracking.

Upstream: packages/session/session-persistence-jsonl/src/storage.ts
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Protocol

from deepseek_cordis import Context

from deskagents.core.session import (
    Session,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
)
from deskagents.session.persistence import (
    SessionAccess,
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionHandleClosedError,
    SessionHandleReadResult,
    SessionPersistenceNotFoundError,
    SessionPersistenceRevision,
    SessionReadOnlyError,
    assert_contiguous,
    materialize_append_batch,
)
from deskagents.util.values import MAX_SAFE_INTEGER, is_safe_integer

if TYPE_CHECKING:
    from .lease import SessionWriteLease

logger = logging.getLogger(__name__)

LIVE_WRITE_BATCH_MAX_DELAY_S = 0.2
"""Live events are batched for at most this long before a background write (200 ms)."""


class JsonlHandleStorage(Protocol):
    """What a handle needs from its backend."""

    async def persist_batch(
        self,
        header: SessionHeader,
        events: Sequence[SessionEvent],
        is_materialized: bool,
        inherited_event_count: SessionLogOffset,
    ) -> None: ...
    async def persist_header(
        self, header: SessionHeader, inherited_event_count: SessionLogOffset
    ) -> None: ...
    async def truncate_torn_tail(self, header: SessionHeader, truncate_to: int) -> None: ...
    async def resolve_current_log(self, session_id: SessionId) -> str | None: ...
    async def read_stored_log(
        self, path: str, expected_id: SessionId
    ) -> SessionHandleReadResult: ...
    def has_pending_session(self, session_id: SessionId) -> bool: ...
    async def acquire_write_lease(self, header: SessionHeader) -> SessionWriteLease: ...
    def release_handle(self, handle: JsonlSessionHandle, materialized: bool) -> None: ...


@dataclass
class StorageHandleState:
    """Mutable storage state one handle shares with its backend."""

    cursor: int
    materialized: bool
    inherited_event_count: SessionLogOffset
    torn_truncate_to: int | None = None
    recovered_tail: list[SessionEvent] | None = None
    primed: SessionHandleReadResult | None = None


def _check_count(value: object, name: str) -> int:
    if not is_safe_integer(value) or isinstance(value, bool) or value < 0:  # type: ignore[operator]
        raise TypeError(f"read {name} must be a non-negative safe integer, got {value!r}")
    return int(value)  # type: ignore[call-overload,no-any-return]


def _slice(source: SessionHandleReadResult, offset: int, length: int) -> SessionHandleReadResult:
    return SessionHandleReadResult(
        source.event_state, list(source.events[offset : offset + length])
    )


class JsonlSessionHandle:
    """One open channel onto a stored JSONL session (see ``SessionHandle``)."""

    def __init__(
        self,
        storage: JsonlHandleStorage,
        session_id: SessionId,
        header: SessionHeader,
        access: SessionAccess,
        state: StorageHandleState,
        lease: SessionWriteLease | None = None,
    ) -> None:
        self._storage = storage
        self._id = session_id
        self._header = header
        self._access: SessionAccess = access
        self._state = state
        self._lease = lease
        self._lock = asyncio.Lock()
        self._closing: asyncio.Future[None] | None = None
        self._observed_length = 0
        self._buffered: list[SessionEvent] = []
        self._batch_timer: asyncio.TimerHandle | None = None
        self._drain_paused = False
        self._draining: asyncio.Future[None] | None = None

    @property
    def id(self) -> SessionId:
        return self._id

    @property
    def header(self) -> SessionHeader:
        return self._header

    @property
    def access(self) -> SessionAccess:
        return self._access

    @property
    def inherited_event_count(self) -> SessionLogOffset:
        return self._state.inherited_event_count

    async def read(self, offset: int = 0, length: int | None = None) -> SessionHandleReadResult:
        self._assert_open("read")
        offset = _check_count(offset, "offset")
        length = MAX_SAFE_INTEGER if length is None else _check_count(length, "length")
        primed = self._state.primed
        if primed is not None:
            if self._access == "write":
                return self._read_primed(primed, offset, length)
            current = await self._storage.resolve_current_log(self._id)
            if current is None:
                return self._read_primed(primed, offset, length)
            self._state.primed = None
            return await self._read_current(current, offset, length)
        if self._access == "write" and not self._state.materialized:
            return SessionHandleReadResult("detached", [])
        current = await self._storage.resolve_current_log(self._id)
        if current is not None:
            return await self._read_current(current, offset, length)
        if self._storage.has_pending_session(self._id):
            return SessionHandleReadResult("detached", [])
        raise SessionPersistenceNotFoundError(self._id)

    def _read_primed(
        self, source: SessionHandleReadResult, offset: int, length: int
    ) -> SessionHandleReadResult:
        self._observed_length = max(self._observed_length, len(source.events))
        return _slice(source, offset, length)

    async def _read_current(self, path: str, offset: int, length: int) -> SessionHandleReadResult:
        source = await self._storage.read_stored_log(path, self._id)
        if len(source.events) < self._observed_length:
            raise ValueError(
                f'session "{self._id}": stored log shrank below a previously observed prefix '
                f"({len(source.events)} < {self._observed_length})"
            )
        self._observed_length = len(source.events)
        return _slice(source, offset, length)

    async def append(self, events: Sequence[SessionEvent]) -> None:
        self._assert_open("append")
        batch = materialize_append_batch(events)
        await self._run("append", lambda: self._persist_contiguous(batch))

    async def flush(self) -> None:
        async def op() -> None:
            if self._access != "write":
                raise SessionReadOnlyError(self._id, "flush")
            if self._state.materialized:
                return  # appends are durable on return
            await self._ensure_lease()
            await self._storage.persist_header(self._header, self._state.inherited_event_count)
            self._state.materialized = True

        await self._run("flush", op)

    def close(self) -> asyncio.Future[None]:
        """Release the handle; idempotent, and not cancelled by cancelling a waiter."""
        if self._closing is None:
            self._closing = asyncio.ensure_future(self._close())
        return asyncio.shield(self._closing)

    async def _close(self) -> None:
        drain_failure: BaseException | None = None
        while True:
            try:
                await self.drain_live()
            except Exception as error:
                drain_failure = error
                break
            async with self._lock:
                pass
            if not self._buffered:
                break
        async with self._lock:
            pass
        failures: list[Exception] = []
        if isinstance(drain_failure, Exception):
            failures.append(drain_failure)
        if self._lease is not None:
            try:
                await self._lease.release()
            except Exception as error:
                failures.append(error)
        self._storage.release_handle(self, self._state.materialized)
        if len(failures) > 1:
            raise ExceptionGroup(
                f'session "{self._id}": close failed to drain and to release its write lock',
                failures,
            )
        if failures:
            raise failures[0]

    async def __aenter__(self) -> JsonlSessionHandle:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def enqueue_live(
        self, event: SessionEvent, report_background_failure: Callable[[BaseException], None]
    ) -> None:
        """Buffer one live ``session/event`` for a batched background write."""
        self._buffered.append(event)  # events are frozen; no copy is needed
        if self._batch_timer is not None or self._drain_paused:
            return
        loop = asyncio.get_running_loop()

        def fire() -> None:
            self._batch_timer = None
            task = asyncio.ensure_future(self.drain_live())

            def done(finished: asyncio.Future[None]) -> None:
                if finished.cancelled():
                    return
                error = finished.exception()
                if error is not None:
                    report_background_failure(error)

            task.add_done_callback(done)

        self._batch_timer = loop.call_later(LIVE_WRITE_BATCH_MAX_DELAY_S, fire)

    def drain_live(self) -> asyncio.Future[None]:
        """Write every buffered live event now; concurrent callers share one drain."""
        if self._draining is None:
            self._draining = asyncio.ensure_future(self._drain_buffered())

            def clear(_: asyncio.Future[None]) -> None:
                self._draining = None

            self._draining.add_done_callback(clear)
        return asyncio.shield(self._draining)

    async def _drain_buffered(self) -> None:
        if self._batch_timer is not None:
            self._batch_timer.cancel()
            self._batch_timer = None
        self._drain_paused = False
        while self._buffered:
            async with self._lock:
                batch = self._buffered[:]
                del self._buffered[:]
                try:
                    await self._persist_contiguous(materialize_append_batch(batch))
                except BaseException:
                    self._buffered[:0] = batch
                    self._drain_paused = True
                    raise

    async def _persist_contiguous(self, batch: Sequence[SessionEvent]) -> None:
        if self._access != "write":
            raise SessionReadOnlyError(self._id, "append")
        if not batch:
            return
        await self._ensure_lease()
        assert_contiguous(self._id, batch, self._state.cursor)
        if self._state.torn_truncate_to is not None:
            await self._storage.truncate_torn_tail(self._header, self._state.torn_truncate_to)
            self._state.torn_truncate_to = None
        if self._state.recovered_tail is not None:
            if self._state.recovered_tail:
                await self._storage.persist_batch(
                    self._header,
                    self._state.recovered_tail,
                    self._state.materialized,
                    self._state.inherited_event_count,
                )
            self._state.recovered_tail = None
        await self._storage.persist_batch(
            self._header, batch, self._state.materialized, self._state.inherited_event_count
        )
        self._state.materialized = True
        self._state.cursor += len(batch)
        self._state.primed = None
        self._observed_length = self._state.cursor

    async def _ensure_lease(self) -> None:
        if self._lease is None:
            self._lease = await self._storage.acquire_write_lease(self._header)

    async def _run(self, operation: str, op: Callable[[], object]) -> None:
        self._assert_open(operation)
        async with self._lock:
            self._assert_open(operation)
            result = op()
            if asyncio.iscoroutine(result):
                await result

    def _assert_open(self, operation: str) -> None:
        if self._closing is not None:
            raise SessionHandleClosedError(self._id, operation)


@dataclass(frozen=True)
class PendingSession:
    """A created session not yet materialized on disk."""

    header: SessionHeader
    revision: SessionPersistenceRevision
    inherited_event_count: SessionLogOffset


@dataclass
class JsonlBackendTracker:
    """In-process write ownership, pending sessions and live ``session/*`` routing."""

    name: str
    open_handles: set[JsonlSessionHandle] = field(default_factory=set)
    _writers: dict[SessionId, JsonlSessionHandle | None] = field(default_factory=dict)
    _pending: dict[SessionId, PendingSession] = field(default_factory=dict)
    _counter: int = 0

    def register_created(
        self, header: SessionHeader, inherited_event_count: SessionLogOffset
    ) -> None:
        sid = SessionId(header["id"])
        if sid in self._writers:
            raise SessionAlreadyExistsError(sid)
        self._writers[sid] = None
        self._counter += 1
        self._pending[sid] = PendingSession(
            header,
            SessionPersistenceRevision(f"memory:{self.name}:{self._counter}"),
            inherited_event_count,
        )

    def claim_write(self, session_id: SessionId) -> None:
        if session_id in self._writers:
            raise SessionAlreadyOwnedError(session_id)
        self._writers[session_id] = None

    def release_claim(self, session_id: SessionId) -> None:
        self._writers.pop(session_id, None)

    def pending_of(self, session_id: SessionId) -> PendingSession | None:
        return self._pending.get(session_id)

    def has_pending(self, session_id: SessionId) -> bool:
        return session_id in self._pending

    def pending_entries(self) -> Iterator[tuple[SessionId, PendingSession]]:
        return iter(list(self._pending.items()))

    def materialized(self, session_id: SessionId) -> None:
        self._pending.pop(session_id, None)

    def adopt(self, handle: JsonlSessionHandle) -> JsonlSessionHandle:
        self.open_handles.add(handle)
        if handle.access == "write":
            self._writers[handle.id] = handle
        return handle

    def release(self, handle: JsonlSessionHandle, materialized: bool) -> None:
        self.open_handles.discard(handle)
        if handle.access != "write":
            return
        self._writers.pop(handle.id, None)
        if not materialized:
            self._pending.pop(handle.id, None)

    async def flush_all(self) -> None:
        errors: list[Exception] = []
        for writer in list(self._writers.values()):
            if writer is None:
                continue  # a claim mid-construction routes nothing yet
            try:
                await writer.drain_live()
                await writer.flush()
            except SessionHandleClosedError:
                continue
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup(f"{self.name} flush failed", errors)

    def install(self, ctx: Context) -> Callable[[], object]:
        """Route live session events to their writers and close open handles on unload."""
        writers, open_handles, name = self._writers, self.open_handles, self.name

        def on_event(session: Session, event: SessionEvent) -> None:
            writer = writers.get(session.id)
            if writer is None:
                return

            def report(error: BaseException) -> None:
                logger.warning(
                    'session-persistence: background write for session "%s" failed '
                    "(buffered events retained): %s",
                    session.id,
                    error,
                )

            writer.enqueue_live(event, report)

        def on_flush(session: Session) -> object:
            writer = writers.get(session.id)
            if writer is None:
                return None

            async def flush() -> None:
                await writer.drain_live()
                await writer.flush()

            return flush()

        def on_disposed(session: Session) -> None:
            writer = writers.get(session.id)
            if writer is None:
                return
            closing = writer.close()

            def report(done: asyncio.Future[None]) -> None:
                if not done.cancelled() and done.exception() is not None:
                    logger.warning(
                        'session-persistence: final drain for session "%s" failed: %s',
                        session.id,
                        done.exception(),
                    )

            closing.add_done_callback(report)

        def setup() -> Iterator[Callable[[], object]]:
            async def close_all() -> None:
                errors: list[Exception] = []
                for handle in list(open_handles):
                    try:
                        await handle.close()
                    except Exception as error:
                        errors.append(error)
                if errors:
                    raise ExceptionGroup(f"{name} dispose failed", errors)

            yield close_all
            yield ctx.on("session/event", on_event)
            yield ctx.on("session/flush", on_flush)
            yield ctx.on("session/disposed", on_disposed)

        return ctx.effect(setup, f"{self.name} live routing and open handles")
