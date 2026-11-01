"""Reusable live-write-path contract for any ``SessionPersistence`` backend.

Upstream: packages/session/session-persistence/tests/live-write-contract.ts
(``runLiveWritePathContract``) at ce7073d.

Published live events route by session id into the active write handle,
``session/flush`` is the durability and error-observation barrier, ``session/disposed``
drains and closes, and ``close()`` itself drains the routed buffer — including through
backend teardown. Each check takes a factory producing one fresh context with the
``sessions`` store and the backend mounted, plus the backend's live batching window in
seconds.

Like upstream, failures are injected at the service's storage primitive: the backend
service must route every live write through an awaitable ``persist_batch(header, ...)``
method (upstream ``persistBatch``), which the checks temporarily replace on the
instance.

Python differences:

- No fake timers: the checks wait real time, scaled from ``batch_delay`` (the
  "one tick short of the window" probe reads at half the window).
- ``ctx.logger.warn`` spies become a handler on the root :mod:`logging` logger.
- ``AggregateError`` becomes :class:`ExceptionGroup`.
- Upstream's "close normalizes a non-Error drain failure" has no counterpart: Python can
  only raise exceptions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from deepseek_cordis import Context

from deskagents.core.session import (
    PluginRecord,
    Session,
    SessionEvent,
    SessionId,
    SessionSeq,
    SessionStore,
    append_plugin_record,
    plugin_record_of,
)
from deskagents.session.persistence import SessionPersistence

from .persistence_contract import ContractResult, _expect_equal, _expect_raises, run_cases

__all__ = [
    "LIVE_WRITE_PATH_CONTRACT_SUITE",
    "LiveWriteBackend",
    "LiveWriteBackendFactory",
    "LiveWriteCheck",
    "live_write_path_contract_cases",
    "run_live_write_path_contract",
]

LIVE_WRITE_PATH_CONTRACT_SUITE = "live session write path"
"""Upstream's describe title prefix; the full title is ``f"{SUITE}: {backend name}"``."""

_POLL_TIMEOUT_S = 5.0


@dataclass
class LiveWriteBackend:
    """One mounted backend under a session store, plus same-storage remount support."""

    ctx: Context
    """Context with ``sessions`` and ``sessionPersistence`` mounted."""
    remount: Callable[[], Awaitable[Context]]
    """Mount a FRESH context over the SAME storage, as after a process restart."""


LiveWriteBackendFactory = Callable[[], Awaitable[LiveWriteBackend]]
"""Produces one fresh mounted backend per call; each check disposes what it made."""

LiveWriteCheck = Callable[[LiveWriteBackendFactory, float], Awaitable[None]]
"""One contract case: ``check(make, batch_delay_seconds)``."""


# --- helpers ---------------------------------------------------------------------------


def _sessions(ctx: Context) -> SessionStore:
    store = ctx.get("sessions")
    if not isinstance(store, SessionStore):
        raise AssertionError(f"ctx.sessions: expected a SessionStore, got {store!r}")
    return store


def _persistence(ctx: Context) -> SessionPersistence:
    service = ctx.get("sessionPersistence")
    if not isinstance(service, SessionPersistence):
        raise AssertionError(f"ctx.sessionPersistence: expected a backend, got {service!r}")
    return service


async def _read_all(persistence: SessionPersistence, session_id: SessionId) -> list[SessionEvent]:
    reader = await persistence.open(session_id, "read")
    try:
        return list((await reader.read()).events)
    finally:
        await reader.close()


async def _seqs(persistence: SessionPersistence, session_id: SessionId) -> list[int]:
    return [event["seq"] for event in await _read_all(persistence, session_id)]


async def _eventually(probe: Callable[[], Awaitable[object]], expected: object, what: str) -> None:
    """``expect.poll``: retry ``probe`` until it equals ``expected`` or time runs out."""
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while True:
        actual = await probe()
        if actual == expected:
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"{what}: expected {expected!r}, last saw {actual!r}")
        await asyncio.sleep(0.01)


async def _wait_until(condition: Callable[[], bool], what: str) -> None:
    """``vi.waitFor`` over a synchronous condition."""
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


class _Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def text(self) -> str:
        return "\n".join(self.messages)


@contextmanager
def _capture_warnings() -> Iterator[_Warnings]:
    handler = _Warnings()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)


class _PersistBatchSpy:
    """``vi.spyOn(service, 'persistBatch')``: replace the instance method, count calls."""

    def __init__(self, service: object) -> None:
        if not callable(getattr(service, "persist_batch", None)):
            raise AssertionError(
                "live-write contract: the backend service must expose persist_batch()"
            )
        self._service = service
        self.original: Callable[..., Awaitable[None]] = service.persist_batch  # type: ignore[attr-defined]
        self.calls = 0
        self._impl: Callable[..., Awaitable[None]] = self.original

        async def spy(*args: Any, **kwargs: Any) -> None:
            self.calls += 1
            await self._impl(*args, **kwargs)

        vars(service)["persist_batch"] = spy

    def reject(self, failure: BaseException) -> None:
        async def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise failure

        self._impl = refuse

    def implement(self, impl: Callable[..., Awaitable[None]]) -> None:
        self._impl = impl

    def restore(self) -> None:
        vars(self._service).pop("persist_batch", None)


@contextmanager
def _spy_persist_batch(service: object) -> Iterator[_PersistBatchSpy]:
    spy = _PersistBatchSpy(service)
    try:
        yield spy
    finally:
        spy.restore()


async def _owned_session(
    ctx: Context, session_id: str
) -> tuple[Session, Callable[[], Awaitable[None]]]:
    """Create a session inside a child plugin, so disposing that fiber disposes it."""
    created: list[Session] = []

    def owner(inner: Context, _config: object) -> None:
        created.append(_sessions(inner).create(SessionId(session_id)))

    fiber = ctx.inject(["sessions"], owner)
    await fiber
    if not created:
        raise AssertionError("session was not created")
    return created[0], fiber.dispose


# --- cases ------------------------------------------------------------------------------

_CASES: list[tuple[str, LiveWriteCheck]] = []


def _case(title: str) -> Callable[[LiveWriteCheck], LiveWriteCheck]:
    def register(check: LiveWriteCheck) -> LiveWriteCheck:
        _CASES.append((title, check))
        return check

    return register


@_case("routes published events into the active write handle within one batching window")
async def routes_published_events(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    session = _sessions(ctx).create(SessionId("routed"))
    handle = await persistence.create(session.header)
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await asyncio.sleep(batch_delay / 2)
    # Short of the window: nothing stored yet (in-process visibility serves the
    # created-but-empty session).
    _expect_equal(await _read_all(persistence, session.id), [], "inside the window")

    async def stored() -> list[tuple[str, int]]:
        return [(e["type"], e["seq"]) for e in await _read_all(persistence, session.id)]

    # The deadline started a background write; wait for its durability.
    await _eventually(stored, [("turn/start", 0), ("turn/end", 1)], "routed events")
    await handle.close()
    await ctx.fiber.dispose()


@_case("a session without an active write handle persists nothing")
async def unrouted_session_persists_nothing(
    make: LiveWriteBackendFactory, batch_delay: float
) -> None:
    ctx = (await make()).ctx
    session = _sessions(ctx).create(SessionId("unrouted"))
    session.append("turn/start", {"turn": 1})
    await _sessions(ctx).flush(session)
    _expect_equal(await _persistence(ctx).stat(session.id), None, "stat")
    await ctx.fiber.dispose()


@_case("persists a plugin record that a restarted process reads back")
async def persists_plugin_record(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    backend = await make()
    ctx = backend.ctx
    session = _sessions(ctx).create(SessionId("plugin-record"))
    handle = await _persistence(ctx).create(session.header)
    session.append("turn/start", {"turn": 1})
    append_plugin_record(session, "plugin:persistence-test/state", {"count": 1})
    _expect_equal(await _sessions(ctx).flush(session), True, "flush participated")
    await handle.close()
    await ctx.fiber.dispose()

    restarted = await backend.remount()
    events = await _read_all(_persistence(restarted), session.id)
    _expect_equal(
        [(event["type"], event.get("ignorable")) for event in events],
        [("turn/start", None), ("plugin:persistence-test/state", True)],
        "types and ignorable flags",
    )
    record = PluginRecord(
        type="plugin:persistence-test/state",
        seq=SessionSeq(1),
        time=events[1]["time"],
        data={"count": 1},
    )
    _expect_equal([plugin_record_of(event) for event in events], [None, record], "records")
    await restarted.fiber.dispose()


@_case("session/flush drains immediately and surfaces a retained background failure")
async def session_flush_surfaces_failure(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    session = _sessions(ctx).create(SessionId("flush-surfaces"))
    handle = await persistence.create(session.header)
    failure = Exception("backend write refused")
    with _capture_warnings() as warned, _spy_persist_batch(persistence) as persist:
        # Inject at the service's storage primitive: the routed drain writes through the
        # handle's internal chain, not the public append.
        persist.reject(failure)
        session.append("turn/start", {"turn": 1})
        await asyncio.sleep(batch_delay)
        await _wait_until(
            lambda: 'background write for session "flush-surfaces" failed' in warned.text(),
            "the background-failure warning",
        )

        # The barrier retries the retained batch and raises loudly...
        raised = await _expect_raises(_sessions(ctx).flush(session), Exception, "flush")
        if raised is not failure:
            raise AssertionError(f"flush: expected the injected failure, got {raised!r}")
        # ...and once the backend recovers, the same events land exactly once.
        persist.restore()
        _expect_equal(await _sessions(ctx).flush(session), True, "recovered flush")
        _expect_equal(await _seqs(persistence, session.id), [0], "seqs")
    await handle.close()
    await ctx.fiber.dispose()


@_case("service-level flush drains every active handle and aggregates the failures")
async def service_flush_aggregates(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    healthy = _sessions(ctx).create(SessionId("flush-all-healthy"))
    failing = _sessions(ctx).create(SessionId("flush-all-failing"))
    healthy_handle = await persistence.create(healthy.header)
    failing_handle = await persistence.create(failing.header)
    failure = Exception("backend write refused")
    with _spy_persist_batch(persistence) as persist:
        original = persist.original

        async def selective(header: Any, *rest: Any) -> None:
            if header["id"] == failing.id:
                raise failure
            await original(header, *rest)

        persist.implement(selective)
        healthy.append("turn/start", {"turn": 1})
        failing.append("turn/start", {"turn": 1})
        # No batching window elapses: the service barrier itself drains both buffers
        # and reports the one failure without abandoning the sweep.
        raised = await _expect_raises(persistence.flush(), Exception, "service flush")
        if not (
            isinstance(raised, ExceptionGroup)
            and len(raised.exceptions) == 1
            and raised.exceptions[0] is failure
        ):
            raise AssertionError(f"service flush: expected one aggregated failure, got {raised!r}")
        # The healthy session flushed durably despite its neighbor's failure...
        _expect_equal(await _seqs(persistence, healthy.id), [0], "healthy seqs")
        # ...and the failed batch is retained: a recovered backend flushes it exactly once.
        persist.restore()
        await persistence.flush()
        _expect_equal(await _seqs(persistence, failing.id), [0], "failing seqs")
    await healthy_handle.close()
    await failing_handle.close()
    await ctx.fiber.dispose()


@_case("session/disposed drains buffered events and closes the handle")
async def disposed_drains_and_closes(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    session, dispose_owner = await _owned_session(ctx, "disposed-drains")
    handle = await persistence.create(session.header)
    session.append("turn/start", {"turn": 1})
    # Buffered, not yet written: disposal must drain before the close.
    await dispose_owner()

    async def seqs() -> list[int]:
        return await _seqs(persistence, SessionId("disposed-drains"))

    await _eventually(seqs, [0], "drained seqs")
    await _expect_raises(handle.append([]), Exception, "append after close", "closed handle")
    await ctx.fiber.dispose()


@_case("a failing final drain on disposal is warned, not thrown")
async def failing_final_drain_warned(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    session, dispose_owner = await _owned_session(ctx, "disposed-drain-fails")
    handle = await persistence.create(session.header)
    with _capture_warnings() as warned:

        def exploding_close() -> asyncio.Future[None]:
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            future.set_exception(Exception("drain exploded"))
            return future

        vars(handle)["close"] = exploding_close
        try:
            session.append("turn/start", {"turn": 1})
            await dispose_owner()
            await _wait_until(
                lambda: 'final drain for session "disposed-drain-fails" failed' in warned.text(),
                "the final-drain warning",
            )
        finally:
            vars(handle).pop("close", None)
    await ctx.fiber.dispose()


@_case("backend teardown drains buffered events through the close sweep")
async def teardown_drains(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    backend = await make()
    ctx = backend.ctx
    session = _sessions(ctx).create(SessionId("teardown-drains"))
    handle = await _persistence(ctx).create(session.header)
    session.append("turn/start", {"turn": 1})
    # Root disposal closes the still-open handle; close drains the buffer.
    await ctx.fiber.dispose()
    await _expect_raises(handle.append([]), Exception, "append after teardown", "closed handle")

    verify = await backend.remount()
    _expect_equal(await _seqs(_persistence(verify), session.id), [0], "seqs")
    await verify.fiber.dispose()


@_case("close itself surfaces a failing drain and still releases write ownership")
async def close_surfaces_failing_drain(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    session = _sessions(ctx).create(SessionId("close-drain-fails"))
    handle = await persistence.create(session.header)
    failure = Exception("storage refused the drain")
    with _spy_persist_batch(persistence) as persist:
        persist.reject(failure)
        session.append("turn/start", {"turn": 1})
        raised = await _expect_raises(handle.close(), Exception, "close")
        if raised is not failure:
            raise AssertionError(f"close: expected the injected failure, got {raised!r}")
        # Ownership was released despite the failed drain: the id is claimable.
        second = await persistence.create(session.header)
        await second.close()
    await ctx.fiber.dispose()


@_case("a failed drain retains order, quiets the timer, and recovers exactly once")
async def failed_drain_retains_order(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    session = _sessions(ctx).create(SessionId("retained-order"))
    handle = await persistence.create(session.header)
    with _capture_warnings():
        # Materialize first, so the first live drain is a plain append.
        await handle.flush()
        with _spy_persist_batch(persistence) as persist:
            real = persist.original
            persist.reject(Exception("first drain refused"))

            session.append("turn/start", {"turn": 1})
            session.append("step/start", {"turn": 1, "step": 1})
            await asyncio.sleep(batch_delay)
            await _wait_until(lambda: persist.calls >= 1, "the first background drain")
            # Events arriving after the failure join the retained queue, and no new
            # timer fires while the automatic path is paused.
            session.append("step/end", {"turn": 1, "step": 1})
            session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
            await asyncio.sleep(batch_delay * 4)
            _expect_equal(persist.calls, 1, "persist_batch calls")

            persist.implement(real)
            _expect_equal(await _sessions(ctx).flush(session), True, "recovered flush")
            _expect_equal(await _seqs(persistence, session.id), [0, 1, 2, 3], "seqs")
    await handle.close()
    await ctx.fiber.dispose()


@_case("a second write handle after close routes subsequent events")
async def second_handle_routes(make: LiveWriteBackendFactory, batch_delay: float) -> None:
    ctx = (await make()).ctx
    persistence = _persistence(ctx)
    session = _sessions(ctx).create(SessionId("rebind"))
    first = await persistence.create(session.header)
    session.append("turn/start", {"turn": 1})
    await _sessions(ctx).flush(session)
    await first.close()

    second = await persistence.open(session.id, "write")
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await _sessions(ctx).flush(session)
    _expect_equal(await _seqs(persistence, session.id), [0, 1], "seqs")
    await second.close()
    await ctx.fiber.dispose()


def live_write_path_contract_cases() -> list[tuple[str, LiveWriteCheck]]:
    """Every live-write-path case as ``(upstream it title, check)``, in upstream order."""
    return list(_CASES)


async def run_live_write_path_contract(
    make_backend: LiveWriteBackendFactory, batch_delay: float
) -> list[ContractResult]:
    """Run every live-write-path case and collect the outcomes.

    ``batch_delay`` is the backend's fixed live batching window in seconds.
    """

    def bind(check: LiveWriteCheck) -> Callable[[], Awaitable[None]]:
        return lambda: check(make_backend, batch_delay)

    cases: Sequence[tuple[str, Callable[[], Awaitable[None]]]] = [
        (title, bind(check)) for title, check in _CASES
    ]
    return await run_cases(cases)
