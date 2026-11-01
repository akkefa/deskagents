"""Reusable handle contract for any :class:`~deskagents.session.persistence.SessionPersistence`.

Upstream: packages/session/session-persistence/tests/contract.ts
(``runPersistenceContract``) at ce7073d.

A backend package runs every check against a factory that yields a fresh backend over
fresh, empty storage (plus teardown, an optional same-storage reopen and an optional
physical tail corruptor), so every backend is held to the same create/open/handle
semantics: append-only contiguous seqs, single-writer ownership, lazy materialization,
fail-closed vocabulary, freshness and torn-tail repair.

The suite has no test-runner dependency. Each check is an ``async`` function taking the
factory; it raises :class:`AssertionError` (or the unexpected exception) on failure and
:class:`ContractSkip` when the backend lacks an optional capability the check needs
(upstream returns early instead). Wire it into pytest by parametrizing over
:func:`persistence_contract_cases`, or call :func:`run_persistence_contract`.

Python differences:

- Upstream's ``BigInt`` and ``undefined`` non-JSON values become an integer outside the
  safe range and a plain ``object()``.
- ``Symbol.asyncDispose`` becomes ``async with``; ``Promise.allSettled`` becomes
  ``asyncio.gather(..., return_exceptions=True)``.
- ``releasedV1OneTurnLog`` is not provided: Python reads format v4 only.
"""

from __future__ import annotations

import asyncio
import re
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionSeq,
)
from deskagents.llm import MessageId, freeze_message
from deskagents.session.persistence import (
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionFormatUnsupportedError,
    SessionHandle,
    SessionHandleClosedError,
    SessionPersistence,
    SessionPersistenceNotFoundError,
    SessionReadOnlyError,
)
from deskagents.util.values import MAX_SAFE_INTEGER, FrozenJsonObject

__all__ = [
    "PERSISTENCE_CONTRACT_SUITE",
    "ContractBackend",
    "ContractBackendFactory",
    "ContractBackendInstance",
    "ContractCheck",
    "ContractResult",
    "ContractSkip",
    "meta",
    "one_turn_log",
    "persistence_contract_cases",
    "run_persistence_contract",
    "second_turn",
]

PERSISTENCE_CONTRACT_SUITE = "SessionPersistence contract"
"""Upstream's describe title prefix; the full title is ``f"{SUITE}: {backend name}"``."""

T = TypeVar("T")


@dataclass
class ContractBackendInstance:
    """One backend service instance under test plus its teardown."""

    persistence: SessionPersistence
    dispose: Callable[[], Awaitable[None]]


@dataclass
class ContractBackend(ContractBackendInstance):
    """A backend under test: the primary instance plus optional storage-level capabilities."""

    reopen: Callable[[], Awaitable[ContractBackendInstance]] | None = None
    """Open a FRESH instance over the SAME storage, as another process would after this
    one exits. Enables the cross-instance tests; without it those checks skip."""
    corrupt_tail: Callable[[SessionId, str | None], Awaitable[None]] | None = None
    """Inject a torn physical tail after one stored session's committed log, simulating
    a crash mid-write. Enables the torn-tail check."""


ContractBackendFactory = Callable[[], Awaitable[ContractBackend]]
"""Produces one fresh backend over fresh, empty storage per call."""

ContractCheck = Callable[[ContractBackendFactory], Awaitable[None]]
"""One contract case: builds its own backend(s) from the factory and disposes them."""


class ContractSkip(Exception):
    """A check needs an optional backend capability (``reopen``, ``corrupt_tail``)."""


@dataclass(frozen=True)
class ContractResult:
    """The outcome of one contract case from a ``run_*_contract`` call."""

    title: str
    outcome: Literal["passed", "failed", "skipped"]
    error: BaseException | None = None
    detail: str = field(default="", compare=False)
    """Formatted traceback for a failure, or the skip reason."""


async def run_cases(
    cases: Sequence[tuple[str, Callable[[], Awaitable[None]]]],
) -> list[ContractResult]:
    """Run bound cases one after another, collecting outcomes instead of raising."""
    results: list[ContractResult] = []
    for title, run in cases:
        try:
            await run()
        except ContractSkip as skip:
            results.append(ContractResult(title, "skipped", None, str(skip)))
        except Exception as error:
            detail = "".join(traceback.format_exception(error))
            results.append(ContractResult(title, "failed", error, detail))
        else:
            results.append(ContractResult(title, "passed"))
    return results


# --- fixtures ----------------------------------------------------------------------------


def meta(session_id: str, cwd: str | None = None) -> SessionHeader:
    """A minimal :data:`SessionHeader` for a session id."""
    header: dict[str, Any] = {
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId(session_id),
        "createdAt": 1000,
        "isSeeded": False,
    }
    if cwd is not None:
        header["cwd"] = cwd
    return header


def one_turn_log() -> list[SessionEvent]:
    """A well-formed one-turn event log (contiguous seqs from 0)."""
    user = freeze_message(
        {
            "id": MessageId("one-turn-user"),
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
            "source": {"kind": "user"},
        }
    )
    assistant = freeze_message(
        {
            "id": MessageId("one-turn-assistant"),
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "source": {"kind": "model", "provider": "mock", "model": "mock"},
        }
    )
    stream = [
        {
            "type": "chunk",
            "time": 3,
            "chunk": {"type": "block-start", "index": 0, "blockType": "text"},
        },
        {"type": "text-chunks", "time0": 3, "index": 0, "dt": [], "texts": ["hello"]},
        {
            "type": "chunk",
            "time": 4,
            "chunk": {"type": "block-end", "index": 0, "block": {"type": "text", "text": "hello"}},
        },
        {"type": "chunk", "time": 4, "chunk": {"type": "finish", "reason": {"kind": "stop"}}},
    ]
    return [
        {"type": "turn/start", "seq": SessionSeq(0), "time": 1, "data": {"turn": 1}},
        {
            "type": "user/message",
            "seq": SessionSeq(1),
            "time": 2,
            "data": user,
            "surfaceOp": "append",
        },
        {"type": "step/start", "seq": SessionSeq(2), "time": 3, "data": {"turn": 1, "step": 1}},
        {
            "type": "assistant/message",
            "seq": SessionSeq(3),
            "time": 4,
            "data": {"turn": 1, "step": 1, "message": assistant, "stream": stream},
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": SessionSeq(4), "time": 5, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": SessionSeq(5),
            "time": 6,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]


def second_turn(start_seq: int = 6) -> list[SessionEvent]:
    """A contiguous second-turn batch continuing :func:`one_turn_log`."""
    return [
        {"type": "turn/start", "seq": SessionSeq(start_seq), "time": 9, "data": {"turn": 2}},
        {
            "type": "turn/end",
            "seq": SessionSeq(start_seq + 1),
            "time": 10,
            "data": {"turn": 2, "reason": {"kind": "completed"}},
        },
    ]


# --- assertion helpers -----------------------------------------------------------------


def _expect_equal(actual: object, expected: object, what: str) -> None:
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def _expect_matches(
    actual: Mapping[str, Any] | None, expected: Mapping[str, Any], what: str
) -> None:
    """``toMatchObject``: every expected key is present with an equal value."""
    if actual is None:
        raise AssertionError(f"{what}: expected an object matching {dict(expected)!r}, got None")
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise AssertionError(f"{what}: expected {dict(expected)!r} within {dict(actual)!r}")


async def _expect_raises(
    awaitable: Awaitable[object],
    kind: type[BaseException] | tuple[type[BaseException], ...],
    what: str,
    match: str | None = None,
) -> BaseException:
    try:
        await awaitable
    except kind as error:
        if match is not None and re.search(match, str(error)) is None:
            raise AssertionError(f"{what}: {error!r} does not match /{match}/") from error
        return error
    except BaseException as error:
        raise AssertionError(f"{what}: expected {kind!r}, got {error!r}") from error
    raise AssertionError(f"{what}: expected {kind!r}, but nothing was raised")


def _seqs(events: Sequence[SessionEvent]) -> list[int]:
    return [event["seq"] for event in events]


def _ids(snapshots: Sequence[Any]) -> list[str]:
    return [snapshot.header["id"] for snapshot in snapshots]


async def _events(
    handle: SessionHandle, offset: int = 0, length: int | None = None
) -> list[SessionEvent]:
    return list((await handle.read(offset, length)).events)


def _skip_without(backend: ContractBackend, *capabilities: str) -> None:
    missing = [name for name in capabilities if getattr(backend, name) is None]
    if missing:
        raise ContractSkip(f"backend lacks {', '.join(missing)}")


# --- cases -------------------------------------------------------------------------------

_CASES: list[tuple[str, ContractCheck]] = []


def _case(title: str) -> Callable[[ContractCheck], ContractCheck]:
    def register(check: ContractCheck) -> ContractCheck:
        _CASES.append((title, check))
        return check

    return register


@_case("round-trips through one write handle: append, self-read, offset/length defaults")
async def round_trips_through_one_write_handle(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        m = meta("round-trip", "/work")
        log = one_turn_log()
        handle = await backend.persistence.create(m)
        _expect_equal(handle.access, "write", "access")
        _expect_equal(handle.id, m["id"], "id")
        _expect_matches(handle.header, m, "header")

        await handle.append(log)
        # An empty batch is a no-op, not an error.
        await handle.append([])
        # A write handle reads its own successful appends.
        full = await handle.read()
        _expect_equal(list(full.events), log, "self-read")
        if full.event_state == "shared-frozen":
            frozen = all(
                isinstance(event, FrozenJsonObject) and isinstance(event["data"], FrozenJsonObject)
                for event in full.events
            )
            _expect_equal(frozen, True, "shared-frozen events are deeply frozen")
        _expect_equal(await _events(handle, 3), log[3:], "read(3)")
        _expect_equal(await _events(handle, 0, 2), log[0:2], "read(0, 2)")
        _expect_equal(await _events(handle, 1, 3), log[1:4], "read(1, 3)")
        # At/past the stored end: an empty list, never an error.
        _expect_equal(await _events(handle, len(log)), [], "read at end")
        _expect_equal(await _events(handle, len(log) + 100), [], "read past end")
        # flush after a durable append is a satisfied barrier, not an error.
        await handle.flush()
        await handle.close()
    finally:
        await backend.dispose()


@_case("read rejects negative or fractional offsets and lengths")
async def read_rejects_bad_offsets(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        handle = await backend.persistence.create(meta("read-args"))
        pattern = "non-negative safe integer"
        await _expect_raises(handle.read(-1), Exception, "read(-1)", pattern)
        await _expect_raises(handle.read(1.5), Exception, "read(1.5)", pattern)  # type: ignore[arg-type]
        await _expect_raises(handle.read(0, -1), Exception, "read(0, -1)", pattern)
        await handle.close()
    finally:
        await backend.dispose()


@_case(
    "duplicate create rejects against a live pending session and allows the id after an "
    "erasing close"
)
async def duplicate_create_pending(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        first = await backend.persistence.create(meta("dup-pending"))
        await _expect_raises(
            backend.persistence.create(meta("dup-pending")), SessionAlreadyExistsError, "duplicate"
        )
        # Closing the creator without ever appending erases the session, so the id is
        # free again.
        await first.close()
        second = await backend.persistence.create(meta("dup-pending"))
        await second.close()
    finally:
        await backend.dispose()


@_case("concurrent duplicate creates: one wins, the loser rejects SessionAlreadyExistsError")
async def concurrent_duplicate_creates(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        m = meta("dup-race")
        # Both calls pass the stored-existence check before either registers, so the
        # loser is refused at the claim, still as a duplicate create.
        results = await asyncio.gather(
            backend.persistence.create(m), backend.persistence.create(m), return_exceptions=True
        )
        winners = [result for result in results if not isinstance(result, BaseException)]
        losers = [result for result in results if isinstance(result, BaseException)]
        _expect_equal(len(winners), 1, "winners")
        _expect_equal(len(losers), 1, "losers")
        if not isinstance(losers[0], SessionAlreadyExistsError):
            raise AssertionError(f"loser: expected SessionAlreadyExistsError, got {losers[0]!r}")
        await winners[0].close()
    finally:
        await backend.dispose()


@_case("duplicate create rejects against a materialized artifact seen by a fresh instance")
async def duplicate_create_stored(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        _skip_without(backend, "reopen")
        assert backend.reopen is not None
        m = meta("dup-stored", "/work")
        handle = await backend.persistence.create(m)
        await handle.append(one_turn_log())
        await handle.close()

        reopened = await backend.reopen()
        try:
            await _expect_raises(
                reopened.persistence.create(meta("dup-stored", "/work")),
                SessionAlreadyExistsError,
                "duplicate against stored",
            )
        finally:
            await reopened.dispose()
    finally:
        await backend.dispose()


@_case("open of an absent session rejects with SessionPersistenceNotFoundError for both accesses")
async def open_absent(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        absent = SessionId("absent")
        await _expect_raises(
            backend.persistence.open(absent, "read"), SessionPersistenceNotFoundError, "read"
        )
        await _expect_raises(
            backend.persistence.open(absent, "write"), SessionPersistenceNotFoundError, "write"
        )
    finally:
        await backend.dispose()


@_case("write ownership is single-holder per instance and released by close")
async def write_ownership_single_holder(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        persistence = backend.persistence
        m = meta("owned")
        creator = await persistence.create(m)
        # The creator holds ownership even before materialization.
        await _expect_raises(
            persistence.open(m["id"], "write"), SessionAlreadyOwnedError, "before materialization"
        )
        await creator.append(one_turn_log())
        await _expect_raises(
            persistence.open(m["id"], "write"), SessionAlreadyOwnedError, "after materialization"
        )
        await creator.close()

        # After close, a new write handle continues at the stored next-seq.
        writer = await persistence.open(m["id"], "write")
        await writer.append(second_turn())
        _expect_equal(_seqs(await _events(writer)), [0, 1, 2, 3, 4, 5, 6, 7], "seqs")
        await writer.close()
    finally:
        await backend.dispose()


@_case("a read handle refuses append and flush with SessionReadOnlyError")
async def read_handle_refuses_mutation(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        m = meta("read-only")
        writer = await backend.persistence.create(m)
        await writer.append(one_turn_log())
        await writer.close()

        reader = await backend.persistence.open(m["id"], "read")
        _expect_equal(reader.access, "read", "access")
        await _expect_raises(reader.append(second_turn()), SessionReadOnlyError, "append")
        await _expect_raises(reader.flush(), SessionReadOnlyError, "flush")
        # The refusals mutated nothing.
        _expect_equal(await _events(reader), one_turn_log(), "log")
        await reader.close()
    finally:
        await backend.dispose()


@_case("operations on a closed handle reject; close is idempotent; asyncDispose releases ownership")
async def closed_handle_rejects(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        persistence = backend.persistence
        m = meta("closed")
        handle = await persistence.create(m)
        await handle.append(one_turn_log())
        await handle.close()
        await handle.close()
        await _expect_raises(handle.read(), SessionHandleClosedError, "read")
        await _expect_raises(handle.append(second_turn()), SessionHandleClosedError, "append")
        await _expect_raises(handle.flush(), SessionHandleClosedError, "flush")

        async with await persistence.open(m["id"], "write") as writer:
            await writer.append(second_turn())
        # Leaving the block disposed the handle, so ownership is free again.
        reopened = await persistence.open(m["id"], "write")
        await reopened.close()
    finally:
        await backend.dispose()


@_case(
    "service-level flush materializes every active write handle and counts a closing one as flushed"
)
async def service_flush_materializes(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        materialized = await backend.persistence.create(meta("flush-all"))
        abandoned = await backend.persistence.create(meta("flush-all-closing"))
        # Close starts before the barrier: the swept handle refuses its flush, which
        # counts as flushed — close itself drained durably.
        closing = abandoned.close()
        await backend.persistence.flush()
        await closing

        if backend.reopen is not None:
            reopened = await backend.reopen()
            try:
                # The barrier materialized the empty session durably...
                if await reopened.persistence.stat(SessionId("flush-all")) is None:
                    raise AssertionError("flush-all: expected a durable empty session")
                # ...while the one that closed unappended never existed.
                _expect_equal(
                    await reopened.persistence.stat(SessionId("flush-all-closing")),
                    None,
                    "flush-all-closing",
                )
            finally:
                await reopened.dispose()
        await materialized.close()
    finally:
        await backend.dispose()


@_case("a created-but-unappended session is visible to this instance and invisible to a fresh one")
async def lazy_materialization_visibility(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        m = meta("lazy", "/work")
        creator = await backend.persistence.create(m)
        # The creator's own reads see the empty log before materialization.
        _expect_equal(await _events(creator), [], "creator read")

        snapshot = await backend.persistence.stat(m["id"])
        _expect_matches(None if snapshot is None else snapshot.header, m, "stat header")
        listed = _ids(await backend.persistence.list())
        if m["id"] not in listed:
            raise AssertionError(f"list: expected {m['id']!r} in {listed!r}")
        reader = await backend.persistence.open(m["id"], "read")
        _expect_equal(await _events(reader), [], "reader read")
        await reader.close()

        if backend.reopen is not None:
            reopened = await backend.reopen()
            try:
                _expect_equal(await reopened.persistence.stat(m["id"]), None, "fresh stat")
                fresh = _ids(await reopened.persistence.list())
                if m["id"] in fresh:
                    raise AssertionError(f"fresh list: expected no {m['id']!r} in {fresh!r}")
                await _expect_raises(
                    reopened.persistence.open(m["id"], "read"),
                    SessionPersistenceNotFoundError,
                    "fresh open",
                )
            finally:
                await reopened.dispose()
        await creator.close()
    finally:
        await backend.dispose()


@_case("close without an append erases the created session from this instance")
async def close_without_append_erases(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        persistence = backend.persistence
        m = meta("never-was")
        creator = await persistence.create(m)
        await creator.close()

        _expect_equal(await persistence.stat(m["id"]), None, "stat")
        listed = _ids(await persistence.list())
        if m["id"] in listed:
            raise AssertionError(f"list: expected no {m['id']!r} in {listed!r}")
        await _expect_raises(
            persistence.open(m["id"], "read"), SessionPersistenceNotFoundError, "open"
        )
    finally:
        await backend.dispose()


@_case("flush materializes an empty session durably for a fresh instance")
async def flush_materializes_empty(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        _skip_without(backend, "reopen")
        assert backend.reopen is not None
        m = meta("durable-empty", "/work")
        creator = await backend.persistence.create(m)
        await creator.flush()
        await creator.close()

        reopened = await backend.reopen()
        try:
            listed = _ids(await reopened.persistence.list())
            if m["id"] not in listed:
                raise AssertionError(f"list: expected {m['id']!r} in {listed!r}")
            snapshot = await reopened.persistence.stat(m["id"])
            _expect_matches(None if snapshot is None else snapshot.header, m, "stat header")
            reader = await reopened.persistence.open(m["id"], "read")
            _expect_equal(await _events(reader), [], "read")
            await reader.close()
        finally:
            await reopened.dispose()
    finally:
        await backend.dispose()


@_case("freshness: reads started after an append resolves observe that prefix on any handle")
async def freshness(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        persistence = backend.persistence
        m = meta("fresh")
        writer = await persistence.create(m)
        await writer.append(one_turn_log())

        before = await persistence.open(m["id"], "read")
        _expect_equal(await _events(before), one_turn_log(), "before")

        await writer.append(second_turn())
        # Both a pre-existing read handle and a freshly opened one observe the append
        # once it has resolved.
        expected = [0, 1, 2, 3, 4, 5, 6, 7]
        _expect_equal(_seqs(await _events(before)), expected, "pre-existing reader")
        after = await persistence.open(m["id"], "read")
        _expect_equal(_seqs(await _events(after)), expected, "fresh reader")
        await before.close()
        await after.close()
        await writer.close()
    finally:
        await backend.dispose()


@_case("a fresh instance continues the stored log at the committed next-seq")
async def fresh_instance_continues(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        _skip_without(backend, "reopen")
        assert backend.reopen is not None
        m = meta("continue", "/work")
        creator = await backend.persistence.create(m)
        await creator.append(one_turn_log())
        await creator.close()

        reopened = await backend.reopen()
        try:
            writer = await reopened.persistence.open(m["id"], "write")
            await writer.append(second_turn())
            _expect_equal(_seqs(await _events(writer)), [0, 1, 2, 3, 4, 5, 6, 7], "seqs")
            await writer.close()
        finally:
            await reopened.dispose()
    finally:
        await backend.dispose()


@_case(
    "append rejects a batch that does not contiguously continue the log, naming the expected seq"
)
async def append_rejects_non_contiguous(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        handle = await backend.persistence.create(meta("contiguity"))
        await handle.append(one_turn_log())  # seqs 0..5, next-seq = 6
        # A re-append of an already-stored seq is rejected, not duplicated.
        await _expect_raises(handle.append(one_turn_log()), Exception, "re-append", "expected 6")
        # A mid-batch gap is rejected as a whole.
        gapped: list[SessionEvent] = [
            {"type": "turn/start", "seq": SessionSeq(6), "time": 9, "data": {"turn": 2}},
            {
                "type": "turn/end",
                "seq": SessionSeq(8),
                "time": 10,
                "data": {"turn": 2, "reason": {"kind": "completed"}},
            },
        ]
        await _expect_raises(handle.append(gapped), Exception, "gap", "expected 7")
        # Neither rejection changed the stored log.
        _expect_equal(_seqs(await _events(handle)), [0, 1, 2, 3, 4, 5], "seqs")
        await handle.close()
    finally:
        await backend.dispose()


@_case("append rejects non-JSON-serializable event data without storing anything")
async def append_rejects_non_json(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        handle = await backend.persistence.create(meta("non-json"))

        def bad(extra: object) -> list[SessionEvent]:
            return [
                {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1, "extra": extra}}
            ]

        big = MAX_SAFE_INTEGER + 1  # upstream: a BigInt
        await _expect_raises(handle.append(bad(big)), TypeError, "unsafe integer")
        pattern = "losslessly JSON-serializable"
        await _expect_raises(handle.append(bad(big)), Exception, "unsafe integer", pattern)
        # upstream: undefined
        await _expect_raises(handle.append(bad(object())), Exception, "object()", pattern)
        # The rejected batches left no events behind: seq 0 is still free.
        await handle.append(one_turn_log())
        _expect_equal(await _events(handle), one_turn_log(), "log")
        await handle.close()
    finally:
        await backend.dispose()


@_case("vocabulary fail-closed: an unknown stored event type refuses reads and write opens")
async def vocabulary_fail_closed(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        persistence = backend.persistence
        m = meta("foreign-vocabulary")
        handle = await persistence.create(m)
        # The append side is permissive — a newer producer's event type is stored
        # verbatim…
        await handle.append(one_turn_log())
        await handle.append(
            [{"type": "mystery/event", "seq": SessionSeq(6), "time": 7, "data": {"payload": True}}]
        )
        await handle.close()

        # …but this build refuses to interpret the stored log: a write open rejects,
        # and a read handle (or its first read) rejects.
        await _expect_raises(
            persistence.open(m["id"], "write"), SessionFormatUnsupportedError, "write open"
        )
        # The failed write open released its ownership claim: retrying yields the same
        # refusal, never SessionAlreadyOwnedError.
        await _expect_raises(
            persistence.open(m["id"], "write"), SessionFormatUnsupportedError, "write reopen"
        )
        read_failure: BaseException | None = None
        try:
            reader = await persistence.open(m["id"], "read")
        except Exception as error:
            read_failure = error
        else:
            try:
                await reader.read()
            except Exception as error:
                read_failure = error
            finally:
                await reader.close()
        if not isinstance(read_failure, SessionFormatUnsupportedError):
            raise AssertionError(
                f"read: expected SessionFormatUnsupportedError, got {read_failure!r}"
            )
        if "mystery/event" not in str(read_failure):
            raise AssertionError(f"read: {read_failure!r} does not name mystery/event")
    finally:
        await backend.dispose()


@_case("a torn physical tail is never served and is durably truncated by the write path")
async def torn_tail(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        _skip_without(backend, "reopen", "corrupt_tail")
        assert backend.reopen is not None and backend.corrupt_tail is not None
        m = meta("torn", "/work")
        creator = await backend.persistence.create(m)
        await creator.append(one_turn_log())
        await creator.close()
        await backend.corrupt_tail(m["id"], m.get("cwd"))

        # A reader over the corrupted artifact serves only the committed prefix.
        reader_instance = await backend.reopen()
        try:
            reader = await reader_instance.persistence.open(m["id"], "read")
            _expect_equal(await _events(reader), one_turn_log(), "committed prefix")
            await reader.close()

            # A write open + first append durably truncates the torn tail and continues
            # at the committed next-seq.
            writer = await reader_instance.persistence.open(m["id"], "write")
            await writer.append(second_turn())
            _expect_equal(_seqs(await _events(writer)), [0, 1, 2, 3, 4, 5, 6, 7], "writer")
            await writer.close()
        finally:
            await reader_instance.dispose()

        # The repaired log is intact for the next instance.
        verify_instance = await backend.reopen()
        try:
            verify = await verify_instance.persistence.open(m["id"], "read")
            _expect_equal(_seqs(await _events(verify)), [0, 1, 2, 3, 4, 5, 6, 7], "repaired")
            await verify.close()
        finally:
            await verify_instance.dispose()
    finally:
        await backend.dispose()


@_case("stat and list agree on stable revisions that change after an append")
async def stat_and_list_revisions(make: ContractBackendFactory) -> None:
    backend = await make()
    try:
        persistence = backend.persistence
        m = meta("revisions", "/work")
        writer = await persistence.create(m)
        await writer.append(one_turn_log())

        def find(snapshots: Sequence[Any]) -> Any:
            return next((s for s in snapshots if s.header["id"] == m["id"]), None)

        stat_first = await persistence.stat(m["id"])
        stat_again = await persistence.stat(m["id"])
        list_first = find(await persistence.list())
        if stat_first is None or stat_again is None or list_first is None:
            raise AssertionError("stat/list: expected a snapshot")
        _expect_equal(stat_again.revision, stat_first.revision, "stable stat revision")
        _expect_equal(list_first.revision, stat_first.revision, "list revision")

        await writer.append(second_turn())
        stat_changed = await persistence.stat(m["id"])
        if stat_changed is None:
            raise AssertionError("stat after append: expected a snapshot")
        if stat_changed.revision == stat_first.revision:
            raise AssertionError(f"revision did not change: {stat_changed.revision!r}")
        list_changed = find(await persistence.list())
        if list_changed is None:
            raise AssertionError("list after append: expected a snapshot")
        _expect_equal(list_changed.revision, stat_changed.revision, "changed list revision")

        # Snapshot headers carry the stored header, identically everywhere.
        reader = await persistence.open(m["id"], "read")
        _expect_equal(stat_changed.header, reader.header, "stat header")
        _expect_equal(list_changed.header, reader.header, "list header")
        _expect_matches(stat_changed.header, m, "stat header matches meta")
        await reader.close()
        await writer.close()

        _expect_equal(await persistence.stat(SessionId("absent-stat")), None, "absent stat")
    finally:
        await backend.dispose()


def persistence_contract_cases() -> list[tuple[str, ContractCheck]]:
    """Every handle-contract case as ``(upstream it title, check)``, in upstream order."""
    return list(_CASES)


async def run_persistence_contract(make_backend: ContractBackendFactory) -> list[ContractResult]:
    """Run every handle-contract case against ``make_backend`` and collect the outcomes.

    ``make_backend`` must return a fresh backend over fresh, empty storage each call.
    """

    def bind(check: ContractCheck) -> Callable[[], Awaitable[None]]:
        return lambda: check(make_backend)

    return await run_cases([(title, bind(check)) for title, check in _CASES])
