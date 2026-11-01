"""Ported from packages/session/session-persistence-jsonl/tests/jsonl.spec.ts.

Only the "JsonlSessionPersistence: durability and crash semantics" block is ported here.
Upstream's ``vi.mock('node:fs/promises')`` probes become monkeypatches of the functions
the Python modules call (``os.stat``, ``os.fsync``, ``generation._read_stable``) and of
the backend's private methods.

Not applicable (``AbortSignal`` has no Python counterpart; operations are cancelled by
cancelling their task, and no method takes a signal):

- "rejects pre-aborted operations with the exact cancellation reason"
- "forwards list cancellation and awaits in-flight discovery cleanup"
- "checks cancellation after an uncancellable list stat settles"
- the third (signal) step of "stat handles artifact removal and non-ENOENT failures after
  log discovery"; its first two steps are ported.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
)
from deskagents.llm import create_message
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import SessionPersistence
from deskagents.session.persistence_jsonl import JsonlSessionPersistence
from deskagents.session.persistence_jsonl import generation as generation_module
from deskagents.session.persistence_jsonl.format import log_path, session_dir, to_header_line
from deskagents.session.persistence_jsonl.storage import JsonlSessionHandle
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/jsonl.spec.ts"
DESCRIBE = "JsonlSessionPersistence: durability and crash semantics"


def upstream(title: str) -> pytest.MarkDecorator:
    return pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > {title}")


# --- helpers (local copies of upstream's contract fixtures) ---------------------------


def meta(id_: str, cwd: str | None = None) -> dict[str, Any]:
    header: dict[str, Any] = {
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId(id_),
        "createdAt": 1000,
        "isSeeded": False,
    }
    if cwd is not None:
        header["cwd"] = cwd
    return header


def one_turn_log() -> list[dict[str, Any]]:
    return [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "user/message",
            "seq": 1,
            "time": 2,
            "data": {
                "id": "one-turn-user",
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {"type": "step/start", "seq": 2, "time": 3, "data": {"turn": 1, "step": 1}},
        {
            "type": "assistant/message",
            "seq": 3,
            "time": 4,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "one-turn-assistant",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "source": {"kind": "model", "provider": "mock", "model": "mock"},
                },
                "stream": [
                    {
                        "type": "chunk",
                        "time": 3,
                        "chunk": {"type": "block-start", "index": 0, "blockType": "text"},
                    },
                    {"type": "text-chunks", "time0": 3, "index": 0, "dt": [], "texts": ["hello"]},
                    {
                        "type": "chunk",
                        "time": 4,
                        "chunk": {
                            "type": "block-end",
                            "index": 0,
                            "block": {"type": "text", "text": "hello"},
                        },
                    },
                    {
                        "type": "chunk",
                        "time": 4,
                        "chunk": {"type": "finish", "reason": {"kind": "stop"}},
                    },
                ],
            },
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 4, "time": 5, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 5,
            "time": 6,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]


def second_turn() -> list[dict[str, Any]]:
    return [
        {"type": "turn/start", "seq": 6, "time": 9, "data": {"turn": 2}},
        {
            "type": "turn/end",
            "seq": 7,
            "time": 10,
            "data": {"turn": 2, "reason": {"kind": "completed"}},
        },
    ]


# Blocking file helpers: sync, so the async cases stay free of ASYNC2xx findings.


def exists(path: str) -> bool:
    return os.path.exists(path)


def is_dir(path: str) -> bool:
    return os.path.isdir(path)


def is_file(path: str) -> bool:
    return os.path.isfile(path)


def read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    Path(path).write_text(text, encoding=encoding)


def append_text(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(text)


def raw_log_path(root: str, cwd: str | None, id_: str) -> str:
    return log_path(root, cwd, id_, "none")


def rewrite_header(path: str, update: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite only a stored header while preserving every event byte below it."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    header = dict(parse_json(lines[0]))  # type: ignore[arg-type]
    update(header)
    lines[0] = stringify_json(header)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


async def write_log(
    persistence: SessionPersistence, m: SessionHeader, events: Sequence[SessionEvent]
) -> None:
    handle = await persistence.create(m)
    try:
        await handle.append(events)
    finally:
        await handle.close()


async def read_all(
    persistence: SessionPersistence, id_: str
) -> tuple[SessionHeader, list[SessionEvent]]:
    handle = await persistence.open(id_, "read")
    try:
        return handle.header, list((await handle.read()).events)
    finally:
        await handle.close()


async def append_batch(
    persistence: SessionPersistence, id_: str, events: Sequence[SessionEvent]
) -> None:
    handle = await persistence.open(id_, "write")
    try:
        await handle.append(events)
    finally:
        await handle.close()


async def mount(root: str) -> Context:
    ctx = Context()
    await ctx.plugin(jsonl.apply, {"root": root, "compression": "none"})
    return ctx


def backend(ctx: Context) -> JsonlSessionPersistence:
    service = ctx.get("sessionPersistence")
    assert isinstance(service, JsonlSessionPersistence)
    return service


class FakeStat:
    """The identity fields ``JsonlPhysicalIdentity.of`` reads, with an adjustable mtime."""

    def __init__(self, real: os.stat_result, mtime_delta: int) -> None:
        self.st_dev = real.st_dev
        self.st_ino = real.st_ino
        self.st_size = real.st_size
        self.st_mtime_ns = real.st_mtime_ns + mtime_delta
        self.st_ctime_ns = real.st_ctime_ns
        self.st_mode = real.st_mode


@pytest.fixture
def root(tmp_path: Path) -> str:
    return str(tmp_path / "root")


@pytest_asyncio.fixture
async def ctx(root: str) -> AsyncIterator[Context]:
    context = await mount(root)
    yield context
    await context.fiber.dispose()


@pytest.fixture
def persistence(ctx: Context) -> JsonlSessionPersistence:
    return backend(ctx)


# --- cases ----------------------------------------------------------------------------


@upstream("lazy materialization: create() writes no file until the first append")
@pytest.mark.asyncio
async def test_lazy_materialization(root: str, persistence: JsonlSessionPersistence) -> None:
    m = meta("lazy", "/work")
    handle = await persistence.create(m)
    assert (await handle.read()).event_state == "detached"
    directory = session_dir(root, "/work", m["id"])
    assert not exists(raw_log_path(root, "/work", m["id"]))
    pending = await persistence.stat(m["id"])
    pending_rows = await persistence.list()
    assert m["id"] in [s.header["id"] for s in pending_rows]
    assert pending is not None
    assert pending.size_bytes is None
    assert pending.format_status is None
    assert pending.location is None
    row = next(s for s in pending_rows if s.header["id"] == m["id"])
    assert row.format_status is None
    assert row.location is None

    await handle.append(one_turn_log())
    assert is_dir(directory)
    assert is_file(raw_log_path(root, "/work", m["id"]))
    materialized_path = raw_log_path(root, "/work", m["id"])
    assert m["id"] in [s.header["id"] for s in await persistence.list()]
    snapshot = await persistence.stat(m["id"])
    assert snapshot is not None
    assert snapshot.format_status == "current"
    assert snapshot.location is not None
    assert (snapshot.location.kind, snapshot.location.path) == ("jsonl", materialized_path)
    listed = next(s for s in await persistence.list() if s.header["id"] == m["id"])
    assert listed.location is not None
    assert (listed.location.kind, listed.location.path) == ("jsonl", materialized_path)
    await handle.close()


@upstream("flush materializes an explicitly durable empty session without an event row")
@pytest.mark.asyncio
async def test_flush_materializes_empty_session(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    m = meta("durable-empty", "/work")
    handle = await persistence.create(m)
    await handle.flush()
    await handle.close()

    assert read_text(raw_log_path(root, "/work", m["id"])) == (
        f"{stringify_json(to_header_line(m))}\n"
    )
    reader = await persistence.open(m["id"], "read")
    read = await reader.read()
    assert read.event_state == "shared-frozen"
    assert list(read.events) == []
    await reader.close()


@upstream("close drains a routed event that arrives while it waits for an in-flight append")
@pytest.mark.asyncio
async def test_close_drains_late_routed_event(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("late-closer", "/work")
    handle = await persistence.create(m)
    assert isinstance(handle, JsonlSessionHandle)
    original = persistence.persist_batch
    gate = asyncio.Event()
    entered = asyncio.Event()
    calls = 0

    async def gated(*args: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await gate.wait()
        await original(*args)

    monkeypatch.setattr(persistence, "persist_batch", gated)

    start, *rest = one_turn_log()
    inflight = asyncio.ensure_future(handle.append([start]))
    await entered.wait()
    closing = handle.close()
    for event in rest:
        handle.enqueue_live(event, lambda _error: None)
    gate.set()
    await inflight
    await closing

    reopened = await mount(root)
    _, events = await read_all(backend(reopened), m["id"])
    assert events == one_turn_log()
    await reopened.fiber.dispose()


@upstream("service flush skips a write claim whose handle is still opening")
@pytest.mark.asyncio
async def test_service_flush_skips_opening_claim(
    persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("opening-claim", "/work")
    await write_log(persistence, m, one_turn_log())
    original = persistence._require_stored_log
    gate = asyncio.Event()
    entered = asyncio.Event()
    calls = 0

    async def gated(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await gate.wait()
        return await original(*args)

    monkeypatch.setattr(persistence, "_require_stored_log", gated)

    opening = asyncio.ensure_future(persistence.open(m["id"], "write"))
    await entered.wait()
    await persistence.flush()
    gate.set()
    await (await opening).close()


@upstream("stat and list carry the physical artifact size once materialized")
@pytest.mark.asyncio
async def test_stat_and_list_carry_size(root: str, persistence: JsonlSessionPersistence) -> None:
    m = meta("sized", "/work")
    await write_log(persistence, m, one_turn_log())
    size = os.stat(raw_log_path(root, "/work", m["id"])).st_size
    snapshot = await persistence.stat(m["id"])
    assert snapshot is not None and snapshot.size_bytes == size
    listed = next(s for s in await persistence.list() if s.header["id"] == m["id"])
    assert listed.size_bytes == size


@upstream("binds revisions to the physical artifact identity")
@pytest.mark.asyncio
async def test_revisions_bind_to_physical_identity(
    root: str, tmp_path: Path, persistence: JsonlSessionPersistence
) -> None:
    m = meta("revision-source")
    await write_log(persistence, m, one_turn_log())
    first = await persistence.stat(m["id"])
    assert first is not None
    revision = first.revision

    reopened = await mount(root)
    again = await backend(reopened).stat(m["id"])
    assert again is not None and again.revision == revision

    other = await mount(str(tmp_path / "other-root"))
    await write_log(backend(other), m, one_turn_log())
    foreign = await backend(other).stat(m["id"])
    assert foreign is not None and foreign.revision != revision

    await reopened.fiber.dispose()
    await other.fiber.dispose()


def tally_reads(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count physical whole-file reads per path."""
    tally: dict[str, int] = {}
    real = generation_module._read_stable

    def counted(path: str) -> Any:
        tally[path] = tally.get(path, 0) + 1
        return real(path)

    monkeypatch.setattr(generation_module, "_read_stable", counted)
    return tally


@upstream("an unchanged cold log parses once across an observe-then-resume handoff")
@pytest.mark.asyncio
async def test_cold_log_parses_once_across_handoff(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("memo-handoff", "/work")
    await write_log(persistence, m, one_turn_log())
    path = raw_log_path(root, "/work", m["id"])
    tally = tally_reads(monkeypatch)

    assert (await read_all(persistence, m["id"]))[1] == one_turn_log()
    assert tally.get(path) == 1
    writer = await persistence.open(m["id"], "write")
    assert len((await writer.read()).events) == len(one_turn_log())
    assert tally.get(path) == 1

    await writer.append(second_turn())
    await writer.close()
    assert [e["seq"] for e in (await read_all(persistence, m["id"]))[1]] == list(range(8))
    assert tally.get(path) == 2


@upstream("a foreign write misses the memo through the revision guard")
@pytest.mark.asyncio
async def test_foreign_write_misses_memo(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("memo-foreign", "/work")
    await write_log(persistence, m, one_turn_log())
    path = raw_log_path(root, "/work", m["id"])
    await read_all(persistence, m["id"])

    foreign = await mount(root)
    writer = await backend(foreign).open(m["id"], "write")
    await writer.append(second_turn())
    await writer.close()
    await foreign.fiber.dispose()

    tally = tally_reads(monkeypatch)
    assert [e["seq"] for e in (await read_all(persistence, m["id"]))[1]] == list(range(8))
    assert tally.get(path) == 1


@upstream("the cold-log memo keeps only the handoff window and evicts the oldest id")
@pytest.mark.asyncio
async def test_cold_log_memo_evicts_oldest(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = meta("memo-evict-a", "/work")
    second = meta("memo-evict-b", "/work")
    third = meta("memo-evict-c", "/work")
    for m in (first, second, third):
        await write_log(persistence, m, one_turn_log())
    tally = tally_reads(monkeypatch)

    await read_all(persistence, first["id"])
    await read_all(persistence, second["id"])
    await read_all(persistence, third["id"])  # evicts the first id
    await read_all(persistence, third["id"])  # still memoized
    await read_all(persistence, first["id"])  # re-parses after eviction
    assert tally.get(raw_log_path(root, "/work", first["id"])) == 2
    assert tally.get(raw_log_path(root, "/work", third["id"])) == 1


def race_stat(monkeypatch: pytest.MonkeyPatch, path: str, mode: str) -> list[int]:
    """'settle': the revision changes once and then holds; 'churn': every stat differs."""
    reads = [0]
    real = os.stat

    def raced(target: Any, *args: Any, **kwargs: Any) -> Any:
        identity = real(target, *args, **kwargs)
        if str(target) != path:
            return identity
        reads[0] += 1
        if mode == "churn":
            return FakeStat(identity, reads[0])
        if reads[0] < 3:
            return identity
        return FakeStat(identity, 1)

    monkeypatch.setattr(os, "stat", raced)
    return reads


@upstream("a handle read retries once when the file revision changes during the read")
@pytest.mark.asyncio
async def test_read_retries_once_on_revision_change(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("read-revision-race", "/work")
    await write_log(persistence, m, one_turn_log())
    handle = await persistence.open(m["id"], "read")
    try:
        persistence._cold_log_memo.clear()
        reads = race_stat(monkeypatch, raw_log_path(root, "/work", m["id"]), "settle")
        assert list((await handle.read()).events) == one_turn_log()
        assert reads[0] == 4
    finally:
        monkeypatch.undo()
        await handle.close()


@upstream("a continuously churning revision yields the committed prefix instead of looping")
@pytest.mark.asyncio
async def test_churning_revision_yields_committed_prefix(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("read-revision-churn", "/work")
    await write_log(persistence, m, one_turn_log())
    handle = await persistence.open(m["id"], "read")
    try:
        persistence._cold_log_memo.clear()
        reads = race_stat(monkeypatch, raw_log_path(root, "/work", m["id"]), "churn")
        assert list((await handle.read()).events) == one_turn_log()
        assert reads[0] == 4
    finally:
        monkeypatch.undo()
        await handle.close()


@upstream("appends on reopen extend the same artifact and a fork materializes its own")
@pytest.mark.asyncio
async def test_reopen_appends_and_fork_materializes(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    parent = meta("location-parent", "/work")
    await write_log(persistence, parent, one_turn_log())
    parent_path = raw_log_path(root, "/work", parent["id"])
    size_before = os.stat(parent_path).st_size

    loaded_meta, loaded_events = await read_all(persistence, parent["id"])
    writer = await persistence.open(parent["id"], "write")
    await writer.append(second_turn())
    await writer.close()
    assert os.stat(parent_path).st_size > size_before

    child = {
        **loaded_meta,
        "id": SessionId("location-child"),
        "parentSession": parent["id"],
        "isSeeded": True,
    }
    child_handle = await persistence.create(
        child, inherited_event_count=SessionLogOffset(len(loaded_events))
    )
    await child_handle.append(
        [
            *loaded_events,
            {
                "type": "session/end-seed",
                "seq": len(loaded_events),
                "time": loaded_events[-1]["time"] if loaded_events else 1,
                "data": {"inherited": True},
            },
        ]
    )
    await child_handle.close()
    child_path = raw_log_path(root, "/work", child["id"])
    assert child_path != parent_path
    assert is_file(child_path)


@upstream("round-trip is byte-identical including nested Assistant stream records")
@pytest.mark.asyncio
async def test_round_trip_nested_stream(persistence: JsonlSessionPersistence) -> None:
    m = meta("chunks")
    log: list[SessionEvent] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 2, "data": {"turn": 1, "step": 1}},
        {
            "type": "assistant/message",
            "seq": 2,
            "time": 5,
            "data": {
                "turn": 1,
                "step": 1,
                "message": create_message(
                    role="assistant",
                    content=[{"type": "text", "text": "hello"}],
                    source={"kind": "model", "provider": "mock", "model": "mock"},
                ),
                "stream": [
                    {
                        "type": "text-chunks",
                        "time0": 3,
                        "index": 0,
                        "dt": [1],
                        "texts": ["he", "llo"],
                    },
                    {
                        "type": "chunk",
                        "time": 5,
                        "chunk": {"type": "finish", "reason": {"kind": "stop"}},
                    },
                ],
            },
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 3, "time": 6, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 4,
            "time": 7,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    await write_log(persistence, m, log)
    assert (await read_all(persistence, m["id"]))[1] == log


@upstream("a torn crash tail is served as the committed prefix and repaired only by the write path")
@pytest.mark.asyncio
async def test_torn_tail_repaired_by_write_path(
    root: str, persistence: JsonlSessionPersistence, caplog: pytest.LogCaptureFixture
) -> None:
    m = meta("crash", "/proj")
    await write_log(persistence, m, one_turn_log())
    path = raw_log_path(root, "/proj", m["id"])
    committed = read_bytes(path)
    tail = "\n".join(
        [
            stringify_json({"type": "turn/start", "seq": 6, "time": 8, "data": {"turn": 2}}),
            stringify_json(
                {"type": "step/start", "seq": 7, "time": 9, "data": {"turn": 2, "step": 1}}
            ),
            '{"type":"assistant/chunk","seq":8,"ti',
        ]
    )
    append_text(path, tail)

    _, events = await read_all(persistence, m["id"])
    assert [e["seq"] for e in events] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert read_bytes(path) == committed + tail.encode()

    with caplog.at_level(logging.WARNING):
        await append_batch(
            persistence,
            m["id"],
            [
                {"type": "step/end", "seq": 8, "time": 10, "data": {"turn": 2, "step": 1}},
                {
                    "type": "turn/end",
                    "seq": 9,
                    "time": 11,
                    "data": {"turn": 2, "reason": {"kind": "interrupted"}},
                },
            ],
        )
    assert any("recovered from a torn tail" in record.getMessage() for record in caplog.records)
    repaired = read_text(path)
    assert repaired.startswith(committed.decode())
    assert "assistant/chunk" not in repaired
    _, reloaded = await read_all(persistence, m["id"])
    assert [e["seq"] for e in reloaded] == list(range(10))


@upstream("a stored open turn is served as stored, with no synthetic closers")
@pytest.mark.asyncio
async def test_stored_open_turn_served_as_stored(persistence: JsonlSessionPersistence) -> None:
    m = meta("open-turn", "/h")
    await write_log(
        persistence, m, [{"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}]
    )
    _, events = await read_all(persistence, m["id"])
    assert [e["type"] for e in events] == ["turn/start"]


def fail_fsync_once(monkeypatch: pytest.MonkeyPatch, *failures: OSError) -> None:
    """Make the next ``len(failures)`` fsync calls fail, in order."""
    real = os.fsync
    pending = list(failures)

    def fsync(fd: int) -> None:
        if pending:
            raise pending.pop(0)
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)


@upstream("a failed appendLines truncates partial bytes so a retry has no seq gap")
@pytest.mark.asyncio
async def test_failed_append_truncates_partial_bytes(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("truncate-retry")
    await write_log(persistence, m, one_turn_log())
    size_before = os.stat(raw_log_path(root, None, m["id"])).st_size

    handle = await persistence.open(m["id"], "write")
    try:
        fail_fsync_once(monkeypatch, OSError(errno.ENOSPC, "simulated fsync ENOSPC"))
        turn2 = second_turn()
        with pytest.raises(OSError, match="ENOSPC"):
            await handle.append(turn2)
        assert os.stat(raw_log_path(root, None, m["id"])).st_size == size_before
        monkeypatch.undo()

        await handle.append(turn2)
        assert [e["seq"] for e in (await handle.read()).events] == list(range(8))
    finally:
        await handle.close()


@upstream("reports both the append failure and a failed rollback")
@pytest.mark.asyncio
async def test_reports_append_and_rollback_failures(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("rollback-failure")
    await write_log(persistence, m, one_turn_log())
    path = raw_log_path(root, None, m["id"])
    # Python rolls back inline in ``_append_synced``: its truncate's fsync is the next one.
    fail_fsync_once(
        monkeypatch,
        OSError("simulated append fsync failure"),
        OSError("simulated rollback failure"),
    )

    with pytest.raises(ExceptionGroup) as caught:
        await append_batch(
            persistence, m["id"], [{"type": "turn/start", "seq": 6, "time": 9, "data": {"turn": 2}}]
        )
    monkeypatch.undo()
    group = caught.value
    assert f'failed to roll back append to "{path}"' in group.message
    assert len(group.exceptions) == 2
    assert str(group.exceptions[0]) == "simulated append fsync failure"
    assert str(group.exceptions[1]) == "simulated rollback failure"


@upstream("rejects a mismatched header id before serving either session log")
@pytest.mark.asyncio
async def test_rejects_mismatched_header_id(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    a = meta("identity-a", "/same")
    b = meta("identity-b", "/same")
    await write_log(
        persistence, a, [{"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}]
    )
    await write_log(persistence, b, one_turn_log())

    a_path = raw_log_path(root, a["cwd"], a["id"])
    b_path = raw_log_path(root, b["cwd"], b["id"])

    def retarget(header: dict[str, Any]) -> None:
        header["id"] = b["id"]

    rewrite_header(a_path, retarget)
    before_a = read_bytes(a_path)
    before_b = read_bytes(b_path)

    with pytest.raises(
        Exception, match='requested id "identity-a" does not match header id "identity-b"'
    ):
        await read_all(persistence, a["id"])
    assert read_bytes(a_path) == before_a
    assert read_bytes(b_path) == before_b


@upstream("path-traversal session ids are neutralized (no escape from root)")
@pytest.mark.asyncio
async def test_path_traversal_ids_neutralized(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    m: SessionHeader = {
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId("../../etc/pwn"),
        "createdAt": 1,
        "isSeeded": False,
    }
    await write_log(persistence, m, one_turn_log())
    found = [os.path.join(d, name) for d, _, names in os.walk(root) for name in names]
    assert len(found) > 0
    assert all(p.startswith(root) for p in found)


@upstream("stat handles artifact removal and non-ENOENT failures after log discovery")
@pytest.mark.asyncio
async def test_stat_after_discovery_failures(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("stored-revision-race")
    await write_log(persistence, m, one_turn_log())
    path = raw_log_path(root, m.get("cwd"), m["id"])
    selected: list[Any] = [jsonl._ResolvedGeneration(path, SESSION_FORMAT_VERSION, path)]

    async def find_log(_id: SessionId) -> Any:
        return selected[0]

    monkeypatch.setattr(persistence, "_find_log", find_log)

    os.remove(path)
    assert await persistence.stat(m["id"]) is None

    # Node rejects with ERR_INVALID_ARG_VALUE; Python's equivalent is ValueError.
    invalid = f"{path}\0"
    selected[0] = jsonl._ResolvedGeneration(invalid, SESSION_FORMAT_VERSION, invalid)
    with pytest.raises(ValueError, match="null (byte|character)"):
        await persistence.stat(m["id"])


@upstream(
    "stat reports absence for an artifact vanishing before its identity stat and surfaces "
    "other faults"
)
@pytest.mark.asyncio
async def test_stat_identity_vanish_and_fault(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("stat-fault", "/work")
    await write_log(persistence, m, one_turn_log())
    target = raw_log_path(root, "/work", m["id"])
    failure: list[OSError] = [FileNotFoundError(errno.ENOENT, "ENOENT: vanished")]
    real = os.stat

    def failing(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path) == target:
            raise failure[0]
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", failing)
    assert await persistence.stat(m["id"]) is None
    failure[0] = PermissionError(errno.EACCES, "EACCES: denied")
    with pytest.raises(OSError, match="EACCES"):
        await persistence.stat(m["id"])


@upstream("an empty append batch is a no-op that does not materialize")
@pytest.mark.asyncio
async def test_empty_append_is_noop(root: str, persistence: JsonlSessionPersistence) -> None:
    m = meta("empty-batch", "/work")
    handle = await persistence.create(m)
    await handle.append([])
    assert not exists(raw_log_path(root, "/work", m["id"]))
    await handle.append(one_turn_log())
    assert [e["seq"] for e in (await handle.read()).events] == [0, 1, 2, 3, 4, 5]
    await handle.close()


@upstream("lists a pending id once when an artifact for the same id appears beneath it")
@pytest.mark.asyncio
async def test_lists_pending_id_once(root: str, persistence: JsonlSessionPersistence) -> None:
    m = meta("shadowed", "/w")
    creator = await persistence.create(m)
    os.makedirs(session_dir(root, "/w", m["id"]), exist_ok=True)
    lines = [stringify_json(to_header_line(m)), *(stringify_json(e) for e in one_turn_log())]
    write_text(raw_log_path(root, "/w", m["id"]), "\n".join(lines) + "\n", encoding="utf-8")

    entries = [s for s in await persistence.list() if s.header["id"] == m["id"]]
    assert len(entries) == 1
    assert entries[0].size_bytes is not None
    await creator.close()


@upstream("a read handle over an erased pending session fails loudly")
@pytest.mark.asyncio
async def test_read_handle_over_erased_pending(persistence: JsonlSessionPersistence) -> None:
    m = meta("erased-pending")
    creator = await persistence.create(m)
    reader = await persistence.open(m["id"], "read")
    assert list((await reader.read()).events) == []
    await creator.close()
    with pytest.raises(Exception, match="not found"):
        await reader.read()
    await reader.close()


@upstream("a read handle rejects a stored log that shrank below an observed prefix")
@pytest.mark.asyncio
async def test_read_handle_rejects_shrunk_log(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    m = meta("shrunk", "/work")
    await write_log(persistence, m, one_turn_log())
    reader = await persistence.open(m["id"], "read")
    try:
        assert len((await reader.read()).events) == 6
        lines = [stringify_json(to_header_line(m)), stringify_json(one_turn_log()[0])]
        write_text(raw_log_path(root, "/work", m["id"]), "\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(Exception, match="shrank below a previously observed prefix"):
            await reader.read()
    finally:
        await reader.close()


def _groups(error: BaseException | None) -> list[BaseExceptionGroup[BaseException]]:
    if not isinstance(error, BaseExceptionGroup):
        return []
    found: list[BaseExceptionGroup[BaseException]] = [error]
    for inner in error.exceptions:
        found.extend(_groups(inner))
    return found


@upstream("backend dispose aggregates open-handle close failures into one reported error")
@pytest.mark.asyncio
async def test_dispose_aggregates_close_failures(
    root: str, caplog: pytest.LogCaptureFixture
) -> None:
    ctx2 = Context()
    fiber = await ctx2.plugin(jsonl.apply, {"root": root, "compression": "none"})
    handle = await backend(ctx2).create(meta("dispose-fail"))
    failure = RuntimeError("close exploded")

    def close() -> asyncio.Future[None]:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        future.set_exception(failure)
        return future

    handle.close = close  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="deepseek_cordis"):
        await fiber.dispose()
    aggregates = [
        group
        for record in caplog.records
        if record.exc_info is not None
        for group in _groups(record.exc_info[1])
        if "session-persistence-jsonl dispose failed" in group.message
    ]
    assert aggregates
    assert list(aggregates[0].exceptions) == [failure]


@upstream("omits a listed artifact removed after discovery")
@pytest.mark.asyncio
async def test_omits_artifact_removed_after_discovery(
    persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("vanishing-snapshot")
    await write_log(persistence, m, one_turn_log())
    original = persistence._list_artifacts

    async def discover() -> Any:
        artifacts = await original()
        os.remove(artifacts[0][1])
        return artifacts

    monkeypatch.setattr(persistence, "_list_artifacts", discover)
    assert list(await persistence.list()) == []


@upstream("surfaces non-ENOENT stat failures during listing")
@pytest.mark.asyncio
async def test_surfaces_non_enoent_list_stat_failures(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def discover() -> Any:
        header = jsonl._GenerationHeader(meta("snapshot-stat-failure"), "current")
        return [(header, f"{root}\0snapshot-stat-failure")]

    monkeypatch.setattr(persistence, "_list_artifacts", discover)
    # Node says "null bytes"; CPython says "embedded null byte" (3.11) or "null character" (3.12+).
    with pytest.raises(ValueError, match="null (byte|character)"):
        await persistence.list()
