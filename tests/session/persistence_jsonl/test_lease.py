"""Ported from packages/session/session-persistence-jsonl/tests/lease.spec.ts.

Cross-process write-lock behaviour through fresh backend instances over one shared root
(``flock`` locks conflict between two descriptors even inside one process). Upstream's
``node:fs/promises`` and flock-addon mocks become monkeypatches of ``open``, ``os.stat``
and ``fcntl.flock`` as seen by ``deskagents.session.persistence_jsonl.lease``.

The last test replaces ``lease.two-process.e2e.ts``: a child Python process holds the
lease and the parent's write open is refused.
"""

from __future__ import annotations

import asyncio
import errno
import os
import sys
import textwrap
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import SESSION_FORMAT_VERSION, SessionHeader, SessionId
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import (
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionPersistence,
    SessionPersistenceNotFoundError,
)
from deskagents.session.persistence_jsonl import lease as lease_module
from deskagents.session.persistence_jsonl.format import session_dir
from deskagents.session.persistence_jsonl.lease import LEASE_FILENAME, SessionWriteLease
from deskagents.session.persistence_jsonl.storage import JsonlSessionHandle

SPEC = "packages/session/session-persistence-jsonl/tests/lease.spec.ts"
DESCRIBE = "cross-process write lock"
WIN32 = sys.platform == "win32"
posix_only = pytest.mark.skipif(WIN32, reason="POSIX flock lease behaviour")

EVENTS: list[dict[str, Any]] = [
    {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
    {"type": "turn/end", "seq": 1, "time": 2, "data": {"turn": 1, "reason": {"kind": "completed"}}},
]
TURN_2: dict[str, Any] = {"type": "turn/start", "seq": 2, "time": 3, "data": {"turn": 2}}


def meta(id_: str, cwd: str = "/work") -> SessionHeader:
    return {
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId(id_),
        "createdAt": 1_000,
        "cwd": cwd,
        "isSeeded": False,
    }


def lock_path(root: str, id_: str, cwd: str = "/work") -> str:
    return os.path.join(session_dir(root, cwd, SessionId(id_)), LEASE_FILENAME)


def exists(path: str) -> bool:
    return os.path.exists(path)


def corrupt_header(path: Path) -> None:
    """Garble the stored header line (a garbled tail would be recovered as torn)."""
    stored = path.read_text("utf-8")
    path.write_text(f"#{stored[1:]}", "utf-8")


def denied(syscall: str) -> OSError:
    return PermissionError(errno.EACCES, f"EACCES: injected {syscall} refusal")


Mount = Callable[[str], Awaitable[SessionPersistence]]


@pytest_asyncio.fixture
async def mount() -> AsyncIterator[Mount]:
    contexts: list[Context] = []

    async def mount(root: str) -> SessionPersistence:
        ctx = Context()
        contexts.append(ctx)
        await ctx.plugin(jsonl.apply, {"root": root, "compression": "none"})
        service = ctx.get("sessionPersistence")
        assert isinstance(service, SessionPersistence)
        return service

    yield mount
    for ctx in contexts:
        await ctx.fiber.dispose()


@pytest.fixture
def root(tmp_path: Path) -> str:
    return str(tmp_path)


def fail_release_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The next release does its real work, then reports failure (a close(2) EIO)."""
    original = SessionWriteLease.release

    async def release(self: SessionWriteLease) -> None:
        monkeypatch.setattr(SessionWriteLease, "release", original)
        await original(self)
        raise OSError(errno.EIO, "EIO: injected release failure")

    monkeypatch.setattr(SessionWriteLease, "release", release)


def patch_lock_stat(monkeypatch: pytest.MonkeyPatch, before: Callable[[str], None]) -> None:
    real_stat = os.stat

    def stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if str(path).endswith(LEASE_FILENAME):
            before(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)


async def materialized(backend: SessionPersistence, id_: str) -> None:
    writer = await backend.create(meta(id_))
    await writer.append(EVENTS)
    await writer.close()


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > excludes a second instance while the holder is live, "
    "and admits it after close"
)
@pytest.mark.asyncio
async def test_excludes_a_second_instance_while_live(root: str, mount: Mount) -> None:
    first = await mount(root)
    second = await mount(root)
    holder = await first.create(meta("excluded"))
    await holder.append(EVENTS)

    with pytest.raises(SessionAlreadyExistsError):
        await second.create(meta("excluded"))
    with pytest.raises(SessionAlreadyOwnedError):
        await second.open(SessionId("excluded"), "write")
    pending_winner = await first.create(meta("excluded-pending"))
    pending_loser = await second.create(meta("excluded-pending"))
    await pending_winner.append(EVENTS)
    with pytest.raises(SessionAlreadyOwnedError):
        await pending_loser.append(EVENTS)
    await pending_loser.close()
    await pending_winner.close()
    reader = await second.open(SessionId("excluded"), "read")
    assert [event["seq"] for event in (await reader.read()).events] == [0, 1]
    await reader.close()

    await holder.close()
    if not WIN32:
        assert exists(lock_path(root, "excluded"))
    reopened = await second.open(SessionId("excluded"), "write")
    await reopened.append([TURN_2])
    await reopened.close()


@posix_only
@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > removing the lock file forfeits a wedged holder: "
    "a fresh inode admits a successor"
)
@pytest.mark.asyncio
async def test_removing_the_lock_file_forfeits_a_wedged_holder(root: str, mount: Mount) -> None:
    first = await mount(root)
    second = await mount(root)
    wedged = await first.create(meta("wedged"))
    await wedged.append(EVENTS)

    os.remove(lock_path(root, "wedged"))
    successor = await second.open(SessionId("wedged"), "write")
    await successor.append([TURN_2])
    await successor.close()
    await wedged.close()


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > write-opening an absent session leaves no lock residue"
)
@pytest.mark.asyncio
async def test_write_opening_an_absent_session_leaves_no_residue(root: str, mount: Mount) -> None:
    backend = await mount(root)
    with pytest.raises(SessionPersistenceNotFoundError):
        await backend.open(SessionId("absent"), "write")
    assert not exists(os.path.join(root, LEASE_FILENAME))


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > write-opening an absent id under an existing project directory "
    "reports not-found"
)
@pytest.mark.asyncio
async def test_write_opening_an_absent_sibling_reports_not_found(root: str, mount: Mount) -> None:
    backend = await mount(root)
    await materialized(backend, "present-sibling")
    with pytest.raises(SessionPersistenceNotFoundError):
        await backend.open(SessionId("absent-sibling"), "write")


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > a never-materialized create leaves no filesystem footprint at all"
)
@pytest.mark.asyncio
async def test_a_never_materialized_create_leaves_no_footprint(root: str, mount: Mount) -> None:
    backend = await mount(root)
    handle = await backend.create(meta("erased"))
    directory = os.path.dirname(lock_path(root, "erased"))
    assert not exists(directory)
    await handle.close()
    assert not exists(directory)
    assert await backend.stat(SessionId("erased")) is None


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > materialization publishes the lock before the first log bytes "
    "and keeps it on the handle"
)
@pytest.mark.asyncio
async def test_materialization_publishes_and_keeps_the_lock(root: str, mount: Mount) -> None:
    first = await mount(root)
    second = await mount(root)
    creator = await first.create(meta("lazy-lock"))
    await creator.append(EVENTS)
    if not WIN32:
        assert exists(lock_path(root, "lazy-lock"))
    with pytest.raises(SessionAlreadyOwnedError):
        await second.open(SessionId("lazy-lock"), "write")
    await creator.append([TURN_2])
    await creator.close()
    reopened = await second.open(SessionId("lazy-lock"), "write")
    await reopened.close()


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > an explicitly flushed empty session takes the lock with its header"
)
@pytest.mark.asyncio
async def test_a_flushed_empty_session_takes_the_lock(root: str, mount: Mount) -> None:
    first = await mount(root)
    second = await mount(root)
    creator = await first.create(meta("flush-lock"))
    await creator.flush()
    with pytest.raises(SessionAlreadyOwnedError):
        await second.open(SessionId("flush-lock"), "write")
    await creator.close()


@posix_only
@pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > surfaces a filesystem refusal opening the lock file")
@pytest.mark.asyncio
async def test_surfaces_a_refusal_opening_the_lock_file(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "open-blocked")

    real_open = open
    refused = [True]

    def refusing_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if refused[0] and str(path).endswith(LEASE_FILENAME):
            refused[0] = False
            raise denied("open")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(lease_module, "open", refusing_open, raising=False)
    with pytest.raises(OSError, match=r"EACCES"):
        await backend.open(SessionId("open-blocked"), "write")


def patch_flock_once(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    import fcntl

    real_flock = fcntl.flock

    def flock(fd: int, operation: int) -> None:
        monkeypatch.setattr(fcntl, "flock", real_flock)
        raise error

    monkeypatch.setattr(fcntl, "flock", flock)


@posix_only
@pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > surfaces a non-contention flock failure")
@pytest.mark.asyncio
async def test_surfaces_a_non_contention_flock_failure(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "flock-blocked")

    patch_flock_once(monkeypatch, PermissionError(errno.EACCES, "EACCES: injected flock refusal"))
    with pytest.raises(OSError, match=r"EACCES"):
        await backend.open(SessionId("flock-blocked"), "write")


@posix_only
@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > maps the EWOULDBLOCK contention spelling to already-owned"
)
@pytest.mark.asyncio
async def test_maps_ewouldblock_to_already_owned(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "win-contended")

    patch_flock_once(
        monkeypatch, BlockingIOError(errno.EWOULDBLOCK, "EWOULDBLOCK: injected contention")
    )
    with pytest.raises(SessionAlreadyOwnedError):
        await backend.open(SessionId("win-contended"), "write")


@posix_only
@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > surfaces a lock-path stat refusal from the inode verification"
)
@pytest.mark.asyncio
async def test_surfaces_a_lock_path_stat_refusal(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "stat-blocked")

    refused = [True]

    def before(_path: str) -> None:
        if refused[0]:
            refused[0] = False
            raise denied("stat")

    patch_lock_stat(monkeypatch, before)
    with pytest.raises(OSError, match=r"EACCES"):
        await backend.open(SessionId("stat-blocked"), "write")


def swap_lock_on_stat(monkeypatch: pytest.MonkeyPatch, times: int) -> None:
    remaining = [times]

    def before(path: str) -> None:
        if remaining[0] > 0:
            remaining[0] -= 1
            os.remove(path)
            Path(path).write_bytes(b"")

    patch_lock_stat(monkeypatch, before)


@posix_only
@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > retries when the locked inode is no longer the lock path, "
    "and wins on a stable pass"
)
@pytest.mark.asyncio
async def test_retries_when_the_locked_inode_churns(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "churned")

    swap_lock_on_stat(monkeypatch, 1)
    reopened = await backend.open(SessionId("churned"), "write")
    await reopened.append([TURN_2])
    await reopened.close()


@posix_only
@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > retries when the lock path vanishes under the verify read"
)
@pytest.mark.asyncio
async def test_retries_when_the_lock_path_vanishes(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "vanished")

    dropped = [True]

    def before(path: str) -> None:
        if dropped[0]:
            dropped[0] = False
            os.remove(path)

    patch_lock_stat(monkeypatch, before)
    reopened = await backend.open(SessionId("vanished"), "write")
    await reopened.close()


@posix_only
@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > gives up as already-owned when the lock path never stabilizes"
)
@pytest.mark.asyncio
async def test_gives_up_when_the_lock_path_never_stabilizes(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "unstable")

    swap_lock_on_stat(monkeypatch, 3)
    with pytest.raises(SessionAlreadyOwnedError):
        await backend.open(SessionId("unstable"), "write")


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > a failing lock release still frees the in-process claim on close"
)
@pytest.mark.asyncio
async def test_a_failing_release_still_frees_the_claim(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    holder = await backend.create(meta("release-fails"))
    await holder.append(EVENTS)

    fail_release_once(monkeypatch)
    with pytest.raises(Exception, match=r"injected release failure"):
        await holder.close()
    reopened = await backend.open(SessionId("release-fails"), "write")
    await reopened.append([TURN_2])
    await reopened.close()


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > a write-open failure with a failing release aggregates both "
    "and frees the claim"
)
@pytest.mark.asyncio
async def test_a_write_open_failure_with_a_failing_release_aggregates(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    await materialized(backend, "open-and-release-fail")
    directory = os.path.dirname(lock_path(root, "open-and-release-fail"))
    log = next(name for name in os.listdir(directory) if name.endswith(".jsonl"))
    corrupt_header(Path(directory, log))

    fail_release_once(monkeypatch)
    with pytest.raises(ExceptionGroup) as caught:
        await backend.open(SessionId("open-and-release-fail"), "write")
    errors = caught.value.exceptions
    assert len(errors) == 2
    assert "corrupt" in str(errors[0]).lower()
    assert "injected release failure" in str(errors[1])
    with pytest.raises(Exception, match=r"(?i)corrupt"):
        await backend.open(SessionId("open-and-release-fail"), "write")


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > a drain failure and a release failure reject close as one AggregateError"
)
@pytest.mark.asyncio
async def test_a_drain_and_release_failure_reject_close_together(
    root: str, mount: Mount, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = await mount(root)
    holder = await backend.create(meta("drain-and-release-fail"))
    assert isinstance(holder, JsonlSessionHandle)
    await holder.append(EVENTS)

    real_persist = backend.persist_batch  # type: ignore[attr-defined]

    async def refuse_once(*args: Any) -> None:
        monkeypatch.setattr(backend, "persist_batch", real_persist)
        raise RuntimeError("injected drain refusal")

    monkeypatch.setattr(backend, "persist_batch", refuse_once)
    holder.enqueue_live(TURN_2, lambda _error: None)
    fail_release_once(monkeypatch)
    with pytest.raises(ExceptionGroup) as caught:
        await holder.close()
    joined = "\n".join(str(error) for error in caught.value.exceptions)
    assert "drain refusal" in joined
    assert joined.index("drain refusal") < joined.index("release failure")
    reopened = await backend.open(SessionId("drain-and-release-fail"), "write")
    await reopened.close()


@posix_only
@pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > release is idempotent and never removes the lock file")
@pytest.mark.asyncio
async def test_release_is_idempotent_and_keeps_the_lock_file(root: str) -> None:
    directory = os.path.join(root, "solo")
    lease = await SessionWriteLease.acquire(directory, SessionId("solo"))
    await lease.release()
    await lease.release()
    assert exists(os.path.join(directory, LEASE_FILENAME))
    successor = await SessionWriteLease.acquire(directory, SessionId("solo"))
    await successor.release()
    assert exists(os.path.join(directory, LEASE_FILENAME))


@pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > keeps distinct sessions independently lockable")
@pytest.mark.asyncio
async def test_keeps_distinct_sessions_independently_lockable(root: str, mount: Mount) -> None:
    backend = await mount(root)
    a = await backend.create(meta("indep-a"))
    b = await backend.create(meta("indep-b"))
    await a.append(EVENTS)
    await b.append(EVENTS)
    if not WIN32:
        names = os.listdir(os.path.dirname(lock_path(root, "indep-a")))
        assert [name for name in names if name == LEASE_FILENAME] == [LEASE_FILENAME]
    await a.close()
    await b.close()


_CHILD = textwrap.dedent(
    """
    import asyncio, sys
    from deskagents.core.session import SessionId
    from deskagents.session.persistence_jsonl.lease import SessionWriteLease

    async def main():
        lease = await SessionWriteLease.acquire(sys.argv[1], SessionId(sys.argv[2]))
        print("held", flush=True)
        sys.stdin.read()
        await lease.release()

    asyncio.run(main())
    """
)


@pytest.mark.asyncio
async def test_a_lease_held_by_another_process_refuses_a_write_open(
    root: str, mount: Mount
) -> None:
    """Replaces lease.two-process.e2e.ts: real cross-process exclusion and crash release."""
    backend = await mount(root)
    await materialized(backend, "two-process")
    directory = os.path.dirname(lock_path(root, "two-process"))
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CHILD,
        directory,
        "two-process",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert child.stdout is not None
        line = await asyncio.wait_for(child.stdout.readline(), timeout=30)
        assert line.strip() == b"held"
        with pytest.raises(SessionAlreadyOwnedError):
            await backend.open(SessionId("two-process"), "write")
        child.kill()
        await asyncio.wait_for(child.wait(), timeout=10)
        # The kernel lock dies with the holder: the id is admitted again.
        reopened = await backend.open(SessionId("two-process"), "write")
        await reopened.close()
    finally:
        if child.returncode is None:
            child.kill()
            await asyncio.wait_for(child.wait(), timeout=10)
