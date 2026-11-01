"""Ported from packages/session/session-persistence-jsonl/tests/win32.spec.ts.

Upstream unit-tests its koffi ``kernel32`` bindings (``MoveFileExW`` publication,
staged directory creation, a named write-lock semaphore) against a mocked binding on
every host. Python has no such binding: Windows publication is ``os.rename`` (which
refuses an existing target), directories come from ``os.makedirs``, and the write lease
is ``msvcrt.locking`` on ``session.lock``. The publication case is ported against the
backend's own no-overwrite publish (any platform); the lock cases that describe
observable lease behaviour run on Windows only. The binding-specific cases are recorded
as not applicable in the compatibility catalog.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from deskagents.core.session import SessionId
from deskagents.session.persistence import SessionAlreadyOwnedError
from deskagents.session.persistence_jsonl import _publish_new
from deskagents.session.persistence_jsonl.lease import SessionWriteLease

SPEC = "packages/session/session-persistence-jsonl/tests/win32.spec.ts"


def lock_exists(directory: str) -> bool:
    return os.path.exists(os.path.join(directory, "session.lock"))


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows write lease")


@pytest.mark.upstream(
    f"{SPEC}::Windows durable namespace helpers > "
    "publishes a new file with write-through MoveFileExW semantics"
)
def test_publishes_a_new_file(tmp_path: Path) -> None:
    tmp = tmp_path / "log.tmp"
    final = tmp_path / "log.jsonl"
    tmp.write_text("content", "utf-8")

    _publish_new(str(tmp), str(final))
    assert not tmp.exists()
    assert final.read_text("utf-8") == "content"


@windows_only
@pytest.mark.upstream(
    f"{SPEC}::Windows write-lock semaphore > "
    "maps a held semaphore (wait timeout) to EBUSY and closes the probe handle"
)
@pytest.mark.asyncio
async def test_a_held_lease_refuses_and_closes_the_probe(tmp_path: Path) -> None:
    directory = str(tmp_path / "s")
    holder = await SessionWriteLease.acquire(directory, SessionId("held"))
    try:
        with pytest.raises(SessionAlreadyOwnedError):
            await SessionWriteLease.acquire(directory, SessionId("held"))
    finally:
        await holder.release()
    # The refused probe kept no handle: the next acquire wins.
    successor = await SessionWriteLease.acquire(directory, SessionId("held"))
    await successor.release()


@windows_only
@pytest.mark.upstream(
    f"{SPEC}::Windows write-lock semaphore > "
    "releases by restoring the count and closing, surfacing a failed release"
)
@pytest.mark.asyncio
async def test_release_restores_the_lease(tmp_path: Path) -> None:
    directory = str(tmp_path / "s")
    first = await SessionWriteLease.acquire(directory, SessionId("released"))
    await first.release()
    second = await SessionWriteLease.acquire(directory, SessionId("released"))
    await second.release()
    assert lock_exists(directory)
