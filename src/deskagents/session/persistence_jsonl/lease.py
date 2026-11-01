"""Cross-process single-writer lease on one session directory.

Upstream: packages/session/session-persistence-jsonl/src/lease.ts

POSIX: an exclusive non-blocking ``flock`` on ``session.lock``, re-checked against the
path's inode so a lock on a deleted-and-recreated file never counts. Python
difference on Windows: upstream opens the lock file with an exclusive share mode through
a native call; Python takes ``msvcrt.locking`` on its first byte. Either way a second
writer (Python or ``dsh``) is refused.
"""

from __future__ import annotations

import asyncio
import errno
import os
import sys
from typing import BinaryIO

from deskagents.core.session import SessionId
from deskagents.session.persistence import SessionAlreadyOwnedError

LEASE_FILENAME = "session.lock"


def _acquire_posix(path: str, session_id: SessionId) -> BinaryIO:
    import fcntl

    for _ in range(3):
        handle = open(path, "wb")  # noqa: SIM115 - held for the lease lifetime
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise SessionAlreadyOwnedError(session_id) from None
                raise
            held = os.fstat(handle.fileno())
            try:
                current = os.stat(path)
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_ino, current.st_dev) == (
                held.st_ino,
                held.st_dev,
            ):
                return handle
        except BaseException:
            handle.close()
            raise
        handle.close()
    raise SessionAlreadyOwnedError(session_id)


def _acquire_windows(path: str, session_id: SessionId) -> BinaryIO:
    import msvcrt

    try:
        handle = open(path, "a+b")  # noqa: SIM115 - held for the lease lifetime
    except PermissionError:
        raise SessionAlreadyOwnedError(session_id) from None
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
    except OSError:
        handle.close()
        raise SessionAlreadyOwnedError(session_id) from None
    return handle


def _acquire(directory: str, session_id: SessionId) -> BinaryIO:
    os.makedirs(directory, mode=0o700, exist_ok=True)
    path = os.path.join(directory, LEASE_FILENAME)
    if sys.platform == "win32":
        return _acquire_windows(path, session_id)
    return _acquire_posix(path, session_id)


class SessionWriteLease:
    """One held write lease; release is idempotent."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._released = False

    @classmethod
    async def acquire(cls, directory: str, session_id: SessionId) -> SessionWriteLease:
        """Take the lease or raise ``SessionAlreadyOwnedError`` when another writer holds it."""
        return cls(await asyncio.to_thread(_acquire, directory, session_id))

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await asyncio.to_thread(self._handle.close)
