"""Stable reads of one immutable generation file.

Upstream: packages/session/session-persistence-jsonl/src/generation.ts. Only the stable
read is ported: the rest of that file prepares and publishes migrated generations, which
waits for the v0–v3 migrations.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class JsonlPhysicalIdentity:
    """File identity used for revisions: device, inode, size and timestamps."""

    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> JsonlPhysicalIdentity:
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def token(self) -> str:
        return f"{self.dev}:{self.ino}:{self.size}:{self.mtime_ns}:{self.ctime_ns}"


@dataclass(frozen=True)
class StablePhysicalFile:
    bytes: bytes
    identity: JsonlPhysicalIdentity


def _read_stable(path: str) -> StablePhysicalFile:
    before = JsonlPhysicalIdentity.of(os.stat(path))
    for attempt in range(2):
        with open(path, "rb") as handle:
            data = handle.read()
        after = JsonlPhysicalIdentity.of(os.stat(path))
        if before == after:
            return StablePhysicalFile(data, after)
        if attempt == 1:
            # A writer kept appending: keep the prefix the first stat described.
            return StablePhysicalFile(data[: before.size], before)
        before = after
    raise AssertionError("unreachable")


async def read_stable_jsonl_file(path: str) -> StablePhysicalFile:
    """Read a file whose identity did not change during the read (or its stable prefix)."""
    return await asyncio.to_thread(_read_stable, path)
