"""Write a minimal Python Session v4 artifact for cross-runtime verification."""

from __future__ import annotations

import argparse
import asyncio

from deepseek_cordis import Context

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
)
from deskagents.session import persistence_jsonl
from deskagents.session.persistence import SessionPersistence


async def write(root: str, compression: str, session_id: str) -> None:
    ctx = Context()
    try:
        await ctx.plugin(persistence_jsonl, {"root": root, "compression": compression})
        backend = ctx.get("sessionPersistence")
        assert isinstance(backend, SessionPersistence)
        header: SessionHeader = {
            "version": SESSION_FORMAT_VERSION,
            "id": SessionId(session_id),
            "createdAt": 1_733_000_000_000,
            "cwd": "/fixture",
            "isSeeded": False,
        }
        events: list[SessionEvent] = [
            {
                "type": "turn/start",
                "seq": 0,
                "time": 1_733_000_000_001,
                "data": {"turn": 1},
            },
            {
                "type": "turn/end",
                "seq": 1,
                "time": 1_733_000_000_002,
                "data": {"turn": 1, "reason": {"kind": "completed"}},
            },
        ]
        handle = await backend.create(header)
        await handle.append(events)
        await handle.close()
    finally:
        await ctx.fiber.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("compression", choices=("none", "zstd"))
    parser.add_argument("session_id")
    args = parser.parse_args()
    asyncio.run(write(args.root, args.compression, args.session_id))


if __name__ == "__main__":
    main()
