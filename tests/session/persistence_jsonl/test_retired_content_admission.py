"""Ported from packages/session/session-persistence-jsonl/tests/retired-content-admission.spec.ts.

Only native V4 rows with ``compression: 'none'`` are ported here. The V3 rows check the
v3->v4 migration's diagnostics; Python refuses every v3 generation without migrating it.
Zstandard framing and admission are covered by the backend and contract suites.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import SessionId, SessionStore
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import (
    SessionAccess,
    SessionPersistence,
    SessionPersistenceCorruptionError,
)
from deskagents.session.persistence_jsonl.format import generation_log_path
from deskagents.util.values import stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/retired-content-admission.spec.ts"
DESCRIBE = "retired content in JSONL ($compression, $access)"

ID = SessionId("retired-content-admission")
ACCESS = pytest.mark.parametrize("access", ["read", "write"])
WRAPPER = {
    "type": "tool-result",
    "toolCallId": "retired-call",
    "content": [{"type": "text", "text": "result"}],
}


def write_log(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@dataclass
class Stored:
    persistence: SessionPersistence
    path: Path
    data: bytes
    event: dict[str, Any]


Store = Callable[[int, bool, bool], Awaitable[Stored]]


@pytest_asyncio.fixture
async def stored(tmp_path: Path) -> AsyncIterator[Store]:
    contexts: list[Context] = []

    async def store(version: int, opaque: bool, corrupt_prefix: bool) -> Stored:
        root = tmp_path / f"root-{len(contexts)}"
        event: dict[str, Any] = {
            "type": "future/opaque" if opaque else "user/message",
            "seq": 0,
            "time": 1,
            **({"ignorable": True} if opaque else {"surfaceOp": "append"}),
            "data": {
                "id": "user",
                "role": "user",
                "source": {"kind": "user"},
                "content": [WRAPPER],
            },
        }
        header = {
            "type": "session",
            "version": version,
            "id": ID,
            "createdAt": 1,
            "delegationDepth": 0,
            "isSeeded": False,
        }
        lines = [
            stringify_json(header) + "\n",
            *(["{invalid json}\n"] if corrupt_prefix else []),
            stringify_json(event) + "\n",
        ]
        data = "".join(lines).encode()
        path = Path(generation_log_path(str(root), None, ID, version, "none"))
        write_log(path, data)
        ctx = Context()
        contexts.append(ctx)
        await ctx.plugin(SessionStore)
        await ctx.plugin(jsonl.apply, {"root": str(root), "compression": "none"})
        service = ctx.get("sessionPersistence")
        assert isinstance(service, SessionPersistence)
        return Stored(service, path, data, event)

    try:
        yield store
    finally:
        for ctx in reversed(contexts):
            await ctx.fiber.dispose()


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > refuses V$version content with corruptPrefix=$corruptPrefix and "
    "preserves the only generation"
)
@ACCESS
@pytest.mark.parametrize("corrupt_prefix", [False, True])
@pytest.mark.asyncio
async def test_refuses_v4_retired_content(
    stored: Store, access: SessionAccess, corrupt_prefix: bool
) -> None:
    s = await stored(4, False, corrupt_prefix)
    with pytest.raises(SessionPersistenceCorruptionError, match=r"released tool-result wrapper"):
        handle = await s.persistence.open(ID, access)
        await handle.close()
    assert s.path.read_bytes() == s.data
    assert [n for n in os.listdir(s.path.parent) if n != "session.lock"] == [s.path.name]


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > preserves an unknown ignorable V%s payload without interpreting its "
    "content"
)
@ACCESS
@pytest.mark.asyncio
async def test_preserves_unknown_ignorable_v4_payload(stored: Store, access: SessionAccess) -> None:
    s = await stored(4, True, False)
    handle = await s.persistence.open(ID, access)
    try:
        assert list((await handle.read()).events) == [s.event]
    finally:
        await handle.close()
    assert s.path.read_bytes() == s.data
