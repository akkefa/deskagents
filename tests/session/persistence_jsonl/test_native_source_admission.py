"""Ported from packages/session/session-persistence-jsonl/tests/native-source-admission.spec.ts.

Only the ``compression: 'none'`` rows of this upstream matrix are ported here.
Zstandard framing and admission are covered by the backend and contract suites.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import SESSION_FORMAT_VERSION, SessionId, SessionStore
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import (
    SessionAccess,
    SessionPersistence,
    SessionPersistenceCorruptionError,
)
from deskagents.session.persistence_jsonl.format import generation_log_path
from deskagents.util.values import stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/native-source-admission.spec.ts"
DESCRIBE = "native message-source admission ($compression, $access)"

ID = SessionId("native-source-admission")
ACCESS = pytest.mark.parametrize("access", ["read", "write"])


def write_log(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@dataclass
class Stored:
    persistence: SessionPersistence
    path: Path
    data: bytes
    event: dict[str, Any]


Store = Callable[..., Awaitable[Stored]]


@pytest_asyncio.fixture
async def stored(tmp_path: Path) -> AsyncIterator[Store]:
    contexts: list[Context] = []

    async def store(
        source: object, type_: str = "agent/inbox/spliced", ignorable: bool = False
    ) -> Stored:
        header = {
            "type": "session",
            "version": SESSION_FORMAT_VERSION,
            "id": ID,
            "createdAt": 1,
            "delegationDepth": 0,
            "isSeeded": False,
        }
        event: dict[str, Any] = {"type": type_, "seq": 0, "time": 1}
        if ignorable:
            event["ignorable"] = True
        event["data"] = {
            "target": "next-turn",
            "start": 0,
            "inserted": [
                {
                    "id": "input",
                    "role": "user",
                    "source": source,
                    "content": [{"type": "text", "text": "queued"}],
                }
            ],
        }
        data = "".join(stringify_json(value) + "\n" for value in (header, event)).encode()
        path = Path(generation_log_path(str(tmp_path), None, ID, SESSION_FORMAT_VERSION, "none"))
        write_log(path, data)
        ctx = Context()
        contexts.append(ctx)
        await ctx.plugin(SessionStore)
        await ctx.plugin(jsonl.apply, {"root": str(tmp_path), "compression": "none"})
        service = ctx.get("sessionPersistence")
        assert isinstance(service, SessionPersistence)
        return Stored(service, path, data, event)

    try:
        yield store
    finally:
        for ctx in reversed(contexts):
            await ctx.fiber.dispose()


async def open_and_close(persistence: SessionPersistence, access: SessionAccess) -> None:
    handle = await persistence.open(ID, access)
    await handle.close()


def siblings(path: Path) -> list[str]:
    return [name for name in os.listdir(path.parent) if name != "session.lock"]


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > refuses malformed queued source $source before exposing a handle"
)
@ACCESS
@pytest.mark.parametrize("source", [None, {}, [], {"kind": ""}, {"kind": 1}])
@pytest.mark.asyncio
async def test_refuses_malformed_queued_source(
    stored: Store, access: SessionAccess, source: object
) -> None:
    s = await stored(source)
    with pytest.raises(SessionPersistenceCorruptionError, match="producer-owned source kind"):
        await open_and_close(s.persistence, access)
    assert s.path.read_bytes() == s.data
    assert len(siblings(s.path)) == 1


@pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > still validates known queued events marked ignorable")
@ACCESS
@pytest.mark.asyncio
async def test_still_validates_known_ignorable(stored: Store, access: SessionAccess) -> None:
    s = await stored({}, "agent/inbox/spliced", True)
    with pytest.raises(Exception, match="producer-owned source kind"):
        await open_and_close(s.persistence, access)
    assert s.path.read_bytes() == s.data


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > preserves external attribution and own JSON metadata without a "
    "producer installed"
)
@ACCESS
@pytest.mark.asyncio
async def test_preserves_external_attribution(stored: Store, access: SessionAccess) -> None:
    source = json.loads(
        '{"kind":"external-producer","__proto__":{"retain":true},"metadata":{"count":1}}'
    )
    s = await stored(source)
    handle = await s.persistence.open(ID, access)
    try:
        assert (await handle.read()).events[0] == s.event
    finally:
        await handle.close()
    assert s.path.read_bytes() == s.data


@pytest.mark.upstream(f"{SPEC}::{DESCRIBE} > leaves unknown ignorable queued payloads opaque")
@ACCESS
@pytest.mark.asyncio
async def test_leaves_unknown_ignorable_opaque(stored: Store, access: SessionAccess) -> None:
    s = await stored(None, "external/inbox", True)
    handle = await s.persistence.open(ID, access)
    try:
        assert (await handle.read()).events[0] == s.event
    finally:
        await handle.close()
    assert s.path.read_bytes() == s.data
