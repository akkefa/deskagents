"""Ported from packages/session/session-persistence-jsonl/tests/native-developer-admission.spec.ts.

Only the ``compression: 'none'`` rows of this upstream matrix are ported here.
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

from deskagents.core.session import SESSION_FORMAT_VERSION, SessionId, SessionStore
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import (
    SessionAccess,
    SessionPersistence,
    SessionPersistenceCorruptionError,
)
from deskagents.session.persistence_jsonl.format import generation_log_path
from deskagents.util.values import stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/native-developer-admission.spec.ts"
DESCRIBE = "native developer admission ($compression, $access)"

ID = SessionId("native-developer-admission")
ACCESS = pytest.mark.parametrize("access", ["read", "write"])
COORDINATES = {"turn": 1, "step": 1}
DEVELOPER: dict[str, Any] = {
    "type": "developer/message",
    "surfaceOp": "append",
    "data": {
        **COORDINATES,
        "message": {
            "id": "developer",
            "role": "developer",
            "source": {"kind": "tool-registry"},
            "content": [],
            "extra": {"retained": True},
        },
    },
}


def developer_with(content: list[Any], **data: Any) -> dict[str, Any]:
    return {
        **DEVELOPER,
        "data": {
            **DEVELOPER["data"],
            **data,
            "message": {**DEVELOPER["data"]["message"], "content": content},
        },
    }


def write_log(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@dataclass
class Stored:
    persistence: SessionPersistence
    path: Path
    data: bytes
    events: list[dict[str, Any]]


Store = Callable[..., Awaitable[Stored]]


@pytest_asyncio.fixture
async def stored(tmp_path: Path) -> AsyncIterator[Store]:
    contexts: list[Context] = []

    async def store(event: dict[str, Any], corrupt_prefix: bool = False) -> Stored:
        root = tmp_path / f"root-{len(contexts)}"
        rows: list[dict[str, Any]] = [
            {"type": "turn/start", "data": {"turn": 1}},
            {"type": "step/start", "data": COORDINATES},
            {
                "type": "system/message",
                "surfaceOp": "append",
                "data": {
                    **COORDINATES,
                    "message": {
                        "id": "system",
                        "role": "system",
                        "source": {"kind": "system-prompt"},
                        "content": [{"type": "text", "text": "System"}],
                    },
                },
            },
            {
                "type": "request/header",
                "data": {
                    "reason": "initial",
                    "header": {
                        "config": {"provider": "test", "model": "test"},
                        "tools": [
                            {"name": "search", "description": "Historical", "parameters": {}}
                        ],
                    },
                },
            },
            event,
        ]
        events = [{**row, "seq": seq, "time": seq + 1} for seq, row in enumerate(rows)]
        header = {
            "type": "session",
            "version": SESSION_FORMAT_VERSION,
            "id": ID,
            "createdAt": 1,
            "delegationDepth": 0,
            "isSeeded": False,
        }
        lines = [stringify_json(value) + "\n" for value in (header, *events)]
        if corrupt_prefix:
            lines.insert(len(lines) - 1, "{invalid json}\n")
        data = "".join(lines).encode()
        path = Path(generation_log_path(str(root), None, ID, SESSION_FORMAT_VERSION, "none"))
        write_log(path, data)
        ctx = Context()
        contexts.append(ctx)
        await ctx.plugin(SessionStore)
        await ctx.plugin(jsonl.apply, {"root": str(root), "compression": "none"})
        service = ctx.get("sessionPersistence")
        assert isinstance(service, SessionPersistence)
        return Stored(service, path, data, events)

    try:
        yield store
    finally:
        for ctx in reversed(contexts):
            await ctx.fiber.dispose()


async def open_and_close(persistence: SessionPersistence, access: SessionAccess) -> None:
    handle = await persistence.open(ID, access)
    await handle.close()


async def read_all(persistence: SessionPersistence, access: SessionAccess) -> list[Any]:
    handle = await persistence.open(ID, access)
    try:
        return list((await handle.read()).events)
    finally:
        await handle.close()


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > preserves an empty developer node and additional metadata on a "
    "valid open step (ignorable: %s)"
)
@ACCESS
@pytest.mark.parametrize("ignorable", [False, True])
@pytest.mark.asyncio
async def test_preserves_empty_developer_node(
    stored: Store, access: SessionAccess, ignorable: bool
) -> None:
    s = await stored({**DEVELOPER, **({"ignorable": True} if ignorable else {})})
    assert await read_all(s.persistence, access) == s.events
    assert s.path.read_bytes() == s.data


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > reads name-only additions bound to their historical header without "
    "copying definitions"
)
@ACCESS
@pytest.mark.asyncio
async def test_reads_name_only_additions(stored: Store, access: SessionAccess) -> None:
    event = developer_with(
        [
            {"type": "tool-removal", "toolName": "old"},
            {"type": "tool-addition", "toolName": "search"},
        ],
        headerSeq=3,
    )
    s = await stored(event)
    assert await read_all(s.persistence, access) == s.events
    assert s.path.read_bytes() == s.data


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > refuses invalid historical schema binding %# without rewriting the "
    "generation"
)
@ACCESS
@pytest.mark.parametrize(
    ("header_seq", "tool_name", "error"),
    [
        (0, "search", r"earlier request/header"),
        (4, "search", r"must name an earlier event"),
        (3, "absent", r"exactly one tool"),
    ],
)
@pytest.mark.asyncio
async def test_refuses_invalid_schema_binding(
    stored: Store, access: SessionAccess, header_seq: int, tool_name: str, error: str
) -> None:
    event = developer_with([{"type": "tool-addition", "toolName": tool_name}], headerSeq=header_seq)
    s = await stored(event)
    with pytest.raises(Exception, match=error):
        await open_and_close(s.persistence, access)
    assert s.path.read_bytes() == s.data


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > refuses malformed known ignorable developer data even after a "
    "corrupt row"
)
@ACCESS
@pytest.mark.asyncio
async def test_refuses_malformed_ignorable_after_corrupt_row(
    stored: Store, access: SessionAccess
) -> None:
    invalid = {
        **developer_with([{"type": "tool-addition", "toolName": "search", "tool": None}]),
        "ignorable": True,
    }
    s = await stored(invalid, True)
    with pytest.raises(Exception, match="omit inline tool definitions"):
        await open_and_close(s.persistence, access)
    assert s.path.read_bytes() == s.data


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > refuses a developer event with $name without rewriting the generation"
)
@ACCESS
@pytest.mark.parametrize(
    ("event", "error"),
    [
        pytest.param(
            {**DEVELOPER, "data": {**DEVELOPER["data"], "step": 2}},
            r"open.*step",
            id="mismatched step",
        ),
        pytest.param(
            {
                **DEVELOPER,
                "surfaceOp": {"op": "replace", "startSeq": 2, "endSeq": 2},
                "sourceEventSeqs": [2],
            },
            r"protected system head",
            id="protected system head",
        ),
    ],
)
@pytest.mark.asyncio
async def test_refuses_developer_event(
    stored: Store, access: SessionAccess, event: dict[str, Any], error: str
) -> None:
    s = await stored(event)
    with pytest.raises(SessionPersistenceCorruptionError, match=error):
        await open_and_close(s.persistence, access)
    assert s.path.read_bytes() == s.data
    assert len([n for n in os.listdir(s.path.parent) if n != "session.lock"]) == 1
