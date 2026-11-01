"""Ported from packages/session/session-persistence-jsonl/tests/current-event-admission.spec.ts."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import SESSION_FORMAT_VERSION, SessionId, SessionStore
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import SessionPersistence
from deskagents.session.persistence_jsonl.format import generation_log_path, scan_log
from deskagents.util.values import stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/current-event-admission.spec.ts"
DESCRIBE = "current event admission at EOF"

ID = SessionId("current-admission")
HEADER: dict[str, Any] = {
    "type": "session",
    "version": SESSION_FORMAT_VERSION,
    "id": ID,
    "createdAt": 1000,
    "isSeeded": False,
    "delegationDepth": 0,
}
START: dict[str, Any] = {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
PREFIX = "\n".join(stringify_json(row) for row in (HEADER, START)) + "\n"
OBSOLETE_TYPES = ("tool/code-dispatch-start", "tool/code-dispatch")


def obsolete_event(type_: str, ignorable: bool = False) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": type_,
        "seq": 1,
        "time": 2,
        "data": {
            "rootCallId": "root",
            "parentCallId": "root",
            "subCallId": "child",
            "name": "read",
            "arguments": {},
        },
    }
    if ignorable:
        event["ignorable"] = True
    return event


def store_bytes(root: Path, data: bytes) -> Path:
    path = Path(generation_log_path(str(root), None, ID, SESSION_FORMAT_VERSION, "none"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def identity(path: Path) -> tuple[int, int]:
    info = os.stat(path)
    return info.st_dev, info.st_ino


@pytest_asyncio.fixture
async def persistence(tmp_path: Path) -> AsyncIterator[SessionPersistence]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(jsonl.apply, {"root": str(tmp_path), "compression": "none"})
    service = ctx.get("sessionPersistence")
    assert isinstance(service, SessionPersistence)
    try:
        yield service
    finally:
        await ctx.fiber.dispose()


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > refuses unknown required metadata %j without truncating a "
    "provider append"
)
@pytest.mark.parametrize("metadata", [{"surfaceOp": "append"}, {"sourceEventSeqs": []}])
@pytest.mark.asyncio
async def test_refuses_unknown_required_metadata(
    tmp_path: Path, persistence: SessionPersistence, metadata: dict[str, Any]
) -> None:
    event = {"type": "future/required", "seq": 0, "time": 1, "data": {}, **metadata}
    data = ("\n".join(stringify_json(row) for row in (HEADER, event)) + "\n").encode()
    scan = scan_log(data)
    assert scan.events == [event]
    assert scan.committed_bytes == len(data)
    path = store_bytes(tmp_path, data)
    source = identity(path)
    for access in ("read", "write"):

        async def operation(access: str = access) -> None:
            handle = await persistence.open(ID, access)  # type: ignore[arg-type]
            try:
                if access == "read":
                    await handle.read()
                else:
                    await handle.append(
                        [{"type": "turn/start", "seq": 1, "time": 2, "data": {"turn": 1}}]
                    )
            finally:
                await handle.close()

        with pytest.raises(Exception, match="unknown to this harness and not marked ignorable"):
            await operation()
        assert path.read_bytes() == data
        assert identity(path) == source


@pytest.mark.upstream(
    f"{SPEC}::{DESCRIBE} > retains ignorable %s through scanning and a provider append"
)
@pytest.mark.parametrize("type_", OBSOLETE_TYPES)
@pytest.mark.asyncio
async def test_retains_ignorable_obsolete_type(
    tmp_path: Path, persistence: SessionPersistence, type_: str
) -> None:
    event = obsolete_event(type_, True)
    data = (PREFIX + stringify_json(event) + "\n").encode()
    scan = scan_log(data)
    assert scan.events == [START, event]
    assert scan.committed_bytes == len(data)
    path = store_bytes(tmp_path, data)
    writer = await persistence.open(ID, "write")
    try:
        assert list((await writer.read()).events) == [START, event]
        await writer.append(
            [
                {
                    "type": "turn/end",
                    "seq": 2,
                    "time": 3,
                    "data": {"turn": 1, "reason": {"kind": "completed"}},
                }
            ]
        )
    finally:
        await writer.close()
    assert path.read_bytes()[: len(data)] == data
