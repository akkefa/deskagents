"""Runs the reusable handle contract (deskagents.testing) against the JSONL backend.

Upstream: packages/session/session-persistence-jsonl/tests/jsonl.spec.ts invokes
``runPersistenceContract('jsonl-none', …)`` from session-persistence/tests/contract.ts.
The cases are unmarked: their describe title is a template literal in contract.ts, so no
``jsonl.spec.ts::title`` source can name them literally.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import SessionId
from deskagents.session import persistence_jsonl
from deskagents.session.persistence import SessionPersistence
from deskagents.session.persistence_jsonl import log_path
from deskagents.testing import (
    ContractBackend,
    ContractBackendInstance,
    ContractCheck,
    ContractSkip,
    persistence_contract_cases,
)

CASES = persistence_contract_cases()


def append_bytes(path: Path, content: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(content)


ContractFactory = Callable[[], Awaitable[ContractBackend]]


@pytest_asyncio.fixture(params=["none", "zstd"])
async def make_backend(
    tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncIterator[ContractFactory]:
    """Each encoding gets a fresh backend root for every contract case."""
    contexts: list[Context] = []
    counter = 0
    compression = str(request.param)

    async def make() -> ContractBackend:
        nonlocal counter
        counter += 1
        root = tmp_path / f"store-{counter}"
        root.mkdir()

        async def instance() -> ContractBackendInstance:
            ctx = Context()
            contexts.append(ctx)
            await ctx.plugin(persistence_jsonl, {"root": str(root), "compression": compression})
            service = ctx.get("sessionPersistence")
            assert isinstance(service, SessionPersistence)
            return ContractBackendInstance(service, ctx.fiber.dispose)

        async def corrupt_tail(session_id: SessionId, cwd: str | None) -> None:
            # An uncommitted crash fragment which the write path must truncate.
            path = Path(log_path(str(root), cwd, session_id, compression))  # type: ignore[arg-type]
            content = (
                b"\x28\xb5" if compression == "zstd" else b'{"type":"assistant/chunk","seq":8,"ti'
            )
            await asyncio.to_thread(append_bytes, path, content)

        primary = await instance()
        return ContractBackend(
            primary.persistence, primary.dispose, reopen=instance, corrupt_tail=corrupt_tail
        )

    yield make
    for ctx in contexts:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("title", "check"), CASES, ids=[title for title, _ in CASES])
async def test_persistence_contract(
    title: str, check: ContractCheck, make_backend: ContractFactory
) -> None:
    try:
        await check(make_backend)
    except ContractSkip as skip:
        pytest.skip(str(skip))
