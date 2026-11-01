"""Runs the reusable live-write-path contract (deskagents.testing) against the JSONL backend.

Upstream: packages/session/session-persistence-jsonl/tests/jsonl.spec.ts invokes
``runLiveWritePathContract('jsonl', LIVE_WRITE_BATCH_MAX_DELAY_MS, …)`` from
session-persistence/tests/live-write-contract.ts. The cases are unmarked: their describe
title is a template literal in live-write-contract.ts, so no ``jsonl.spec.ts::title``
source can name them literally.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import SessionStore
from deskagents.session import persistence_jsonl
from deskagents.session.persistence_jsonl.storage import LIVE_WRITE_BATCH_MAX_DELAY_S
from deskagents.testing import LiveWriteBackend, LiveWriteCheck, live_write_path_contract_cases

CASES = live_write_path_contract_cases()

LiveWriteFactory = Callable[[], Awaitable[LiveWriteBackend]]


@pytest_asyncio.fixture(params=["none", "zstd"])
async def make_backend(
    tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncIterator[LiveWriteFactory]:
    """Each call mounts SessionStore and the JSONL backend over its own empty root."""
    contexts: list[Context] = []
    counter = 0
    compression = str(request.param)

    async def make() -> LiveWriteBackend:
        nonlocal counter
        counter += 1
        root = tmp_path / f"live-{counter}"
        root.mkdir()

        async def mount() -> Context:
            ctx = Context()
            contexts.append(ctx)
            await ctx.plugin(SessionStore)
            await ctx.plugin(persistence_jsonl, {"root": str(root), "compression": compression})
            return ctx

        return LiveWriteBackend(await mount(), mount)

    yield make
    # Like upstream's afterEach: dispose whatever a failed check left mounted.
    for ctx in contexts:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("title", "check"), CASES, ids=[title for title, _ in CASES])
async def test_live_write_path_contract(
    title: str, check: LiveWriteCheck, make_backend: LiveWriteFactory
) -> None:
    await check(make_backend, LIVE_WRITE_BATCH_MAX_DELAY_S)
