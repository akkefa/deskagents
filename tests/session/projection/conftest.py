"""Shared fixtures for the session-projection ports: a root with the store and registry."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import SessionStore
from deskagents.session.projection import SessionProjectionRegistry


@pytest_asyncio.fixture
async def ctx() -> AsyncIterator[Context]:
    root = Context()
    await root.plugin(SessionStore)
    await root.plugin(SessionProjectionRegistry)
    yield root
    await root.fiber.dispose()


@pytest_asyncio.fixture
async def registry(ctx: Context) -> SessionProjectionRegistry:
    service = ctx.get("sessionProjections")
    assert isinstance(service, SessionProjectionRegistry)
    return service


@pytest_asyncio.fixture
async def store(ctx: Context) -> SessionStore:
    service = ctx.get("sessions")
    assert isinstance(service, SessionStore)
    return service
