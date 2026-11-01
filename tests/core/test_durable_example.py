"""Public composition example exercises persistence, checkpointing, tools and resume."""

from __future__ import annotations

import runpy
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, cast

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("compression", ["none", "zstd"])
async def test_durable_example_resumes_complete_tool_history(
    tmp_path: Path, compression: str
) -> None:
    module = runpy.run_path(str(Path(__file__).parents[2] / "examples/durable_agent.py"))
    run = cast(Callable[..., Coroutine[Any, Any, list[dict[str, Any]]]], module["run"])
    first = await run(tmp_path, compression=compression)
    assert any(event["type"] == "tool/call" for event in first)
    assert any(event["type"] == "tool/result" for event in first)
    resumed = await run(tmp_path, resume=True, compression=compression)
    assert resumed[: len(first)] == first
    assert len([event for event in resumed if event["type"] == "turn/end"]) == 2
    assert all(
        event["data"]["reason"]["kind"] == "completed"
        for event in resumed
        if event["type"] == "turn/end"
    )
