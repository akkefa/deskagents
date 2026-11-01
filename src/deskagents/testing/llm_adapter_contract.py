"""Reusable contract for provider-neutral LLM adapters.

The contract is intentionally transport-free.  A provider package supplies a fresh runtime
with its adapter mounted over a mock transport, plus one canonical request.  The checks
exercise model identity, stream protocol, assembly, and one-shot prepared dispatch without
depending on pytest.
"""

from __future__ import annotations

import traceback
from collections import deque
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from deskagents.llm import BlockAssembler, LlmAdapter, LlmError, LlmRuntime
from deskagents.util.values import deep_freeze

__all__ = [
    "LLM_ADAPTER_CONTRACT_SUITE",
    "AdapterContractBackend",
    "AdapterContractFactory",
    "AdapterContractResult",
    "ScriptedAdapter",
    "llm_adapter_contract_cases",
    "run_llm_adapter_contract",
]

LLM_ADAPTER_CONTRACT_SUITE = "LlmAdapter contract"

Chunk = Mapping[str, Any]
Script = Sequence[Chunk] | BaseException


class ScriptedAdapter(LlmAdapter):
    """Deterministic adapter backed by queued chunk scripts.

    Every call consumes one script.  A script may be a chunk sequence or an exception to
    raise at the final adapter boundary.  Received requests are retained as detached,
    deeply frozen mappings for assertions.
    """

    def __init__(
        self,
        scripts: Sequence[Script] = (),
        *,
        models: Sequence[Mapping[str, Any]] | None = None,
        resolved: Mapping[str, Any] | None = None,
    ) -> None:
        self._scripts: deque[Script] = deque(scripts)
        self._models = tuple(models or ())
        self._resolved = resolved
        self.requests: list[Mapping[str, Any]] = []

    def enqueue(self, script: Script) -> None:
        """Append one future response or failure."""
        self._scripts.append(script)

    async def list_models(self, provider: str) -> Sequence[Mapping[str, Any]]:
        if self._models:
            return self._models
        return ({"provider": provider, "id": "scripted", "name": "Scripted"},)

    async def resolve_model(
        self,
        provider: str,
        model: str,
        signal: object = None,
    ) -> Mapping[str, Any]:
        del signal
        if self._resolved is not None:
            return self._resolved
        return {"provider": provider, "id": model, "name": model}

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        async def generate() -> AsyncIterable[Mapping[str, Any]]:
            self.requests.append(deep_freeze(dict(options)))
            if not self._scripts:
                raise RuntimeError("ScriptedAdapter has no queued response")
            script = self._scripts.popleft()
            if isinstance(script, BaseException):
                raise script
            for chunk in script:
                yield deep_freeze(dict(chunk))

        return generate()


@dataclass
class AdapterContractBackend:
    """One fresh, mounted adapter and its canonical request."""

    runtime: LlmRuntime
    adapter: LlmAdapter
    provider: str
    model: str
    options: Mapping[str, Any]
    dispose: Callable[[], Awaitable[None]]


AdapterContractFactory = Callable[[], Awaitable[AdapterContractBackend]]
AdapterContractCheck = Callable[[AdapterContractFactory], Awaitable[None]]


@dataclass(frozen=True)
class AdapterContractResult:
    title: str
    outcome: Literal["passed", "failed"]
    error: BaseException | None = None
    detail: str = ""


async def _model_identity(factory: AdapterContractFactory) -> None:
    backend = await factory()
    try:
        info = await backend.runtime.resolve_model_info(backend.provider, backend.model)
        assert info["provider"] == backend.provider
        assert info["id"] == backend.model
        assert isinstance(info["name"], str) and info["name"]
    finally:
        await backend.dispose()


async def _stream_protocol(factory: AdapterContractFactory) -> None:
    backend = await factory()
    try:
        chunks = [chunk async for chunk in backend.runtime.stream(backend.options)]
        assert chunks, "adapter stream must not be empty"
        finishes = [chunk for chunk in chunks if chunk.get("type") == "finish"]
        assert len(finishes) == 1, "adapter stream must contain exactly one finish chunk"
        assert chunks[-1].get("type") == "finish", "finish must be terminal"
        assembler = BlockAssembler()
        for chunk in chunks:
            assembler.push(dict(chunk))
        assembler.blocks()
    finally:
        await backend.dispose()


async def _prepared_call(factory: AdapterContractFactory) -> None:
    backend = await factory()
    try:
        prepared = await backend.runtime.prepare_call(
            {"provider": backend.provider, "model": backend.model}
        )
        options = {**backend.options, **prepared.config}
        chunks = [chunk async for chunk in prepared.stream(options)]
        assert chunks and chunks[-1].get("type") == "finish"
        try:
            prepared.stream(options)
        except LlmError as error:
            assert error.code == "INVALID_PREPARED_CALL"
        else:
            raise AssertionError("prepared calls must be single-use")
    finally:
        await backend.dispose()


_CASES: tuple[tuple[str, AdapterContractCheck], ...] = (
    ("resolves exact model identity", _model_identity),
    ("emits an assemblable terminal stream", _stream_protocol),
    ("dispatches prepared calls exactly once", _prepared_call),
)


def llm_adapter_contract_cases() -> tuple[tuple[str, AdapterContractCheck], ...]:
    """Return stable contract case titles and checks."""
    return _CASES


async def run_llm_adapter_contract(
    factory: AdapterContractFactory,
) -> list[AdapterContractResult]:
    """Run all adapter checks and collect outcomes."""
    results: list[AdapterContractResult] = []
    for title, check in _CASES:
        try:
            await check(factory)
        except Exception as error:
            results.append(
                AdapterContractResult(
                    title,
                    "failed",
                    error,
                    "".join(traceback.format_exception(error)),
                )
            )
        else:
            results.append(AdapterContractResult(title, "passed"))
    return results
