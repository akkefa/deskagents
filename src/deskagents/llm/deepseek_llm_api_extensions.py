"""DeepSeek request-extension registry.

Upstream: packages/llm/deepseek-llm-api-extensions at ce7073d.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

from deepseek_cordis import Context, Service, current_context

from deskagents.util.values import deep_freeze, snapshot_json_value

logger = logging.getLogger(__name__)


class ExtensionProvider(Protocol):
    def prepare(
        self, request: Mapping[str, Any]
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any] | None] | None: ...


class PreparedExtensions:
    def __init__(
        self,
        fields: Mapping[str, Any],
        callbacks: list[Callable[[], object]],
    ) -> None:
        self.fields = fields
        self._callbacks = callbacks
        self._acceptance: asyncio.Task[None] | None = None

    async def accept(self) -> None:
        if self._acceptance is None:

            async def run() -> None:
                outcomes = await asyncio.gather(
                    *(_await_callback(callback) for callback in self._callbacks),
                    return_exceptions=True,
                )
                failures = [item for item in outcomes if isinstance(item, BaseException)]
                if failures:
                    raise ExceptionGroup(
                        "DeepSeek LLM API extension acceptance failed",
                        [item for item in failures if isinstance(item, Exception)],
                    )

            self._acceptance = asyncio.create_task(run())
        await self._acceptance


async def _await_callback(callback: Callable[[], object]) -> None:
    result = callback()
    if inspect.isawaitable(result):
        await result


class DeepSeekLlmApiExtensionRegistry(Service):
    name = "deepseekLlmApiExtensions"

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, config)
        self._providers: dict[str, ExtensionProvider] = {}
        self._warned: set[str] = set()

    def register(
        self,
        field: str,
        provider: ExtensionProvider,
        *,
        ctx: Context | None = None,
    ) -> Callable[[], object]:
        if not field or field.strip() != field:
            raise ValueError("DeepSeek extension field must be a non-blank trimmed string")
        owner = ctx or current_context() or self.ctx

        def setup() -> Callable[[], None]:
            if field in self._providers:
                raise ValueError(f'DeepSeek extension field "{field}" is already registered')
            self._providers[field] = provider

            def cleanup() -> None:
                self._providers.pop(field, None)

            return cleanup

        return owner.effect(setup, f"deepseekLlmApiExtensions.register({field})")

    async def prepare(self, request: Mapping[str, Any]) -> PreparedExtensions:
        fields: dict[str, Any] = {}
        callbacks: list[Callable[[], object]] = []
        for field, provider in tuple(self._providers.items()):
            try:
                result = provider.prepare(request)
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    continue
                detached = snapshot_json_value(result["value"])
                if detached is None and result["value"] is not None:
                    raise ValueError("extension value is not lossless JSON")
                fields[field] = deep_freeze(detached)
                accept = result.get("accept")
                if callable(accept):
                    callbacks.append(accept)
            except asyncio.CancelledError:
                raise
            except Exception:
                if field not in self._warned:
                    self._warned.add(field)
                    logger.warning(
                        'omitting failed DeepSeek extension field "%s"',
                        field,
                        exc_info=True,
                    )
        return PreparedExtensions(deep_freeze(fields), callbacks)


__all__ = [
    "DeepSeekLlmApiExtensionRegistry",
    "ExtensionProvider",
    "PreparedExtensions",
]
