"""Contained Cordis notifications shared by the core packages.

Cordis ``emit`` stops delivery at the first listener that raises and rejects awaitable
results. Upstream's non-vetoing notifications (``agent/*``, ``tools/result``,
``session/*``, ``llm/adapters-updated``) instead take the listener snapshot from
``ctx.events.dispatch`` and contain each listener's synchronous and asynchronous
failure, so one broken observer cannot starve later ones or fail the emitter.

Failures are logged through the calling package's logger, as upstream logs through
each package's ``ctx.logger``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable

from deepseek_cordis import Context
from deepseek_cordis.events import EventFilter

_default_logger = logging.getLogger(__name__)
_observer_tasks: set[asyncio.Task[None]] = set()


def contain(returned: object, describe: str, log: logging.Logger | None = None) -> None:
    """Log, never raise, an awaitable listener result's failure."""
    if not inspect.isawaitable(returned):
        return
    log = log or _default_logger

    async def watch(awaitable: Awaitable[object]) -> None:
        try:
            await awaitable
        except Exception as error:
            log.warning("%s rejected: %s", describe, error)

    try:
        task = asyncio.get_running_loop().create_task(watch(returned))
    except RuntimeError:
        if inspect.iscoroutine(returned):
            returned.close()
        log.warning("%s returned an awaitable outside an event loop", describe)
        return
    _observer_tasks.add(task)
    task.add_done_callback(_observer_tasks.discard)


def emit_contained(
    ctx: Context,
    name: str,
    *args: object,
    filter_: EventFilter | None = None,
    describe: str | None = None,
    log: logging.Logger | None = None,
) -> None:
    """``ctx.emit`` with every listener's failure logged instead of raised."""
    label = describe or f"{name} listener"
    log = log or _default_logger
    for callback in ctx.events.dispatch("emit", name, args, filter_):
        try:
            with ctx.scope():
                contain(callback(*args), label, log)
        except Exception as error:
            log.warning("%s threw: %s", label, error)


__all__ = ["contain", "emit_contained"]
