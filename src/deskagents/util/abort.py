"""Cooperative cancellation: the Python counterpart of the web ``AbortSignal``.

Upstream passes an ``AbortSignal`` through agent turns, tool executions and model
requests. A caller aborts an :class:`AbortController`; work observes its signal by
reading :attr:`AbortSignal.aborted`, adding a listener, or awaiting
:meth:`AbortSignal.wait`. Aborting never cancels the task awaiting the work, so a
started tool body reaches its own cleanup.

The module functions also accept signals from other code that only expose an
``aborted`` attribute (and optionally ``reason``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar, cast

__all__ = [
    "AbortController",
    "AbortError",
    "AbortListener",
    "AbortSignal",
    "abort_error",
    "abortable",
    "is_aborted",
    "on_abort",
    "throw_if_aborted",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")
AbortListener = Callable[[object], object]


class AbortError(Exception):
    """An operation stopped because its signal aborted; ``reason`` is the abort reason."""

    def __init__(self, reason: object = None) -> None:
        super().__init__("This operation was aborted")
        self.reason = reason


class AbortSignal:
    """Abort state with listeners. Abort it through its :class:`AbortController`."""

    def __init__(self) -> None:
        self._aborted = False
        self._reason: object = None
        self._listeners: dict[int, AbortListener] = {}
        self._next_listener = 0

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> object:
        return self._reason if self.aborted else None

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise abort_error(self)

    def add_listener(self, listener: AbortListener) -> Callable[[], None]:
        """Call ``listener(reason)`` once when this signal aborts and return its remover.

        As with ``addEventListener("abort", ...)``, a listener added after the abort is
        never called; read :attr:`aborted` first.
        """
        key = self._next_listener
        self._next_listener += 1
        self._listeners[key] = listener

        def remove() -> None:
            self._listeners.pop(key, None)

        return remove

    async def wait(self) -> object:
        """Return the abort reason once this signal aborts."""
        if self.aborted:
            return self.reason
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()

        def settle(reason: object) -> None:
            if not future.done():
                future.set_result(reason)

        remove = self.add_listener(settle)
        try:
            return await future
        finally:
            remove()

    @staticmethod
    def any(signals: Iterable[object]) -> AbortSignal:
        """A signal that aborts with the reason of the first of ``signals`` to abort."""
        return _AnySignal(signals)

    @staticmethod
    def timeout(milliseconds: float) -> AbortSignal:
        """A signal that aborts with :class:`TimeoutError` after ``milliseconds``."""
        controller = AbortController()
        asyncio.get_running_loop().call_later(
            milliseconds / 1000, controller.abort, TimeoutError("The operation timed out")
        )
        return controller.signal

    def _abort(self, reason: object) -> None:
        if self._aborted:
            return
        self._aborted, self._reason = True, reason
        listeners = list(self._listeners.values())
        self._listeners.clear()
        for listener in listeners:
            try:
                result = listener(reason)
            except Exception:
                logger.exception("abort listener failed")
                continue
            if inspect.iscoroutine(result):
                result.close()
                logger.error("abort listener %r must be synchronous", listener)


class _AnySignal(AbortSignal):
    def __init__(self, sources: Iterable[object]) -> None:
        super().__init__()
        self._sources = tuple(sources)
        self._removers: list[Callable[[], None]] = []
        if not self._poll():
            self._removers = [on_abort(source, self._abort) for source in self._sources]

    def _poll(self) -> bool:
        for source in self._sources:
            if is_aborted(source):
                self._abort(getattr(source, "reason", None))
                return True
        return False

    @property
    def aborted(self) -> bool:
        return self._aborted or self._poll()

    def detach(self) -> None:
        """Stop relaying the sources; the fused signal keeps its current state."""
        removers, self._removers = self._removers, []
        for remove in removers:
            remove()

    def _abort(self, reason: object) -> None:
        super()._abort(reason)
        self.detach()


class AbortController:
    """Owner of one :class:`AbortSignal`."""

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: object = None) -> None:
        """Abort the signal once; the default reason is an :class:`AbortError`."""
        self.signal._abort(AbortError() if reason is None else reason)


def is_aborted(signal: object) -> bool:
    return bool(getattr(signal, "aborted", False))


def abort_error(signal: object) -> BaseException:
    """The exception that represents ``signal``'s abort: its reason if that is one."""
    reason = getattr(signal, "reason", None)
    return reason if isinstance(reason, BaseException) else AbortError(reason)


def throw_if_aborted(signal: object) -> None:
    if is_aborted(signal):
        raise abort_error(signal)


def on_abort(signal: object, listener: AbortListener) -> Callable[[], None]:
    """Add ``listener`` to a signal that supports listeners; returns its remover."""
    add = getattr(signal, "add_listener", None)
    if callable(add):
        remove = add(listener)
        if callable(remove):
            return cast(Callable[[], None], remove)
    return lambda: None


async def abortable(awaitable: Awaitable[T], signal: object) -> T:
    """Await ``awaitable`` unless ``signal`` aborts first.

    On abort the awaitable is cancelled, allowed to unwind, and :func:`abort_error`
    is raised. The awaiting task itself is never cancelled. A signal without
    ``wait`` is only checked before the await.
    """
    if is_aborted(signal):
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise abort_error(signal)
    wait = getattr(signal, "wait", None)
    if not callable(wait):
        return await awaitable
    work = asyncio.ensure_future(awaitable)
    aborted = asyncio.ensure_future(wait())
    try:
        await asyncio.wait({work, aborted}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        work.cancel()
        aborted.cancel()
        await asyncio.gather(work, aborted, return_exceptions=True)
        raise
    if work.done():
        aborted.cancel()
        await asyncio.gather(aborted, return_exceptions=True)
        return work.result()
    work.cancel()
    await asyncio.gather(work, return_exceptions=True)
    raise abort_error(signal)
