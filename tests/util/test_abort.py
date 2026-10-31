from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deskagents.util.abort import (
    AbortController,
    AbortError,
    AbortSignal,
    abortable,
    is_aborted,
    on_abort,
    throw_if_aborted,
)


def test_listeners_run_once_with_the_reason_and_can_be_removed() -> None:
    controller = AbortController()
    seen: list[object] = []
    controller.signal.add_listener(seen.append)
    remove = controller.signal.add_listener(lambda reason: seen.append("removed"))
    remove()
    assert controller.signal.reason is None
    controller.abort("stop")
    controller.abort("again")
    assert seen == ["stop"]
    assert controller.signal.aborted
    assert controller.signal.reason == "stop"
    late: list[object] = []
    controller.signal.add_listener(late.append)
    assert late == []


def test_a_failing_listener_does_not_stop_the_others(caplog: pytest.LogCaptureFixture) -> None:
    controller = AbortController()
    seen: list[object] = []

    def broken(reason: object) -> None:
        raise RuntimeError("listener failed")

    controller.signal.add_listener(broken)
    controller.signal.add_listener(seen.append)
    controller.abort("stop")
    assert seen == ["stop"]
    assert "abort listener failed" in caplog.text


def test_default_reason_and_throw_if_aborted() -> None:
    controller = AbortController()
    controller.signal.throw_if_aborted()
    controller.abort()
    assert isinstance(controller.signal.reason, AbortError)
    with pytest.raises(AbortError):
        controller.signal.throw_if_aborted()
    timeout = AbortController()
    timeout.abort(TimeoutError("late"))
    with pytest.raises(TimeoutError, match="late"):
        throw_if_aborted(timeout.signal)
    mapping = AbortController()
    mapping.abort({"kind": "user"})
    with pytest.raises(AbortError) as raised:
        mapping.signal.throw_if_aborted()
    assert raised.value.reason == {"kind": "user"}


def test_module_functions_accept_duck_typed_signals() -> None:
    plain = SimpleNamespace(aborted=False)
    assert not is_aborted(plain)
    remove = on_abort(plain, lambda reason: None)
    remove()
    plain.aborted = True
    with pytest.raises(AbortError):
        throw_if_aborted(plain)
    assert not is_aborted(None)


@pytest.mark.asyncio
async def test_wait_returns_the_reason() -> None:
    controller = AbortController()
    waiter = asyncio.ensure_future(controller.signal.wait())
    await asyncio.sleep(0)
    controller.abort("done")
    assert await waiter == "done"
    assert await controller.signal.wait() == "done"


@pytest.mark.asyncio
async def test_any_aborts_with_the_first_reason_and_polls_plain_signals() -> None:
    first, second = AbortController(), AbortController()
    fused = AbortSignal.any([first.signal, second.signal])
    seen: list[object] = []
    fused.add_listener(seen.append)
    second.abort("second")
    first.abort("first")
    assert seen == ["second"]
    assert fused.reason == "second"
    plain = SimpleNamespace(aborted=False, reason=None)
    polled = AbortSignal.any([plain])
    assert not polled.aborted
    plain.aborted, plain.reason = True, "plain"
    assert polled.aborted
    assert polled.reason == "plain"
    already = AbortController()
    already.abort("early")
    assert AbortSignal.any([already.signal]).reason == "early"


@pytest.mark.asyncio
async def test_timeout_aborts_with_timeout_error() -> None:
    signal = AbortSignal.timeout(1)
    reason = await asyncio.wait_for(signal.wait(), 1)
    assert isinstance(reason, TimeoutError)


@pytest.mark.asyncio
async def test_abortable_returns_or_cancels_the_work_without_cancelling_the_caller() -> None:
    controller = AbortController()

    async def value() -> str:
        return "value"

    assert await abortable(value(), controller.signal) == "value"
    unwound = asyncio.Event()

    async def forever() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            unwound.set()

    pending = asyncio.ensure_future(abortable(forever(), controller.signal))
    await asyncio.sleep(0)
    controller.abort({"kind": "user"})
    with pytest.raises(AbortError):
        await pending
    assert unwound.is_set()
    current = asyncio.current_task()
    assert current is not None and current.cancelling() == 0
    with pytest.raises(AbortError):
        await abortable(value(), controller.signal)


@pytest.mark.asyncio
async def test_abortable_without_wait_support_only_checks_before() -> None:
    async def value() -> str:
        return "value"

    assert await abortable(value(), SimpleNamespace(aborted=False)) == "value"
