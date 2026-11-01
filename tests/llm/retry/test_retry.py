"""Ported from packages/llm/llm-retry/tests/retry.spec.ts and persistence.spec.ts."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core.session import SessionStore
from deskagents.llm.retry import RetryInternals, apply, local_delay, retry_policy_key
from deskagents.session.projection import SessionProjectionRegistry
from deskagents.util.abort import AbortController

RETRY_SPEC = "packages/llm/llm-retry/tests/retry.spec.ts"


def normal_policy(**overrides: Any) -> dict[str, Any]:
    return {
        "mode": "normal",
        "maxRetries": 2,
        "retryableCodes": ["RATE_LIMIT", "SERVER"],
        "initialDelayMs": 100,
        "maxDelayMs": 1_000,
        "jitterRatio": 0,
        **overrides,
    }


async def setup() -> tuple[Context, SessionStore]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(SessionProjectionRegistry)
    store = ctx.get("sessions")
    assert isinstance(store, SessionStore)
    return ctx, store


async def recover(
    ctx: Context,
    payload: dict[str, Any],
    downstream: Callable[[], Awaitable[Any]],
) -> Any:
    result = ctx.waterfall("agent/request-error", payload, next_=downstream)
    assert inspect.isawaitable(result)
    return await result


@pytest.mark.upstream(
    f"{RETRY_SPEC}::applies bounded exponential jitter and stops after the configured budget"
)
def test_local_delay_is_bounded_and_jittered() -> None:
    policy = normal_policy(jitterRatio=0.1, maxDelayMs=250)
    assert local_delay(policy, 1, 0) == 90
    assert local_delay(policy, 2, 0.5) == 200
    assert local_delay(policy, 3, 1) == 250


def test_policy_key_sorts_retryable_codes() -> None:
    left = retry_policy_key(normal_policy(retryableCodes=["SERVER", "RATE_LIMIT"]))
    right = retry_policy_key(normal_policy(retryableCodes=["RATE_LIMIT", "SERVER"]))
    assert left == right


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RETRY_SPEC}::delegates non-transient failures without scheduling a timer")
async def test_non_retryable_failure_delegates() -> None:
    ctx, store = await setup()
    calls = 0

    async def downstream() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"kind": "stop"}

    try:
        await apply(ctx, internals=RetryInternals(random=lambda: 0.5))
        session = store.create()
        result = await recover(
            ctx,
            {
                "agent": SimpleNamespace(session=session),
                "turn": 1,
                "step": 1,
                "provider": "test",
                "failure": {"message": "bad key", "code": "AUTH"},
                "retryPolicy": normal_policy(),
            },
            downstream,
        )
        assert result == {"kind": "stop"}
        assert calls == 1
        assert all(event["type"] != "llm/retry" for event in session.snapshot_events())
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RETRY_SPEC}::records the scheduled delay before retrying the request")
async def test_retry_is_durable_before_delay() -> None:
    ctx, store = await setup()
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    async def downstream() -> None:
        return None

    try:
        await apply(
            ctx,
            internals=RetryInternals(random=lambda: 0.5, sleep=sleep),
        )
        session = store.create()
        payload = {
            "agent": SimpleNamespace(session=session),
            "turn": 2,
            "step": 3,
            "provider": "test",
            "failure": {"message": "busy", "code": "RATE_LIMIT"},
            "retryPolicy": normal_policy(),
        }
        result = await recover(ctx, payload, downstream)
        assert result == {"kind": "retry"}
        assert waits == [0.1]
        assert [event["type"] for event in session.snapshot_events()] == [
            "llm/retry",
            "llm/retry-started",
        ]
        retry = session.snapshot_events()[0]["data"]
        assert retry["retry"] == 1
        assert retry["delayMs"] == 100
        assert session.snapshot_events()[1]["data"]["retryId"] == retry["retryId"]
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RETRY_SPEC}::starts a fresh retry budget when the next step selects another provider"
)
async def test_retry_state_advances_and_resets_at_step_start() -> None:
    ctx, store = await setup()

    async def no_wait(_seconds: float) -> None:
        return None

    async def downstream() -> None:
        return None

    try:
        await apply(
            ctx,
            internals=RetryInternals(random=lambda: 0.5, sleep=no_wait),
        )
        session = store.create()
        payload = {
            "agent": SimpleNamespace(session=session),
            "turn": 1,
            "step": 1,
            "provider": "test",
            "failure": {"message": "busy", "code": "SERVER"},
            "retryPolicy": normal_policy(),
        }
        assert await recover(ctx, payload, downstream) == {"kind": "retry"}
        assert await recover(ctx, payload, downstream) == {"kind": "retry"}
        assert await recover(ctx, payload, downstream) is None
        retries = [
            event["data"]["retry"]
            for event in session.snapshot_events()
            if event["type"] == "llm/retry"
        ]
        assert retries == [1, 2]

        session.append("step/start", {"turn": 1, "step": 2})
        payload["step"] = 2
        assert await recover(ctx, payload, downstream) == {"kind": "retry"}
        retries = [
            event["data"]["retry"]
            for event in session.snapshot_events()
            if event["type"] == "llm/retry"
        ]
        assert retries == [1, 2, 1]
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RETRY_SPEC}::uses a bounded provider Retry-After verbatim "
    "and delegates an over-cap instruction"
)
async def test_excessive_provider_delay_delegates() -> None:
    ctx, store = await setup()
    delegated = 0

    async def downstream() -> None:
        nonlocal delegated
        delegated += 1
        return None

    try:
        await apply(ctx)
        session = store.create()
        result = await recover(
            ctx,
            {
                "agent": SimpleNamespace(session=session),
                "turn": 1,
                "step": 1,
                "provider": "test",
                "failure": {
                    "message": "slow down",
                    "code": "RATE_LIMIT",
                    "providerRetryAfterMs": 5_000,
                },
                "retryPolicy": normal_policy(maxDelayMs=1_000),
            },
            downstream,
        )
        assert result is None
        assert delegated == 1
        assert session.seq == 0
    finally:
        await ctx.fiber.dispose()


def aborted_payload(session: Any, signal: object) -> dict[str, Any]:
    return {
        "agent": SimpleNamespace(session=session),
        "turn": 1,
        "step": 1,
        "provider": "test",
        "failure": {"message": "busy", "code": "RATE_LIMIT"},
        "retryPolicy": normal_policy(),
        "signal": signal,
    }


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RETRY_SPEC}::provider-routed retry policy > "
    "lets turn cancellation win during backoff without opening another step"
)
async def test_turn_cancellation_ends_the_backoff_without_a_retry() -> None:
    ctx, store = await setup()
    sleeping = asyncio.Event()

    async def sleep(seconds: float) -> None:
        sleeping.set()
        await asyncio.Event().wait()

    async def downstream() -> None:
        return None

    try:
        await apply(ctx, internals=RetryInternals(random=lambda: 0.5, sleep=sleep))
        session = store.create()
        controller = AbortController()
        pending = asyncio.ensure_future(
            recover(ctx, aborted_payload(session, controller.signal), downstream)
        )
        await sleeping.wait()
        controller.abort({"kind": "user"})
        assert await asyncio.wait_for(pending, 1) is None
        assert [event["type"] for event in session.snapshot_events()] == ["llm/retry"]
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RETRY_SPEC}::provider-routed retry policy > "
    "lets an earlier recovery listener cancel before %s retry policy runs"
)
async def test_an_aborted_request_schedules_no_retry() -> None:
    ctx, store = await setup()

    async def downstream() -> None:
        return None

    try:
        await apply(ctx, internals=RetryInternals(random=lambda: 0.5))
        session = store.create()
        controller = AbortController()
        controller.abort({"kind": "user"})
        result = await recover(ctx, aborted_payload(session, controller.signal), downstream)
        assert result is None
        assert session.seq == 0
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RETRY_SPEC}::rejects retry policy configured on the executor instead of a provider"
)
async def test_config_rejects_retry_policy() -> None:
    ctx, _store = await setup()
    try:
        with pytest.raises(
            ValueError,
            match="retryPolicy belongs under each provider configuration",
        ):
            await apply(ctx, {"retryPolicy": {}})
    finally:
        await ctx.fiber.dispose()
