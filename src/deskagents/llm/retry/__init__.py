"""Provider-routed model-request retry policy.

Upstream: packages/llm/llm-retry (``@deepseek-ai/dsh-llm-retry``) at ce7073d.

The listener is installed before the Phase 5 agent package exists.  Its event payload is
therefore structural: ``agent`` must expose ``session``, while the remaining fields follow
upstream's ``agent/request-error`` payload.  A backoff ends early, without retrying, when
the payload's ``signal`` aborts or the plugin is disposed; disposal drains active waits.
"""

from __future__ import annotations

import asyncio
import json
import math
import random as random_module
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, NewType, Protocol, cast

from deepseek_cordis import Context
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from deskagents.core.session import Session
from deskagents.session.projection import (
    ProjectionDefinition,
    ProjectionHandle,
    SessionProjectionRegistry,
)
from deskagents.util.abort import is_aborted

__all__ = [
    "LlmRetryEventData",
    "LlmRetryStartedEventData",
    "RequestErrorAction",
    "RetryId",
    "RetryInternals",
    "RetryStateEntry",
    "apply",
    "local_delay",
    "retry_policy_key",
    "retry_state_key",
]

RetryId = NewType("RetryId", str)
"""Identity shared by every scheduled retry in one step's retry series."""

RequestErrorAction = Mapping[str, Any] | None
"""Recovery decision; ``{"kind": "retry"}`` asks the Phase 5 loop to try again."""


class RetryStateEntry(TypedDict):
    """Durable retry cursor for one provider and resolved policy."""

    retry: int
    retryId: str


RetryState = dict[str, RetryStateEntry]
_RETRY_STATE_SCHEMA = TypeAdapter(RetryState)

LlmRetryEventData = Mapping[str, Any]
LlmRetryStartedEventData = Mapping[str, Any]


class _Agent(Protocol):
    session: Session


class _ProjectionRegistration(Protocol):
    def __await__(self) -> Any: ...


Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RetryInternals:
    """Non-serializable timing hooks used by deterministic tests."""

    random: Callable[[], float] = random_module.random
    sleep: Sleep = asyncio.sleep


def _policy_number(policy: Mapping[str, Any], key: str) -> float:
    value = policy[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"retry policy {key} must be numeric")
    return float(value)


def local_delay(policy: Mapping[str, Any], retry: int, sample: float) -> float:
    """Return bounded exponential backoff in milliseconds."""
    initial = _policy_number(policy, "initialDelayMs")
    maximum = _policy_number(policy, "maxDelayMs")
    jitter_ratio = _policy_number(policy, "jitterRatio")
    exponent = min(retry - 1, 1024)
    exponential = min(initial * 2**exponent, maximum)
    jitter = 1 - jitter_ratio + 2 * jitter_ratio * sample
    bounded: float = min(exponential * jitter, maximum)
    return bounded


def retry_policy_key(policy: Mapping[str, Any]) -> str:
    """Stable key which resets durable retry state when provider policy changes."""
    mode = policy.get("mode")
    if mode == "always":
        parts: list[Any] = [
            mode,
            policy["initialDelayMs"],
            policy["maxDelayMs"],
            policy["jitterRatio"],
        ]
    else:
        parts = [
            mode,
            policy["maxRetries"],
            sorted(cast(list[str], policy["retryableCodes"])),
            policy["initialDelayMs"],
            policy["maxDelayMs"],
            policy["jitterRatio"],
        ]
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def retry_state_key(provider: str, policy_key: str) -> str:
    """Projection key for one provider-policy pair."""
    return json.dumps([provider, policy_key], ensure_ascii=False, separators=(",", ":"))


def _projection_definition() -> ProjectionDefinition[RetryState]:
    def init(_header: object, _inherited: int) -> RetryState:
        return {}

    def fold(state: RetryState, event: Mapping[str, Any]) -> RetryState:
        event_type = event.get("type")
        if event_type in {"step/start", "turn/end"}:
            return {}
        if event_type != "llm/retry":
            return state
        data = cast(Mapping[str, Any], event["data"])
        key = retry_state_key(cast(str, data["provider"]), cast(str, data["policyKey"]))
        retry = cast(int, data["retry"])
        retry_id = cast(str, data["retryId"])
        previous = state.get(key)
        if previous == {"retry": retry, "retryId": retry_id}:
            return state
        return {**state, key: {"retry": retry, "retryId": retry_id}}

    return ProjectionDefinition(
        key="llmRetry",
        state_version=1,
        state_schema=_RETRY_STATE_SCHEMA,
        init=init,
        apply=fold,
    )


async def _delay_or_dispose(
    delay_ms: float,
    disposed: asyncio.Event,
    sleep: Sleep,
    signal: object = None,
) -> bool:
    """Whether the full delay elapsed before disposal or the request's abort."""
    if disposed.is_set() or is_aborted(signal):
        return False

    async def wait_disposed() -> None:
        await disposed.wait()

    sleep_task: asyncio.Future[None] = asyncio.ensure_future(sleep(delay_ms / 1000))
    stops: list[asyncio.Future[Any]] = [asyncio.create_task(wait_disposed())]
    wait_aborted = getattr(signal, "wait", None)
    if callable(wait_aborted):
        stops.append(asyncio.ensure_future(wait_aborted()))
    try:
        done, _pending = await asyncio.wait(
            {sleep_task, *stops}, return_when=asyncio.FIRST_COMPLETED
        )
        if any(stop in done for stop in stops):
            return False
        await sleep_task
        return True
    finally:
        for task in (sleep_task, *stops):
            if not task.done():
                task.cancel()
        await asyncio.gather(sleep_task, *stops, return_exceptions=True)


async def _settle_downstream(
    next_: Callable[[], Awaitable[RequestErrorAction]],
) -> tuple[RequestErrorAction, BaseException | None]:
    try:
        return await next_(), None
    except BaseException as error:
        return None, error


def _valid_provider_delay(failure: Mapping[str, Any]) -> float | None:
    value = failure.get("providerRetryAfterMs")
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    ):
        return float(value)
    return None


async def apply(
    ctx: Context,
    config: Mapping[str, Any] | None = None,
    *,
    internals: RetryInternals | None = None,
) -> None:
    """Install durable, provider-owned request recovery."""
    config = {} if config is None else config
    if config:
        key = next(iter(config))
        if key == "retryPolicy":
            raise ValueError("llm-retry: retryPolicy belongs under each provider configuration")
        raise ValueError(f'llm-retry: unknown key "{key}"')

    projections = ctx.get("sessionProjections")
    if not isinstance(projections, SessionProjectionRegistry):
        raise RuntimeError("llm-retry requires the sessionProjections service")
    projection: ProjectionHandle[RetryState] = await projections.register(
        _projection_definition(), ctx=ctx
    )
    hooks = internals or RetryInternals()
    disposed = asyncio.Event()
    active: set[asyncio.Task[RequestErrorAction]] = set()

    async def recover(
        payload: Mapping[str, Any],
        next_: Callable[[], Awaitable[RequestErrorAction]],
    ) -> RequestErrorAction:
        policy_value = payload.get("retryPolicy")
        if not isinstance(policy_value, Mapping):
            return await next_()
        policy = cast(Mapping[str, Any], policy_value)
        signal = payload.get("signal")
        if disposed.is_set() or is_aborted(signal):
            return None

        failure = cast(Mapping[str, Any], payload["failure"])
        mode = policy.get("mode")
        if mode == "always":
            downstream, downstream_error = await _settle_downstream(next_)
            if disposed.is_set() or is_aborted(signal):
                return None
            if downstream_error is None and downstream is not None:
                if downstream.get("kind") == "retry":
                    return downstream
        else:
            retryable = cast(list[str] | tuple[str, ...], policy["retryableCodes"])
            if failure.get("code") not in retryable:
                return await next_()

        agent = cast(_Agent, payload["agent"])
        provider = cast(str, payload["provider"])
        policy_key = retry_policy_key(policy)
        state = projection.state_of(agent.session)
        previous = state.get(retry_state_key(provider, policy_key))
        previous_retry = previous["retry"] if previous is not None else 0
        if mode == "normal" and previous_retry >= cast(int, policy["maxRetries"]):
            return await next_()

        retry = previous_retry + 1
        retry_id = (
            RetryId(previous["retryId"]) if previous is not None else RetryId(str(uuid.uuid4()))
        )
        provider_delay = _valid_provider_delay(failure)
        max_delay = _policy_number(policy, "maxDelayMs")
        if provider_delay is not None and provider_delay <= max_delay:
            delay_ms = provider_delay
        elif provider_delay is not None and mode == "normal":
            return await next_()
        else:
            delay_ms = local_delay(policy, retry, hooks.random())

        turn = cast(int, payload["turn"])
        step = cast(int, payload["step"])
        event_data: dict[str, Any] = {
            "retryId": retry_id,
            "turn": turn,
            "step": step,
            "provider": provider,
            "mode": mode,
            "policyKey": policy_key,
            "retry": retry,
            "delayMs": delay_ms,
            "failure": failure,
        }
        if mode == "normal":
            event_data["maxRetries"] = policy["maxRetries"]
        agent.session.append("llm/retry", event_data)
        if not await _delay_or_dispose(delay_ms, disposed, hooks.sleep, signal):
            return None
        agent.session.append(
            "llm/retry-started",
            {"retryId": retry_id, "turn": turn, "step": step, "retry": retry},
        )
        return {"kind": "retry"}

    def listener(
        payload: Mapping[str, Any],
        next_: Callable[[], Awaitable[RequestErrorAction]],
    ) -> Awaitable[RequestErrorAction]:
        if disposed.is_set():
            return asyncio.sleep(0, result=None)
        task = asyncio.create_task(recover(payload, next_))
        active.add(task)
        task.add_done_callback(active.discard)
        return task

    dispose_listener = ctx.on("agent/request-error", listener)

    async def cleanup() -> None:
        dispose_listener()
        disposed.set()
        if active:
            await asyncio.gather(*tuple(active), return_exceptions=True)

    ctx.effect(lambda: cleanup, "llm-retry: cancel and drain active recovery")
