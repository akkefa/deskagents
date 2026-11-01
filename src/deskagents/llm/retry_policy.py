"""Provider-owned request retry-policy validation and resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from deskagents.util.values import MAX_SAFE_INTEGER, deep_freeze

from .errors import EMPTY_RESPONSE_CODE

MAX_TIMER_DELAY_MS = 2_147_483_647
_DEFAULT_CODES = (EMPTY_RESPONSE_CODE, "RATE_LIMIT", "SERVER", "TIMEOUT", "TRANSPORT")
_POLICY_KEYS = frozenset({"mode", "maxRetries", "retryableCodes", "backoff"})
_BACKOFF_KEYS = frozenset({"initialDelayMs", "maxDelayMs", "jitterRatio"})


def _keys(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    for key in value:
        if key not in allowed:
            raise ValueError(f'{path}: unknown key "{key}"')


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _backoff(value: object, path: str) -> dict[str, int | float]:
    if value is None:
        config: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        config = value
        _keys(config, _BACKOFF_KEYS, path)
    else:
        raise ValueError(f"{path} must be an object")
    initial = config.get("initialDelayMs", 500)
    maximum = config.get("maxDelayMs", 10_000)
    jitter = config.get("jitterRatio", 0.1)
    if not _number(initial) or not 0 < initial <= MAX_TIMER_DELAY_MS:
        raise ValueError(
            f"{path}.initialDelayMs must be a positive finite number no greater than "
            f"{MAX_TIMER_DELAY_MS}"
        )
    if not _number(maximum) or not 0 < maximum <= MAX_TIMER_DELAY_MS:
        raise ValueError(
            f"{path}.maxDelayMs must be a positive finite number no greater than "
            f"{MAX_TIMER_DELAY_MS}"
        )
    if initial > maximum:
        raise ValueError(f"{path}.initialDelayMs must be less than or equal to maxDelayMs")
    if not _number(jitter) or not 0 <= jitter <= 1:
        raise ValueError(f"{path}.jitterRatio must be between 0 and 1")
    return {"initialDelayMs": initial, "maxDelayMs": maximum, "jitterRatio": jitter}


def resolve_retry_policy(config: Mapping[str, Any] | None, path: str) -> Mapping[str, Any]:
    """Validate, default, detach, and freeze one provider retry policy."""
    if config is None:
        return deep_freeze(
            {
                "mode": "normal",
                "maxRetries": 5,
                "retryableCodes": list(_DEFAULT_CODES),
                **_backoff(None, f"{path}.backoff"),
            }
        )
    _keys(config, _POLICY_KEYS, path)
    mode = config.get("mode")
    if mode == "always":
        return deep_freeze({"mode": "always", **_backoff(config.get("backoff"), f"{path}.backoff")})
    if mode != "normal":
        raise ValueError(f'{path}.mode must be "normal" or "always"')
    retries = config.get("maxRetries", 5)
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries < 0
        or retries > MAX_SAFE_INTEGER
    ):
        raise ValueError(f"{path}.maxRetries must be a non-negative safe integer")
    codes = config.get("retryableCodes", list(_DEFAULT_CODES))
    if not isinstance(codes, list | tuple) or not codes:
        raise ValueError(f"{path}.retryableCodes must not be empty")
    if any(not isinstance(code, str) or not code for code in codes):
        raise ValueError(f"{path}.retryableCodes must contain only non-empty strings")
    if len(set(codes)) != len(codes):
        raise ValueError(f"{path}.retryableCodes must not contain duplicates")
    return cast(
        Mapping[str, Any],
        deep_freeze(
            {
                "mode": "normal",
                "maxRetries": retries,
                "retryableCodes": list(codes),
                **_backoff(config.get("backoff"), f"{path}.backoff"),
            }
        ),
    )
