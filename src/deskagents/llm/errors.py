"""Provider-neutral LLM errors and classifiers.

Faithful to pinned ``packages/llm/llm/src/error.ts`` and
``adapter-failure.ts``. Attachment-specific image-offload production is outside
core (the attachment plugin is not ported), but its stable code and failure fact
remain understood for wire compatibility.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, cast

from deskagents.util.values import deep_freeze, is_safe_integer

CONTEXT_WINDOW_EXCEEDED_CODE = "CONTEXT_WINDOW_EXCEEDED"
QUOTA_EXCEEDED_CODE = "QUOTA"
ACCOUNT_QUOTA_EXCEEDED_CODE = "ACCOUNT_QUOTA"
EMPTY_RESPONSE_CODE = "EMPTY_RESPONSE"
INVALID_CREDENTIAL_CODE = "INVALID_CREDENTIAL"
IMAGE_OFFLOAD_REQUIRED_CODE = "IMAGE_OFFLOAD_REQUIRED"


class HarnessError(Exception):
    """Harness failure carrying a stable machine-routable code and cause."""

    def __init__(self, message: str, code: str, *, cause: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class LlmError(HarnessError):
    """LLM failure with a detached immutable provider-neutral payload."""

    def __init__(
        self,
        message: str,
        code: str,
        *,
        cause: object = None,
        status: int | None = None,
        provider_retry_after_ms: int | float | None = None,
        request_id: str | None = None,
        offload_images: int | None = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("LlmError message must be a non-empty string")
        if not isinstance(code, str) or not code:
            raise ValueError("LlmError code must be a non-empty string")
        if status is not None and (
            isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599
        ):
            raise ValueError("LlmError status must be an integer from 100 through 599")
        if provider_retry_after_ms is not None and (
            isinstance(provider_retry_after_ms, bool)
            or not isinstance(provider_retry_after_ms, int | float)
            or not math.isfinite(provider_retry_after_ms)
            or provider_retry_after_ms <= 0
        ):
            raise ValueError("LlmError provider_retry_after_ms must be a positive finite number")
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("LlmError request_id must be a non-empty string")
        if offload_images is not None and (
            not is_safe_integer(offload_images) or offload_images <= 0
        ):
            raise ValueError("LlmError offload_images must be a positive safe integer")
        super().__init__(message, code, cause=cause)
        facts: dict[str, Any] = {"message": message, "code": code}
        if status is not None:
            facts["status"] = status
        if provider_retry_after_ms is not None:
            facts["providerRetryAfterMs"] = provider_retry_after_ms
        if request_id is not None:
            facts["requestId"] = request_id
        if offload_images is not None:
            facts["offloadImages"] = offload_images
        self.failure: Mapping[str, Any] = deep_freeze(facts)


_STRUCTURED_CONTEXT = re.compile(
    r"(?:^|[^a-z0-9])context[\s_-](?:length|window)[\s_-]"
    r"(?:exceed(?:ed|s)?|overflow(?:ed)?|limit[\s_-]exceeded)(?:$|[^a-z0-9])",
    re.I,
)
_TOO_LARGE = re.compile(
    r"\b(?:request|prompt|input|messages?)\s+(?:is\s+|are\s+)?too\s+(?:large|long)"
    r"\s+for\s+(?:(?:this|the)\s+)?(?:model(?:'s)?\s+)?context(?:\s+window)?\b",
    re.I,
)
_EXCEEDS_CONTEXT = re.compile(
    r"\b(?:input|prompt|request|messages?)\b.{0,40}"
    r"\b(?:exceed(?:s|ed)?|overflows?|is\s+larger\s+than)\b.{0,40}"
    r"\b(?:the\s+)?(?:model(?:'s)?\s+)?context(?:\s+(?:length|window))?\b",
    re.I,
)


def is_context_window_exceeded_error(detail: str) -> bool:
    """Whether provider detail explicitly identifies model context overflow."""
    return bool(
        _STRUCTURED_CONTEXT.search(detail)
        or re.search(
            r"\b(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+"
            r"(?:length|window)\b",
            detail,
            re.I,
        )
        or _TOO_LARGE.search(detail)
        or re.search(
            r"\b(?:input|prompt|request)\s+(?:is\s+)?too\s+(?:long|large)\s+for\s+"
            r"(?:this|the)\s+model\b",
            detail,
            re.I,
        )
        or _EXCEEDS_CONTEXT.search(detail)
    )


def is_quota_exceeded_error(detail: str) -> bool:
    """Whether provider detail identifies terminal quota rather than rate limiting."""
    patterns = (
        r"\binsufficient[\s_-]+(?:quota|balance|credits?)\b",
        r"\b(?:quota|usage[\s_-]+limit)[\s_-]+(?:exceeded|exhausted|reached)\b",
        r"\bexceed(?:ed|s)?[\s_-]+(?:(?:your|the)[\s_-]+)?"
        r"(?:current[\s_-]+)?quota\b",
        r"\b(?:balance|credits?)[\s_-]+(?:exhausted|depleted)\b",
        r"\bout[\s_-]+of[\s_-]+(?:credits?|budget)\b",
    )
    return any(re.search(pattern, detail, re.I) is not None for pattern in patterns)


def is_harness_error(value: object) -> bool:
    return isinstance(value, HarnessError)


def error_chain(value: object) -> str:
    """Render an exception's message, aggregate members, and cause chain."""
    path: set[int] = set()

    def render(current: object) -> str:
        marker = id(current)
        if marker in path:
            return "<circular cause>"
        path.add(marker)
        try:
            if not isinstance(current, BaseException):
                if isinstance(current, Mapping) and isinstance(current.get("message"), str):
                    return str(current["message"])
                try:
                    return str(current)
                except Exception:
                    return "<unrenderable value>"
            message = str(current) or type(current).__name__
            members = ""
            if isinstance(current, BaseExceptionGroup) and current.exceptions:
                members = " [" + "; ".join(render(item) for item in current.exceptions) + "]"
            cause = getattr(current, "cause", None)
            if cause is None:
                cause = current.__cause__
            cause_text = "" if cause is None else render(cause)
            suffix = "" if not cause_text or cause_text == message else f": {cause_text}"
            return f"{message}{members}{suffix}"
        except Exception:
            return "<unrenderable value>"
        finally:
            path.remove(marker)

    return render(value)


def _valid_failure(value: object) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    message, code = value.get("message"), value.get("code")
    status = value.get("status")
    delay = value.get("providerRetryAfterMs")
    request_id = value.get("requestId")
    offload = value.get("offloadImages")
    if not isinstance(message, str) or not message or not isinstance(code, str) or not code:
        return None
    if status is not None and (
        isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599
    ):
        return None
    if delay is not None and (
        isinstance(delay, bool)
        or not isinstance(delay, int | float)
        or not math.isfinite(delay)
        or delay <= 0
    ):
        return None
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        return None
    if offload is not None and (not is_safe_integer(offload) or offload <= 0):
        return None
    keys = ("message", "code", "status", "providerRetryAfterMs", "requestId", "offloadImages")
    return cast(Mapping[str, Any], deep_freeze({key: value[key] for key in keys if key in value}))


def normalize_llm_failure(value: object) -> Mapping[str, Any]:
    """Detach trusted structured facts, or normalize an arbitrary adapter throw."""
    if isinstance(value, BaseException):
        carried = _valid_failure(getattr(value, "failure", None))
        own_code = getattr(value, "code", None)
        if carried is not None and carried["code"] == own_code:
            return carried
        message = str(value) or "LLM adapter failed"
        code = value.code if isinstance(value, HarnessError) else "UNKNOWN"
        return deep_freeze({"message": message, "code": code})
    try:
        message = str(value) or "LLM adapter failed"
    except Exception:
        message = "LLM adapter failed"
    return deep_freeze({"message": message, "code": "UNKNOWN"})
