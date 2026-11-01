"""Provider failures from packages/llm/llm-deepseek/src/transport.ts at ce7073d."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime

from deskagents.llm.errors import (
    LlmError,
    is_context_window_exceeded_error,
    is_quota_exceeded_error,
)


def provider_error(
    raw: object, status: int | None = None, headers: Mapping[str, str] | None = None
) -> LlmError:
    envelope = raw if isinstance(raw, Mapping) else {}
    error = envelope.get("error", {})
    if not isinstance(error, Mapping):
        error = {}
    message = error.get("message")
    if not isinstance(message, str) or not message:
        message = (
            f"DeepSeek Messages request failed ({status if status is not None else 'stream error'})"
        )
    kind = error.get("type", "")
    detail = " ".join(str(x) for x in (kind, error.get("code", ""), message))
    if status in (401, 403) or kind in ("authentication_error", "permission_error"):
        code = "AUTH"
    elif is_quota_exceeded_error(detail) or status == 402:
        code = "QUOTA"
    elif status == 429 or kind == "rate_limit_error":
        code = "RATE_LIMIT"
    elif is_context_window_exceeded_error(detail):
        code = "CONTEXT_WINDOW_EXCEEDED"
    elif status in (400, 413) or kind == "invalid_request_error":
        code = "INVALID_REQUEST"
    elif (status is not None and status >= 500) or kind in ("api_error", "overloaded_error"):
        code = "SERVER"
    else:
        code = "SERVER" if status is None else f"HTTP_{status}"
    fields = {key.lower(): value for key, value in (headers or {}).items()}
    retry = fields.get("retry-after", "")
    delay = 0.0
    try:
        delay = (
            float(retry) * 1000
            if re.fullmatch(r"\d+(?:\.\d+)?", retry)
            else (parsedate_to_datetime(retry).timestamp() - time.time()) * 1000
        )
    except (ValueError, TypeError, OverflowError):
        pass
    request_id = (
        fields.get("request-id")
        or fields.get("x-request-id")
        or fields.get("x-deepseek-request-id")
    )
    return LlmError(
        message,
        code,
        status=status,
        request_id=request_id,
        provider_retry_after_ms=delay if math.isfinite(delay) and delay > 0 else None,
    )
