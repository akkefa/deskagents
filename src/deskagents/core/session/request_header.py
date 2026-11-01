"""Request-header reconstruction over full ``request/header`` events.

Upstream: packages/core/session/src/request-header.ts
"""

from __future__ import annotations

from collections.abc import Iterable

from deskagents.llm import ToolSchema, call_config_equals
from deskagents.util.values import stringify_json

from .types import EpochHeader, SessionEvent


def canonical_header(header: EpochHeader) -> EpochHeader:
    """Normalize a header: an empty tool list or empty adapter defaults become absent."""
    canonical: dict[str, object] = {"config": header["config"]}
    defaults = header.get("adapterDefaults")
    if defaults is not None and (
        defaults.get("reasoningEffort") is True or defaults.get("maxTokens") is True
    ):
        canonical["adapterDefaults"] = defaults
    tools = header.get("tools")
    if tools:
        canonical["tools"] = tools
    return canonical


def _same_schema(a: ToolSchema, b: ToolSchema) -> bool:
    # Upstream compares JSON.stringify output, so key order is significant.
    return stringify_json(a) == stringify_json(b)


def header_equals(a: EpochHeader, b: EpochHeader) -> bool:
    """Field-wise equality over canonical headers; tool schemas compare in order."""
    defaults_a = a.get("adapterDefaults") or {}
    defaults_b = b.get("adapterDefaults") or {}
    if (
        not call_config_equals(a["config"], b["config"])
        or defaults_a.get("reasoningEffort") != defaults_b.get("reasoningEffort")
        or defaults_a.get("maxTokens") != defaults_b.get("maxTokens")
    ):
        return False
    tools_a, tools_b = a.get("tools") or [], b.get("tools") or []
    return len(tools_a) == len(tools_b) and all(
        _same_schema(x, y) for x, y in zip(tools_a, tools_b, strict=True)
    )


def fold_request_header(
    events: Iterable[SessionEvent], start: EpochHeader | None = None
) -> EpochHeader | None:
    """Fold a log (or prefix) into the header in force after its last snapshot."""
    state = start
    for event in events:
        if event["type"] == "request/header":
            state = canonical_header(event["data"]["header"])
    return state
