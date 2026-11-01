"""Upstream: packages/session/session-format-v3-to-v4/src/fork-result.ts

Native V4 fork-result identity and not-started error validation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from deskagents.session.format import SessionFormatError
from deskagents.util.values import is_safe_integer

from ._js import (
    UNDEFINED,
    greater_or_equal,
    is_array,
    is_number,
    js_string,
    prop,
    strict_equal,
)

_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)")


def assert_v4_fork_result(row: object) -> None:
    """Validate fork-generated not-started results without changing their identity or text.

    Append results carry their own sequence; replacement results retain an earlier
    identity. ``row`` is a logical event or physical event row.
    """
    if not isinstance(row, Mapping) or prop(row, "type") != "tool/result":
        return
    data = prop(row, "data")
    if not isinstance(data, Mapping):
        return
    error = prop(data, "error")
    message = prop(data, "message")
    if (
        not isinstance(error, Mapping)
        or prop(error, "code") != "TOOL_NOT_STARTED"
        or not isinstance(message, Mapping)
        or not isinstance(prop(message, "id"), str)
        or not message["id"].startswith("forked-tool-result-")
    ):
        return
    message_id: str = message["id"]
    source = prop(message, "source")
    call_id = prop(source, "callId") if isinstance(source, Mapping) else UNDEFINED
    prefix = f"forked-tool-result-{js_string(call_id)}-"
    suffix = message_id[len(prefix) :]
    operation = prop(row, "surfaceOp")
    replacement = isinstance(operation, Mapping) and prop(operation, "op") == "replace"
    decimal = _DECIMAL.fullmatch(suffix) is not None
    sequence = int(suffix) if decimal else None
    content = prop(message, "content")
    text: object = (
        content[0] if isinstance(content, list | tuple) and len(content) == 1 else UNDEFINED
    )
    source_event_seqs = prop(row, "sourceEventSeqs")
    seq = prop(row, "seq")
    if (
        not isinstance(call_id, str)
        or not message_id.startswith(prefix)
        or not decimal
        or not is_safe_integer(sequence)
        or (
            (not is_number(seq) or greater_or_equal(sequence, seq))
            if replacement
            else not strict_equal(sequence, seq)
        )
        or prop(error, "name") != "ToolNotStartedError"
        or (not replacement and (source_event_seqs is not UNDEFINED or operation != "append"))
        or (
            replacement
            and (
                not is_array(source_event_seqs)
                or len(source_event_seqs) != 1  # type: ignore[arg-type]
                or not strict_equal(source_event_seqs[0], sequence)  # type: ignore[index]
            )
        )
        or prop(message, "role") != "tool"
        or prop(message, "isError") is not True
        or not strict_equal(prop(message, "toolCallId"), call_id)
        or not isinstance(source, Mapping)
        or prop(source, "kind") != "tool"
        or not isinstance(text, Mapping)
        or prop(text, "type") != "text"
        or not isinstance(prop(text, "text"), str)
    ):
        raise SessionFormatError("invalid V4 not-started fork result")
