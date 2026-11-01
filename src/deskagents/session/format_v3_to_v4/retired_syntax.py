"""Upstream: packages/session/session-format-v3-to-v4/src/retired-syntax.ts

Retired native syntax remains a hard refusal even after a recoverable physical-row
failure.
"""

from __future__ import annotations

from collections.abc import Mapping

from deskagents.session.format import SessionFormatError, SessionFormatUnsupportedMigrationError

from ._js import is_array, js_string, prop


def _assert_block(block: object, subject: str) -> None:
    if isinstance(block, Mapping) and prop(block, "type") == "tool-result":
        raise SessionFormatError(f"{subject} must not contain a released tool-result wrapper")


def _assert_content(content: object, subject: str) -> None:
    if is_array(content):
        assert isinstance(content, list | tuple)
        for block in content:
            _assert_block(block, subject)


def _assert_message_content(message: object, subject: str) -> None:
    if isinstance(message, Mapping):
        _assert_content(prop(message, "content"), subject)


def assert_v4_retired_syntax(row: object) -> None:
    """Refuse retired headers, PTC tags, and tool-result blocks in interpreted content slots.

    Tool arguments, replay state, schema parameters, and nested extension values remain
    opaque. ``row`` is a parsed physical row or a complete native logical event.
    """
    if not isinstance(row, Mapping):
        return
    row_type = prop(row, "type")
    if row_type in ("tool/code-dispatch-start", "tool/code-dispatch") and (
        prop(row, "ignorable") is not True
    ):
        raise SessionFormatUnsupportedMigrationError(
            f"format v4 rejects retired event type {js_string(row_type)}"
        )
    data = prop(row, "data")
    if row_type == "request/header":
        header = prop(data, "header") if isinstance(data, Mapping) else None
        if not isinstance(data, Mapping) or not isinstance(header, Mapping):
            raise SessionFormatError("format v4 request/header requires data and header objects")
        if "system" in header:
            raise SessionFormatError("format v4 request/header rejects retired header.system")
        return
    if not isinstance(data, Mapping):
        return
    subject = f"format v4 {js_string(row_type)} at seq {js_string(prop(row, 'seq'))} content"
    if row_type == "user/message":
        _assert_content(prop(data, "content"), subject)
    elif row_type in ("developer/message", "assistant/message", "team/message/queued"):
        _assert_message_content(prop(data, "message"), subject)
    elif row_type in ("agent/inbox/spliced", "session/title-llm-request"):
        messages = prop(data, "inserted" if row_type == "agent/inbox/spliced" else "messages")
        if is_array(messages):
            assert isinstance(messages, list | tuple)
            for message in messages:
                _assert_message_content(message, subject)
    elif row_type == "compaction/summary":
        _assert_content(prop(data, "summary"), subject)
        _assert_content(prop(data, "rawOutput"), subject)
    elif row_type == "tool/ptc-dispatch":
        _assert_content(prop(data, "content"), subject)
    stream = prop(data, "stream")
    if row_type in ("assistant/message", "assistant/attempt") and is_array(stream):
        assert isinstance(stream, list | tuple)
        for entry in stream:
            if not isinstance(entry, Mapping) or prop(entry, "type") != "chunk":
                continue
            chunk = prop(entry, "chunk")
            if not isinstance(chunk, Mapping):
                continue
            if prop(chunk, "type") == "block-end":
                _assert_block(prop(chunk, "block"), subject)
            if prop(chunk, "type") == "block-start" and prop(chunk, "blockType") == "tool-result":
                raise SessionFormatError(
                    f"{subject} must not contain a released tool-result wrapper"
                )
