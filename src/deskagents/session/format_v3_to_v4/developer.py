"""Upstream: packages/session/session-format-v3-to-v4/src/developer.ts

Native V4 developer-message and deferred-tool-schema validation.
"""

from __future__ import annotations

from collections.abc import Mapping

from deskagents.session.format import SessionFormatError, SessionFormatEvent, session_format_count

from ._js import UNDEFINED, is_array, js_string, prop

_TOOL_CHANGES = ("tool-addition", "tool-removal")


def _assert_tool_change(block: object, developer: bool) -> None:
    if not isinstance(block, Mapping) or prop(block, "type") not in _TOOL_CHANGES:
        return
    block_type = js_string(block["type"])
    if not developer:
        raise SessionFormatError(f"format v4 {block_type} requires developer role")
    tool_name = prop(block, "toolName")
    if not isinstance(tool_name, str) or tool_name == "":
        raise SessionFormatError(f"format v4 {block_type} requires a nonempty toolName")
    if block["type"] == "tool-addition" and "tool" in block:
        raise SessionFormatError("format v4 tool-addition must omit inline tool definitions")


def _assert_content(value: object, developer: bool = False) -> None:
    if is_array(value):
        assert isinstance(value, list | tuple)
        for block in value:
            _assert_tool_change(block, developer)


def _assert_developer_message(message: Mapping[str, object]) -> None:
    _assert_content(prop(message, "content"), True)
    source = prop(message, "source")
    message_id = prop(message, "id")
    kind = prop(source, "kind") if isinstance(source, Mapping) else UNDEFINED
    if (
        not isinstance(message_id, str)
        or message_id == ""
        or not is_array(prop(message, "content"))
        or not isinstance(source, Mapping)
        or not isinstance(kind, str)
        or kind == ""
        or kind == "plugin"
    ):
        raise SessionFormatError(
            "format v4 developer message requires id, role, content, and a producer-owned source"
        )


def _assert_ordinary_message(value: object) -> None:
    """Ordinary-message corruption belongs to the decoder; V4-only values are refused here."""
    if not isinstance(value, Mapping):
        return
    if prop(value, "role") == "developer":
        raise SessionFormatError(
            "format v4 developer/message and developer role must occur together"
        )
    _assert_content(prop(value, "content"))


def assert_v4_developer_data(event: SessionFormatEvent) -> None:
    """Validate native V4 developer messages, tool-change blocks, and deferred tool schemas.

    Additional developer JSON fields are preserved. Malformed legacy message slots remain
    available to recoverable decoding. ``event`` is a logical event or a physical row
    before source-event range decoding.
    """
    data = prop(event, "data")
    event_type = prop(event, "type")
    if not isinstance(data, Mapping):
        if event_type == "developer/message":
            raise SessionFormatError("format v4 developer/message data must be an object")
        return
    if event_type == "developer/message":
        message = prop(data, "message")
        if not isinstance(message, Mapping) or prop(message, "role") != "developer":
            raise SessionFormatError(
                "format v4 developer/message requires turn, step, and a developer message"
            )
        for field in ("turn", "step"):
            if session_format_count(prop(data, field), f"developer/message {field}") == 0:
                raise SessionFormatError(f"developer/message {field} must be positive")
        _assert_developer_message(message)
        content = message["content"]
        assert isinstance(content, list | tuple)
        has_additions = any(
            isinstance(block, Mapping) and prop(block, "type") == "tool-addition"
            for block in content
        )
        if has_additions:
            session_format_count(prop(data, "headerSeq"), "developer/message headerSeq")
        elif "headerSeq" in data:
            raise SessionFormatError(
                "format v4 developer/message must omit headerSeq without tool additions"
            )
    elif event_type == "user/message":
        _assert_ordinary_message(data)
    elif event_type in ("system/message", "assistant/message", "tool/result"):
        _assert_ordinary_message(prop(data, "message"))
    elif event_type in ("agent/inbox/spliced", "session/title-llm-request"):
        messages = prop(data, "inserted" if event_type == "agent/inbox/spliced" else "messages")
        if is_array(messages):
            assert isinstance(messages, list | tuple)
            for member in messages:
                _assert_ordinary_message(member)
    if event_type == "compaction/summary":
        _assert_content(prop(data, "summary"))
        _assert_content(prop(data, "rawOutput"))
    stream = prop(data, "stream")
    if event_type in ("assistant/message", "assistant/attempt") and is_array(stream):
        assert isinstance(stream, list | tuple)
        for entry in stream:
            if not isinstance(entry, Mapping) or prop(entry, "type") != "chunk":
                continue
            chunk = prop(entry, "chunk")
            if not isinstance(chunk, Mapping):
                continue
            if prop(chunk, "type") == "block-end":
                _assert_tool_change(prop(chunk, "block"), False)
            if prop(chunk, "type") == "block-start" and prop(chunk, "blockType") in _TOOL_CHANGES:
                raise SessionFormatError("format v4 tool-change blocks require developer role")
    header = prop(data, "header")
    if event_type == "request/header" and isinstance(header, Mapping):
        tools = prop(header, "tools")
        if is_array(tools):
            assert isinstance(tools, list | tuple)
            for tool in tools:
                if not isinstance(tool, Mapping) or "deferLoading" not in tool:
                    continue
                if tool["deferLoading"] is not True:
                    raise SessionFormatError(
                        "format v4 tool deferLoading must be true when present"
                    )
