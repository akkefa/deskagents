"""Upstream: packages/session/session-format-v3-to-v4/src/tool-role.ts

Only ``assertV4ToolResultMessage`` is ported; ``liftToolResult`` belongs to the v3->v4
migration.
"""

from __future__ import annotations

from collections.abc import Mapping

from deskagents.session.format import SessionFormatError, SessionFormatEvent

from ._js import UNDEFINED, is_array, js_string, prop, strict_equal


def assert_v4_tool_result_message(event: SessionFormatEvent) -> None:
    """Validate the native V4 first-class tool-role message of one ``tool/result`` row.

    Raises :class:`SessionFormatError` when the row is not an exact first-class tool
    result.
    """
    event_type = prop(event, "type")
    if event_type != "tool/result":
        return
    subject = f"format v4 {event_type} at seq {js_string(prop(event, 'seq'))}"
    data = prop(event, "data")
    if not isinstance(data, Mapping):
        raise SessionFormatError(f"{subject} data must be an object")
    message = prop(data, "message")
    if not isinstance(message, Mapping):
        raise SessionFormatError(f"{subject} message must be an object")
    message_id = prop(message, "id")
    role = prop(message, "role")
    tool_call_id = prop(message, "toolCallId")
    source = prop(message, "source")
    is_error = prop(message, "isError")
    content = prop(message, "content")
    source_call_id = prop(source, "callId") if isinstance(source, Mapping) else UNDEFINED
    if not isinstance(message_id, str) or message_id == "":
        raise SessionFormatError(f"{subject} requires a first-class message with a string id")
    if role != "tool":
        raise SessionFormatError(f"{subject} requires a tool-role message")
    if (
        not isinstance(tool_call_id, str)
        or tool_call_id == ""
        or not strict_equal(source_call_id, tool_call_id)
    ):
        raise SessionFormatError(f"{subject} requires toolCallId matching its tool source")
    if not isinstance(source, Mapping) or prop(source, "kind") != "tool":
        raise SessionFormatError(f"{subject} requires a tool source")
    if not is_array(content):
        raise SessionFormatError(f"{subject} requires array content")
    assert isinstance(content, list | tuple)
    if any(
        isinstance(block, Mapping) and prop(block, "type") == "tool-result" for block in content
    ):
        raise SessionFormatError(
            f"{subject} content must not contain a released tool-result wrapper"
        )
    if is_error is not UNDEFINED and not isinstance(is_error, bool):
        raise SessionFormatError(f"{subject} isError must be boolean when present")
    if prop(data, "error") is not UNDEFINED and is_error is not True:
        raise SessionFormatError(f"{subject} carries error metadata for a non-error tool result")
