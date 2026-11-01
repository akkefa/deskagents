"""Upstream: packages/session/session-format-v3-to-v4/src/message-sources.ts

Native source admission preserves unknown attribution and refuses retired plugin
wrappers.
"""

from __future__ import annotations

from collections.abc import Mapping

from deskagents.session.format import (
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatJsonObject,
    is_session_format_json_object,
)

from ._js import UNDEFINED, is_array, prop
from .sources import map_event_messages


def _source(message: Mapping[str, object]) -> None:
    value = prop(message, "source")
    kind = prop(value, "kind") if isinstance(value, Mapping) else UNDEFINED
    if (
        not isinstance(value, Mapping)
        or not isinstance(kind, str)
        or kind == ""
        or kind == "plugin"
    ):
        raise SessionFormatError("format v4 message requires a producer-owned source kind")


def _checked(message: SessionFormatJsonObject) -> SessionFormatJsonObject:
    _source(message)
    return message


def assert_v4_message_sources(event: SessionFormatEvent) -> None:
    """Validate producer attribution in every declared durable message slot."""
    map_event_messages(event, _checked)


def assert_v4_source_row_admission(row: object) -> None:
    """Refuse retired source syntax even after a malformed recoverable row.

    Incomplete messages are left to decoding.
    """
    if not is_session_format_json_object(row):
        return
    assert isinstance(row, Mapping)
    data = prop(row, "data")
    if not isinstance(data, Mapping):
        return
    row_type = prop(row, "type")
    messages: object
    if row_type == "user/message":
        messages = [data]
    elif row_type in ("system/message", "assistant/message", "tool/result"):
        messages = [prop(data, "message")]
    elif row_type == "agent/inbox/spliced":
        messages = prop(data, "inserted")
    elif row_type == "session/title-llm-request":
        messages = prop(data, "messages")
    else:
        messages = []
    if not is_array(messages):
        return
    assert isinstance(messages, list | tuple)
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        value = prop(message, "source")
        if isinstance(value, Mapping) and prop(value, "kind") == "plugin":
            _source(message)
