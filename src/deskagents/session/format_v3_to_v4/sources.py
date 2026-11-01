"""Upstream: packages/session/session-format-v3-to-v4/src/sources.ts

Only ``mapEventMessages`` is ported; the released-v3 plugin-source rewrites belong to
the v3->v4 migration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from deskagents.session.format import (
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatJsonObject,
    is_session_format_json_object,
)
from deskagents.util.values import JsonValue

from ._js import is_array, prop

_MESSAGE_SLOT_TYPES = frozenset(
    {"developer/message", "system/message", "assistant/message", "tool/result"}
)


def map_event_messages(
    event: SessionFormatEvent,
    transform: Callable[[SessionFormatJsonObject], SessionFormatJsonObject],
) -> SessionFormatEvent:
    """Visit only messages carried by first-party event payloads.

    Returns the event sharing all untouched payloads (the same object when nothing
    changed).
    """
    data = prop(event, "data")
    if not is_session_format_json_object(data):
        return event
    assert isinstance(data, Mapping)
    event_type = prop(event, "type")
    if event_type == "user/message":
        message = transform(data)
        return event if message is data else {**event, "data": cast(JsonValue, message)}
    if isinstance(event_type, str) and event_type in _MESSAGE_SLOT_TYPES:
        slot = prop(data, "message")
        if not isinstance(slot, Mapping):
            raise SessionFormatError(event_type + " requires a message")
        message = transform(slot)
        return (
            event
            if message is slot
            else {**event, "data": {**data, "message": cast(JsonValue, message)}}
        )
    key = (
        "inserted"
        if event_type == "agent/inbox/spliced"
        else "messages"
        if event_type == "session/title-llm-request"
        else None
    )
    if key is None:
        return event
    assert isinstance(event_type, str)
    messages = prop(data, key)
    if not is_array(messages):
        raise SessionFormatError(event_type + " requires message array")
    assert isinstance(messages, list | tuple)
    mapped: list[SessionFormatJsonObject] = []
    for member in messages:
        if not isinstance(member, Mapping):
            raise SessionFormatError(event_type + " requires message objects")
        mapped.append(transform(member))
    if all(message is messages[index] for index, message in enumerate(mapped)):
        return event
    return {**event, "data": {**data, key: cast(JsonValue, mapped)}}
