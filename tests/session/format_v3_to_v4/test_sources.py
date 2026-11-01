"""Ported from packages/session/session-format-v3-to-v4/tests/sources.spec.ts.

Not ported (v3->v4 migration only: ``rewritePluginSource`` and ``rewriteV3MessageSource``
are not in Python): every case under "rewritePluginSource" and "external V3 source
identities".
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from deskagents.session.format import (
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatJsonObject,
)
from deskagents.session.format_v3_to_v4.sources import map_event_messages

SPEC = "packages/session/session-format-v3-to-v4/tests/sources.spec.ts"
GROUP = "mapEventMessages"


def event(type_: str, data: dict[str, Any]) -> SessionFormatEvent:
    return {"type": type_, "seq": 0, "time": 1, "data": data}


def identity(message: SessionFormatJsonObject) -> SessionFormatJsonObject:
    return message


def rename(id_: str) -> Any:
    return lambda value: {**value, "id": id_}


@pytest.mark.upstream(f"{SPEC}::{GROUP} > returns events whose data is not an object")
def test_returns_events_with_non_object_data() -> None:
    row = cast(SessionFormatEvent, {"type": "turn/start", "seq": 0, "time": 1, "data": "nope"})
    assert map_event_messages(row, identity) is row


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > transforms the user/message payload and preserves identity when unchanged"
)
def test_transforms_user_message_payload() -> None:
    message: dict[str, Any] = {
        "id": "u",
        "role": "user",
        "source": {"kind": "user"},
        "content": [],
    }
    row = event("user/message", message)
    assert map_event_messages(row, identity) is row
    assert map_event_messages(row, rename("u2"))["data"] == {**message, "id": "u2"}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > transforms the message payload of system, assistant, and tool/result rows"
)
def test_transforms_message_slot_payload() -> None:
    message: dict[str, Any] = {
        "id": "s",
        "role": "system",
        "source": {"kind": "system-prompt"},
        "content": [],
    }
    row = event("system/message", {"turn": 1, "step": 1, "message": message})
    assert map_event_messages(row, identity) is row
    data = cast(dict[str, Any], map_event_messages(row, rename("s2"))["data"])
    assert data["message"] == {**message, "id": "s2"}


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses a message payload that is not an object")
def test_refuses_non_object_message_payload() -> None:
    with pytest.raises(SessionFormatError, match=r"requires a message"):
        map_event_messages(event("system/message", {"message": "nope"}), identity)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > transforms message arrays on spliced and title-request rows"
)
def test_transforms_message_arrays() -> None:
    message: dict[str, Any] = {
        "id": "u",
        "role": "user",
        "source": {"kind": "user"},
        "content": [],
    }
    spliced = event(
        "agent/inbox/spliced", {"target": "next-turn", "start": 0, "inserted": [message]}
    )
    assert map_event_messages(spliced, identity) is spliced
    spliced_data = cast(dict[str, Any], map_event_messages(spliced, rename("u2"))["data"])
    assert spliced_data["inserted"] == [{**message, "id": "u2"}]
    title = event("session/title-llm-request", {"messages": [message]})
    assert map_event_messages(title, identity) is title
    title_data = cast(dict[str, Any], map_event_messages(title, rename("u2"))["data"])
    assert title_data["messages"] == [{**message, "id": "u2"}]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses a message array that is not an array and members that are not "
    "objects"
)
def test_refuses_invalid_message_arrays() -> None:
    with pytest.raises(SessionFormatError, match=r"requires message array"):
        map_event_messages(event("agent/inbox/spliced", {"inserted": "nope"}), identity)
    with pytest.raises(SessionFormatError, match=r"requires message objects"):
        map_event_messages(event("agent/inbox/spliced", {"inserted": ["nope"]}), identity)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > returns unrelated events unchanged")
def test_returns_unrelated_events_unchanged() -> None:
    row = event("turn/start", {"turn": 1})
    assert map_event_messages(row, identity) is row
