"""Ported from packages/session/session-format-v3-to-v4/tests/tool-role.spec.ts.

Not ported (v3->v4 migration only: ``liftToolResult`` and ``createSessionFormatV3ToV4``
are not in Python): every case under "V4 tool-role view and lift" and "V3 to V4 edge
lifts released tool results", and the three "native tool-result validation" cases that
call ``liftToolResult``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatEventCollector,
)
from deskagents.session.format_v3_to_v4 import (
    released_v4_session_format_codec,
    restore_released_v4_artifact,
)
from deskagents.session.format_v3_to_v4.tool_role import assert_v4_tool_result_message

SPEC = "packages/session/session-format-v3-to-v4/tests/tool-role.spec.ts"
CODEC = "released V4 codec with native tool-role rows"
NATIVE = "native tool-result validation"

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "tool-role",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}


def event(seq: int, type_: str, data: Any) -> SessionFormatEvent:
    return {"type": type_, "seq": seq, "time": seq + 1, "data": data}


def wrapper_message(call_id: str = "call-1", text: str = "result") -> dict[str, Any]:
    """Released V3 canonical user-role wrapper tool/result message."""
    return {
        "id": f"tool-result-{call_id}",
        "role": "user",
        "source": {"kind": "tool", "callId": call_id},
        "content": [
            {
                "type": "tool-result",
                "toolCallId": call_id,
                "isError": False,
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


def first_class_message(
    call_id: str = "call-1", text: str = "result", is_error: Any = False
) -> dict[str, Any]:
    """Native V4 first-class tool-role message."""
    return {
        "id": f"tool-result-{call_id}",
        "role": "tool",
        "source": {"kind": "tool", "callId": call_id},
        "toolCallId": call_id,
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


@pytest.mark.upstream(f"{SPEC}::{CODEC} > encodes and decodes a first-class tool result unchanged")
def test_encodes_and_decodes_first_class_result() -> None:
    row = {
        **event(0, "tool/result", {"turn": 1, "step": 1, "message": first_class_message()}),
        "surfaceOp": "append",
    }
    encoded = released_v4_session_format_codec.encode_event(row)
    assert encoded == row
    collector = SessionFormatEventCollector()
    decoder = released_v4_session_format_codec.create_decoder(
        {"type": "session", **HEADER}, "strict"
    )
    decoder.decode_row(row, collector)
    assert collector.values == [row]


@pytest.mark.upstream(f"{SPEC}::{CODEC} > rejects a released-wrapper row in the current V4 decoder")
def test_rejects_released_wrapper_row() -> None:
    wrapper_row = {
        **event(0, "tool/result", {"turn": 1, "step": 1, "message": wrapper_message()}),
        "surfaceOp": "append",
    }
    collector = SessionFormatEventCollector()
    decoder = released_v4_session_format_codec.create_decoder(
        {"type": "session", **HEADER}, "strict"
    )
    with pytest.raises(SessionFormatError, match=r"requires a tool-role message"):
        decoder.decode_row(wrapper_row, collector)


@pytest.mark.upstream(
    f"{SPEC}::{CODEC} > keeps native semantic failures hard even in recoverable mode"
)
def test_native_semantic_failures_stay_hard() -> None:
    row = {
        **event(
            0,
            "tool/result",
            {"turn": 1, "step": 1, "message": first_class_message("call-1", "result", "yes")},
        ),
        "surfaceOp": "append",
    }
    decoder = released_v4_session_format_codec.create_decoder(
        {"type": "session", **HEADER}, "recoverable"
    )
    with pytest.raises(SessionFormatError, match=r"isError must be boolean"):
        decoder.decode_row(row, SessionFormatEventCollector())


@pytest.mark.upstream(
    f"{SPEC}::{CODEC} > refuses a native row whose error metadata lacks the message isError flag"
)
def test_refuses_error_metadata_without_is_error() -> None:
    bad = event(
        0,
        "tool/result",
        {
            "turn": 1,
            "step": 1,
            "message": first_class_message("call-1", "result", False),
            "error": {"name": "ToolNotStartedError", "code": "TOOL_NOT_STARTED"},
        },
    )
    with pytest.raises(SessionFormatError, match=r"error metadata for a non-error tool result"):
        released_v4_session_format_codec.encode_event(bad)


@pytest.mark.upstream(
    f"{SPEC}::{CODEC} > restores a first-class artifact through the released V3 relationship rules"
)
def test_restores_first_class_artifact() -> None:
    events: list[SessionFormatEvent] = [
        event(0, "turn/start", {"turn": 1}),
        event(1, "step/start", {"turn": 1, "step": 1}),
        {
            "type": "assistant/message",
            "seq": 2,
            "time": 3,
            "surfaceOp": "append",
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "a-1",
                    "role": "assistant",
                    "content": [
                        {"type": "tool-call", "id": "call-1", "name": "bash", "arguments": "{}"}
                    ],
                    "source": {"kind": "model", "provider": "mock", "model": "mock"},
                },
                "stream": [],
            },
        },
        {
            "type": "tool/call",
            "seq": 3,
            "time": 4,
            "data": {"turn": 1, "step": 1, "callId": "call-1", "name": "bash", "arguments": "{}"},
        },
        {
            "type": "tool/result",
            "seq": 4,
            "time": 5,
            "surfaceOp": "append",
            "data": {"turn": 1, "step": 1, "message": first_class_message("call-1")},
        },
        event(5, "step/end", {"turn": 1, "step": 1}),
        event(6, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    ]
    artifact = SessionFormatArtifact(header=HEADER, inherited_event_count=0, events=events)
    known = {cast(str, item["type"]) for item in events}
    assert restore_released_v4_artifact(artifact, known) is artifact
    message = cast(dict[str, Any], artifact.events[4]["data"])["message"]
    assert message["role"] == "tool"
    assert message["toolCallId"] == "call-1"


def native(**overrides: Any) -> SessionFormatEvent:
    return event(
        0, "tool/result", {"turn": 1, "step": 1, "message": {**first_class_message(), **overrides}}
    )


def raw(data: Any) -> SessionFormatEvent:
    return {"type": "tool/result", "seq": 0, "time": 1, "data": data}


@pytest.mark.upstream(
    f"{SPEC}::{NATIVE} > accepts a native message and ignores rows of other event types"
)
def test_accepts_native_and_ignores_other_types() -> None:
    assert_v4_tool_result_message(native())
    assert_v4_tool_result_message(event(0, "user/message", {"role": "user", "content": []}))


@pytest.mark.upstream(f"{SPEC}::{NATIVE} > refuses data and messages that are not objects")
def test_refuses_non_object_data_and_messages() -> None:
    with pytest.raises(SessionFormatError, match=r"data must be an object"):
        assert_v4_tool_result_message(raw("nope"))
    with pytest.raises(SessionFormatError, match=r"message must be an object"):
        assert_v4_tool_result_message(raw({"message": "nope"}))


@pytest.mark.upstream(f"{SPEC}::{NATIVE} > refuses ids that are not non-empty strings")
def test_refuses_invalid_ids() -> None:
    with pytest.raises(SessionFormatError, match=r"string id"):
        assert_v4_tool_result_message(native(id=""))
    with pytest.raises(SessionFormatError, match=r"string id"):
        assert_v4_tool_result_message(native(id=7))


@pytest.mark.upstream(f"{SPEC}::{NATIVE} > refuses a message that is not the tool role")
def test_refuses_non_tool_role() -> None:
    with pytest.raises(SessionFormatError, match=r"tool-role message"):
        assert_v4_tool_result_message(native(role="user"))


@pytest.mark.upstream(
    f"{SPEC}::{NATIVE} > refuses a tool call id that does not match the tool source"
)
def test_refuses_mismatched_tool_call_id() -> None:
    for message in [
        native(toolCallId="other"),
        native(toolCallId=""),
        native(source={"kind": "tool", "callId": "other"}),
        native(source="nope"),
    ]:
        with pytest.raises(SessionFormatError, match=r"matching its tool source"):
            assert_v4_tool_result_message(message)


@pytest.mark.upstream(f"{SPEC}::{NATIVE} > refuses a source that is not the tool source")
def test_refuses_non_tool_source() -> None:
    with pytest.raises(SessionFormatError, match=r"requires a tool source"):
        assert_v4_tool_result_message(native(source={"kind": "model", "callId": "call-1"}))


@pytest.mark.upstream(f"{SPEC}::{NATIVE} > refuses content that is not an array")
def test_refuses_non_array_content() -> None:
    with pytest.raises(SessionFormatError, match=r"array content"):
        assert_v4_tool_result_message(native(content="nope"))


@pytest.mark.upstream(f"{SPEC}::{NATIVE} > refuses an isError flag that is not boolean")
def test_refuses_non_boolean_is_error() -> None:
    with pytest.raises(SessionFormatError, match=r"isError must be boolean"):
        assert_v4_tool_result_message(native(isError="yes"))


@pytest.mark.upstream(
    f"{SPEC}::{NATIVE} > refuses a native message that still contains a released wrapper"
)
def test_refuses_native_message_with_wrapper() -> None:
    with pytest.raises(
        SessionFormatError, match=r"must not contain a released tool-result wrapper"
    ):
        assert_v4_tool_result_message(
            native(content=[{"type": "tool-result", "toolCallId": "call-1", "content": []}])
        )
