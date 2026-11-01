"""Ported from packages/session/session-format-v3-to-v4/tests/developer.spec.ts."""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEventCollector,
    SessionFormatUnsupportedMigrationError,
)
from deskagents.session.format_v3_to_v4 import (
    assert_v4_row_admission,
    restore_released_v4_artifact,
)
from deskagents.session.format_v3_to_v4 import (
    released_v4_session_format_codec as codec,
)

SPEC = "packages/session/session-format-v3-to-v4/tests/developer.spec.ts"
GROUP = "V4 developer messages"

ABSENT: Any = object()
"""Upstream ``undefined``: the member is omitted."""

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "developer",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}
MESSAGE: dict[str, Any] = {
    "id": "developer-1",
    "role": "developer",
    "source": {"kind": "tool-registry"},
    "content": [
        {"type": "tool-addition", "toolName": "search"},
        {"type": "tool-removal", "toolName": "old"},
    ],
}
DEVELOPER: dict[str, Any] = {
    "type": "developer/message",
    "seq": 3,
    "time": 4,
    "surfaceOp": "append",
    "data": {"turn": 1, "step": 1, "headerSeq": 2, "message": MESSAGE},
}
REQUEST: dict[str, Any] = {
    "type": "request/header",
    "seq": 2,
    "time": 3,
    "data": {
        "reason": "initial",
        "header": {
            "config": {"provider": "test", "model": "test"},
            "tools": [
                {"name": "search", "description": "Search", "parameters": {}, "deferLoading": True}
            ],
        },
    },
}
CONFIG: dict[str, Any] = {"provider": "test", "model": "test"}


def events() -> list[dict[str, Any]]:
    return [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 2, "data": {"turn": 1, "step": 1}},
        REQUEST,
        DEVELOPER,
        {"type": "step/end", "seq": 4, "time": 5, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 5,
            "time": 6,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]


def round_trip(input_: list[dict[str, Any]]) -> list[Any]:
    decoder = codec.create_decoder(codec.encode_header(HEADER, 0), "strict")
    output = SessionFormatEventCollector()
    for event in input_:
        decoder.decode_row(codec.encode_event(event), output)
    decoder.finish(output)
    return list(output.values)


def artifact_of(
    input_: list[dict[str, Any]], header: dict[str, Any] = HEADER, inherited: int = 0
) -> SessionFormatArtifact:
    return SessionFormatArtifact(header=header, inherited_event_count=inherited, events=input_)


def types_of(input_: list[dict[str, Any]]) -> set[str]:
    return {event["type"] for event in input_}


def restore(input_: list[dict[str, Any]], known: set[str] | None = None) -> SessionFormatArtifact:
    return restore_released_v4_artifact(
        artifact_of(input_), types_of(input_) if known is None else known
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > round trips native messages and deferred schemas without rewriting native "
    "payloads"
)
def test_round_trips_native_messages_and_deferred_schemas() -> None:
    input_ = events()
    assert round_trip(input_) == input_
    artifact = artifact_of(input_)
    assert restore_released_v4_artifact(artifact, types_of(input_)) is artifact


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > preserves additional JSON fields in developer data, messages, sources, "
    "and blocks"
)
def test_preserves_additional_json_fields() -> None:
    input_ = events()
    input_[3] = {
        **DEVELOPER,
        "data": {
            "turn": 1,
            "step": 1,
            "headerSeq": 2,
            "metadata": {"future": True},
            "message": {
                **MESSAGE,
                "metadata": ["retained"],
                "source": {"kind": "tool-registry", "future": 1},
                "content": [
                    {"type": "tool-addition", "toolName": "search", "future": True},
                    {"type": "tool-removal", "toolName": "old", "future": True},
                ],
            },
        },
    }
    input_[2] = {
        **REQUEST,
        "data": {
            "reason": "initial",
            "header": {
                "config": CONFIG,
                "tools": [
                    {
                        "name": "search",
                        "description": "Search",
                        "parameters": {},
                        "deferLoading": True,
                        "future": {"retained": True},
                    }
                ],
            },
        },
    }
    assert round_trip(input_) == input_
    artifact = artifact_of(input_)
    assert restore_released_v4_artifact(artifact, types_of(input_)) is artifact


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects a malformed addition name %#")
@pytest.mark.parametrize("tool_name", [ABSENT, None, "", 1])
def test_rejects_a_malformed_addition_name(tool_name: Any) -> None:
    block: dict[str, Any] = {"type": "tool-addition"}
    if tool_name is not ABSENT:
        block["toolName"] = tool_name
    row = {
        **DEVELOPER,
        "data": {"turn": 1, "step": 1, "headerSeq": 2, "message": {**MESSAGE, "content": [block]}},
    }
    with pytest.raises(SessionFormatError, match="nonempty toolName"):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError, match="nonempty toolName"):
        assert_v4_row_admission(row)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses an inline definition even when a header and name are present"
)
def test_refuses_an_inline_definition() -> None:
    block = {
        "type": "tool-addition",
        "toolName": "search",
        "tool": {"name": "search", "description": "Duplicate", "parameters": {}},
    }
    row = {
        **DEVELOPER,
        "data": {"turn": 1, "step": 1, "headerSeq": 2, "message": {**MESSAGE, "content": [block]}},
    }
    with pytest.raises(SessionFormatError, match="omit inline tool definitions"):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError, match="omit inline tool definitions"):
        assert_v4_row_admission(row)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects an absent or invalid addition header sequence %#")
@pytest.mark.parametrize("header_seq", [ABSENT, None, -1, -0.0, 1.5, "2"])
def test_rejects_an_absent_or_invalid_addition_header_sequence(header_seq: Any) -> None:
    data: dict[str, Any] = {"turn": 1, "step": 1, "message": MESSAGE}
    if header_seq is not ABSENT:
        data["headerSeq"] = header_seq
    row = {**DEVELOPER, "data": data}
    with pytest.raises(SessionFormatError, match="headerSeq"):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError, match="headerSeq"):
        assert_v4_row_admission(row)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > requires removal-only messages to omit the header sequence"
)
def test_requires_removal_only_messages_to_omit_the_header_sequence() -> None:
    row = {
        **DEVELOPER,
        "data": {
            "turn": 1,
            "step": 1,
            "headerSeq": 2,
            "message": {**MESSAGE, "content": [{"type": "tool-removal", "toolName": "search"}]},
        },
    }
    with pytest.raises(SessionFormatError, match="omit headerSeq"):
        codec.encode_event(row)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses a non-header or non-earlier schema reference %s")
@pytest.mark.parametrize("header_seq", [0, 3, 50])
def test_refuses_a_non_header_or_non_earlier_schema_reference(header_seq: int) -> None:
    input_ = events()
    input_[3] = {
        **DEVELOPER,
        "data": {"turn": 1, "step": 1, "headerSeq": header_seq, "message": MESSAGE},
    }
    with pytest.raises(SessionFormatError, match=r"headerSeq.*earlier"):
        restore(input_)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses missing or ambiguous names in the referenced header %#"
)
@pytest.mark.parametrize(
    "tools",
    [
        ABSENT,
        [None],
        [{"name": "different", "description": "", "parameters": {}}],
        [
            {"name": "search", "description": "First", "parameters": {}},
            {"name": "search", "description": "Second", "parameters": {}},
        ],
    ],
)
def test_refuses_missing_or_ambiguous_names_in_the_referenced_header(tools: Any) -> None:
    input_ = events()
    header: dict[str, Any] = {"config": CONFIG}
    if tools is not ABSENT:
        header["tools"] = tools
    input_[2] = {**REQUEST, "data": {"reason": "initial", "header": header}}
    with pytest.raises(SessionFormatError, match="exactly one tool"):
        restore(input_)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses an incomplete referenced definition %#")
@pytest.mark.parametrize(
    "tool",
    [
        {"name": "search"},
        {"name": "search", "description": None, "parameters": {}},
        {"name": "search", "description": "", "parameters": []},
        {"name": "search", "description": "", "parameters": None},
    ],
)
def test_refuses_an_incomplete_referenced_definition(tool: dict[str, Any]) -> None:
    input_ = events()
    input_[2] = {
        **REQUEST,
        "data": {"reason": "initial", "header": {"config": CONFIG, "tools": [tool]}},
    }
    with pytest.raises(SessionFormatError, match="complete tool definition"):
        restore(input_)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > does not interpret an unknown ignorable header referenced by a known "
    "developer event %#"
)
@pytest.mark.parametrize("data", [None, REQUEST["data"]])
def test_does_not_interpret_an_unknown_ignorable_header(data: Any) -> None:
    input_ = events()
    input_[2] = {**REQUEST, "ignorable": True, "data": data}
    known = types_of(input_)
    known.discard("request/header")
    with pytest.raises(SessionFormatError, match="known request/header"):
        restore(input_, known)
    input_[3] = {
        **DEVELOPER,
        "ignorable": True,
        "data": {"opaque": True, "headerSeq": {"future": True}},
        "surfaceOp": {"future": True},
    }
    known.discard("developer/message")
    artifact = artifact_of(input_)
    assert restore_released_v4_artifact(artifact, known) is artifact


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps historical schema references when compaction shadows their "
    "developer nodes"
)
def test_keeps_historical_schema_references_when_compaction_shadows() -> None:
    input_ = [
        *events()[:4],
        {
            "type": "compaction/prune",
            "seq": 4,
            "time": 5,
            "data": {
                "shadowedRange": {"start": 3, "end": 3},
                "shadowedSeqs": [3],
                "shadowedTokenCount": 0,
            },
        },
        *({**event, "seq": event["seq"] + 1} for event in events()[4:]),
    ]
    artifact = artifact_of(input_)
    assert restore_released_v4_artifact(artifact, types_of(input_)) is artifact
    assert input_[2] == REQUEST


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses non-object developer data %j before recovery")
@pytest.mark.parametrize("data", [None, [], "invalid"])
def test_refuses_non_object_developer_data(data: Any) -> None:
    row = {**DEVELOPER, "data": data}
    with pytest.raises(SessionFormatError, match="developer/message data must be an object"):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError, match="developer/message data must be an object"):
        assert_v4_row_admission(row)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects malformed developer event fields %#")
@pytest.mark.parametrize("data", [{"turn": 1, "step": 1}, {"turn": 1, "step": 1, "message": None}])
def test_rejects_malformed_developer_event_fields(data: dict[str, Any]) -> None:
    with pytest.raises(SessionFormatError, match="requires turn, step, and a developer message"):
        codec.encode_event({**DEVELOPER, "data": data})


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > leaves malformed legacy stream chunks to canonical decoding"
)
def test_leaves_malformed_legacy_stream_chunks_to_canonical_decoding() -> None:
    assert_v4_row_admission(
        {
            "type": "assistant/attempt",
            "seq": 2,
            "time": 3,
            "data": {"stream": [{"type": "chunk", "chunk": None}]},
        }
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > leaves malformed ordinary arrays in %s to canonical decoding"
)
@pytest.mark.parametrize("type_", ["agent/inbox/spliced", "session/title-llm-request"])
def test_leaves_malformed_ordinary_arrays_to_canonical_decoding(type_: str) -> None:
    key = "inserted" if type_ == "agent/inbox/spliced" else "messages"
    assert_v4_row_admission({"type": type_, "seq": 2, "time": 3, "data": {key: None}})


@pytest.mark.upstream(f"{SPEC}::{GROUP} > preserves ordinary tools alongside deferred tools: %s")
@pytest.mark.parametrize("mixed", [False, True])
def test_preserves_ordinary_tools_alongside_deferred_tools(mixed: bool) -> None:
    ordinary = {"name": "ordinary", "description": "Ordinary", "parameters": {}}
    tools = (
        [ordinary, {**ordinary, "name": "deferred", "deferLoading": True}] if mixed else [ordinary]
    )
    input_ = events()
    input_[2] = {
        **REQUEST,
        "data": {"reason": "initial", "header": {"config": CONFIG, "tools": tools}},
    }
    assert round_trip(input_) == input_


@pytest.mark.upstream(f"{SPEC}::{GROUP} > does not interpret tool entries without deferred markers")
def test_does_not_interpret_tool_entries_without_deferred_markers() -> None:
    row = {
        **REQUEST,
        "data": {"reason": "initial", "header": {"config": CONFIG, "tools": [None]}},
    }
    assert_v4_row_admission(row)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > preserves the historical header while replacing a developer surface node"
)
def test_preserves_the_historical_header_while_replacing_a_developer_surface_node() -> None:
    replacement = {
        **DEVELOPER,
        "seq": 4,
        "time": 5,
        "surfaceOp": {"op": "replace", "startSeq": 3, "endSeq": 3},
        "sourceEventSeqs": [3],
        "data": {
            "turn": 1,
            "step": 1,
            "headerSeq": 2,
            "message": {**MESSAGE, "id": "developer-2"},
        },
    }
    input_ = [
        *events()[:4],
        replacement,
        *({**event, "seq": event["seq"] + 1} for event in events()[4:]),
    ]
    assert round_trip(input_) == input_
    restored = restore(input_)
    assert list(restored.events) == input_


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects invalid deferred marker %j")
@pytest.mark.parametrize("defer_loading", [False, None, "true", 1])
def test_rejects_invalid_deferred_marker(defer_loading: Any) -> None:
    tool = {"name": "search", "description": "", "parameters": {}, "deferLoading": defer_loading}
    row = {
        **REQUEST,
        "data": {"reason": "initial", "header": {"config": CONFIG, "tools": [tool]}},
    }
    with pytest.raises(SessionFormatError, match="deferLoading"):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError, match="deferLoading"):
        assert_v4_row_admission(row)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects malformed developer message %#")
@pytest.mark.parametrize(
    "invalid",
    [
        {**MESSAGE, "role": "user"},
        {**MESSAGE, "id": ""},
        {**MESSAGE, "source": {"kind": "plugin"}},
        {**MESSAGE, "content": [{"type": "tool-addition", "toolName": ""}]},
        {**MESSAGE, "content": [{"type": "tool-removal", "toolName": ""}]},
    ],
)
def test_rejects_malformed_developer_message(invalid: dict[str, Any]) -> None:
    row = {**DEVELOPER, "data": {"turn": 1, "step": 1, "message": invalid}}
    with pytest.raises(SessionFormatError):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError):
        assert_v4_row_admission(row)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > preserves developer messages inside an inherited prefix")
def test_preserves_developer_messages_inside_an_inherited_prefix() -> None:
    inherited_header = {**HEADER, "isSeeded": True, "parentSession": "parent"}
    input_ = events()
    inherited = [
        *input_[:4],
        {"type": "session/end-seed", "seq": 4, "time": 5, "data": {"inherited": True}},
        *({**event, "seq": event["seq"] + 1} for event in input_[4:]),
    ]
    decoder = codec.create_decoder(codec.encode_header(inherited_header, 4), "strict")
    output = SessionFormatEventCollector()
    for event in inherited:
        decoder.decode_row(codec.encode_event(event), output)
    assert decoder.finish(output) == 4
    assert output.values == inherited
    artifact = SessionFormatArtifact(
        header=inherited_header, inherited_event_count=4, events=output.values
    )
    assert restore_released_v4_artifact(artifact, types_of(inherited)) is artifact


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses developer history when the reader does not know the event type"
)
def test_refuses_developer_history_when_the_reader_does_not_know_the_event_type() -> None:
    input_ = events()
    known = types_of(input_)
    known.discard("developer/message")
    with pytest.raises(SessionFormatUnsupportedMigrationError, match="unknown event type"):
        restore(input_, known)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > checks request coordinates and the open step")
def test_checks_request_coordinates_and_the_open_step() -> None:
    input_ = events()
    input_[3] = {**DEVELOPER, "data": {**DEVELOPER["data"], "step": 2}}
    with pytest.raises(SessionFormatError, match=r"open.*step"):
        restore(input_)
    with pytest.raises(SessionFormatError, match="positive"):
        codec.encode_event({**DEVELOPER, "data": {"turn": 0, "step": 1, "message": MESSAGE}})


def _tool_change_row(role: str, block_type: str) -> dict[str, Any]:
    source: dict[str, Any]
    if role == "system":
        source = {"kind": "system-prompt"}
    elif role == "assistant":
        source = {"kind": "model", "provider": "test", "model": "test"}
    elif role == "tool":
        source = {"kind": "tool", "callId": "call"}
    else:
        source = {"kind": "user"}
    invalid: dict[str, Any] = {
        **MESSAGE,
        "role": role,
        "source": source,
        "content": [{"type": block_type, "toolName": "search"}],
        **({"toolCallId": "call", "isError": False} if role == "tool" else {}),
    }
    event_type = "tool/result" if role == "tool" else f"{role}/message"
    return {
        **DEVELOPER,
        "type": event_type,
        "data": invalid if role == "user" else {"turn": 1, "step": 1, "message": invalid},
    }


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects tool-change blocks in %s messages")
@pytest.mark.parametrize("role", ["user", "system", "assistant", "tool"])
def test_rejects_tool_change_blocks_in_messages(role: str) -> None:
    for block_type in ["tool-addition", "tool-removal"]:
        row = _tool_change_row(role, block_type)
        with pytest.raises(SessionFormatError, match="developer role"):
            codec.encode_event(row)
        with pytest.raises(SessionFormatError, match="developer role"):
            assert_v4_row_admission(row)
        decoder = codec.create_decoder(codec.encode_header(HEADER, 0), "strict")
        with pytest.raises(SessionFormatError, match="developer role"):
            decoder.decode_row(row, SessionFormatEventCollector())
        input_ = events()
        input_[3] = row
        with pytest.raises(SessionFormatError, match="developer role"):
            restore(input_)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects developer role inside %s before payload recovery")
@pytest.mark.parametrize(
    "type_", ["user/message", "system/message", "assistant/message", "tool/result"]
)
def test_rejects_developer_role_inside_before_payload_recovery(type_: str) -> None:
    row: dict[str, Any] = {
        **DEVELOPER,
        "type": type_,
        "data": MESSAGE if type_ == "user/message" else {"turn": 1, "step": 1, "message": MESSAGE},
    }
    with pytest.raises(
        SessionFormatError, match=r"must occur together|requires a tool-role message"
    ):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError, match="must occur together"):
        assert_v4_row_admission(row)
    decoder = codec.create_decoder(codec.encode_header(HEADER, 0), "strict")
    with pytest.raises(SessionFormatError, match="must occur together"):
        decoder.decode_row(row, SessionFormatEventCollector())


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects developer role in %s message arrays")
@pytest.mark.parametrize("type_", ["agent/inbox/spliced", "session/title-llm-request"])
def test_rejects_developer_role_in_message_arrays(type_: str) -> None:
    key = "inserted" if type_ == "agent/inbox/spliced" else "messages"
    row: dict[str, Any] = {"type": type_, "seq": 3, "time": 4, "data": {key: [None, MESSAGE]}}
    with pytest.raises(SessionFormatError, match="must occur together"):
        codec.encode_event(row)
    with pytest.raises(SessionFormatError, match="must occur together"):
        assert_v4_row_admission(row)
    decoder = codec.create_decoder(codec.encode_header(HEADER, 0), "strict")
    with pytest.raises(SessionFormatError, match="must occur together"):
        decoder.decode_row(row, SessionFormatEventCollector())
    input_ = events()
    input_[3] = row
    with pytest.raises(SessionFormatError, match="must occur together"):
        restore(input_)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > retains unknown ignorable developer records without interpreting their "
    "payload or surface"
)
def test_retains_unknown_ignorable_developer_records() -> None:
    input_ = events()
    input_[3] = {
        **DEVELOPER,
        "ignorable": True,
        "data": {"future": True},
        "surfaceOp": {"future": True},
    }
    known = types_of(input_)
    known.discard("developer/message")
    artifact = artifact_of(input_)
    assert restore_released_v4_artifact(artifact, known) is artifact


@pytest.mark.upstream(f"{SPEC}::{GROUP} > still validates known ignorable developer records")
def test_still_validates_known_ignorable_developer_records() -> None:
    input_ = events()
    input_[3] = {
        **DEVELOPER,
        "ignorable": True,
        "data": {"turn": 1, "step": 2, "headerSeq": 2, "message": MESSAGE},
    }
    with pytest.raises(SessionFormatError, match=r"open.*step"):
        restore(input_)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects %s outside developer content, including streamed blocks"
)
@pytest.mark.parametrize("type_", ["tool-addition", "tool-removal"])
def test_rejects_outside_developer_content_including_streamed_blocks(type_: str) -> None:
    block = {"type": type_, "toolName": "search"}
    payloads: list[dict[str, Any]] = [
        {"type": "compaction/summary", "data": {"summary": [block]}},
        {"type": "compaction/summary", "data": {"rawOutput": [block]}},
        {
            "type": "assistant/attempt",
            "data": {"stream": [{"type": "chunk", "chunk": {"type": "block-end", "block": block}}]},
        },
        {
            "type": "assistant/attempt",
            "data": {
                "stream": [{"type": "chunk", "chunk": {"type": "block-start", "blockType": type_}}]
            },
        },
        {
            "type": "agent/inbox/spliced",
            "data": {"inserted": [{**MESSAGE, "role": "user", "content": [block]}]},
        },
        {
            "type": "session/title-llm-request",
            "data": {"messages": [{**MESSAGE, "role": "user", "content": [block]}]},
        },
    ]
    for payload in payloads:
        with pytest.raises(SessionFormatError, match="developer role"):
            assert_v4_row_admission({"seq": 2, "time": 3, **payload})
