"""Ported from packages/session/session-format-v3-to-v4/tests/codec.spec.ts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEventCollector,
    SessionFormatHeader,
    SessionFormatRestoreOptions,
)
from deskagents.session.format_catalog import session_format_catalog
from deskagents.session.format_v1_to_v2 import released_v2_session_format_codec
from deskagents.session.format_v3_to_v4 import (
    assert_released_v4_header,
    assert_v4_row_admission,
    released_v4_session_format_codec,
    restore_released_v4_artifact,
)

SPEC = "packages/session/session-format-v3-to-v4/tests/codec.spec.ts"
GROUP = "V4 framing and restoration"

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "parent",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}
FACT: dict[str, Any] = {
    "type": "feedback/record",
    "seq": 0,
    "time": 1,
    "data": {"text": "retained"},
}
CATALOG: dict[str, Any] = {
    "type": "subagent/catalog",
    "seq": 0,
    "time": 1,
    "data": {"version": 0, "childId": "child", "childCreatedAt": 2, "mode": "one-shot"},
}
TYPES = frozenset(
    [
        "subagent/catalog",
        "feedback/record",
        "session-log-deepseek/delivery-accepted",
        "session/end-seed",
    ]
)


def delivery(version: int | None, session_id: str = "parent") -> dict[str, Any]:
    data: dict[str, Any] = {"sessionId": session_id, "throughSeq": 0}
    if version is not None:
        data["sessionFormatVersion"] = version
    return {"type": "session-log-deepseek/delivery-accepted", "seq": 1, "time": 2, "data": data}


def restore(
    events: Sequence[Any], source_header: SessionFormatHeader = HEADER
) -> SessionFormatArtifact:
    return restore_released_v4_artifact(
        SessionFormatArtifact(header=source_header, events=events, inherited_event_count=0), TYPES
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > reuses released V2 physical framing and round trips native V4 rows"
)
def test_reuses_released_v2_framing_and_round_trips() -> None:
    codec = released_v4_session_format_codec
    physical = codec.encode_header(HEADER, 0)
    assert physical == {"type": "session", **HEADER}
    assert codec.decode_header(physical) == HEADER
    decoder = codec.create_decoder(physical, "strict")
    output = SessionFormatEventCollector()
    assert decoder.header == HEADER
    row = codec.encode_event(FACT)
    assert row == released_v2_session_format_codec.encode_event(FACT)
    decoder.decode_row(row, output)
    assert decoder.finish(output) == 0
    assert output.values == [FACT]
    artifact = SessionFormatArtifact(header=HEADER, events=output.values, inherited_event_count=0)
    assert restore_released_v4_artifact(artifact, TYPES) is artifact


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects mismatched versions and malformed logical and physical metadata"
)
def test_rejects_mismatched_versions_and_malformed_metadata() -> None:
    codec = released_v4_session_format_codec
    with pytest.raises(SessionFormatError, match="v4 header"):
        assert_released_v4_header({**HEADER, "version": 3})
    with pytest.raises(SessionFormatError, match="createdAt"):
        assert_released_v4_header({**HEADER, "createdAt": -1})
    with pytest.raises(SessionFormatError, match="v4 physical"):
        codec.decode_header(None)
    with pytest.raises(SessionFormatError, match="v4 physical"):
        codec.create_decoder({**HEADER, "version": 3}, "strict")
    with pytest.raises(SessionFormatError, match=r"field|member"):
        codec.decode_header({"type": "session", **HEADER, "extra": 1})


MISSING_ID = {key: value for key, value in HEADER.items() if key != "id"}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects malformed optional and required native header fields"
)
def test_rejects_malformed_native_header_fields() -> None:
    malformed: list[tuple[dict[str, Any], str]] = [
        (MISSING_ID, "lacks required field id"),
        ({**HEADER, "extra": True}, "unexpected field extra"),
        ({**HEADER, "id": 7}, "id must be a string"),
        ({**HEADER, "isSeeded": "no"}, "isSeeded must be boolean"),
        ({**HEADER, "cwd": "relative"}, "cwd must be absolute"),
        ({**HEADER, "cwd": 7}, "cwd must be absolute"),
        ({**HEADER, "parentSession": 7}, "parentSession must be a string"),
        ({**HEADER, "agentPreset": 7}, "agentPreset must be a string"),
        ({**HEADER, "origin": "parent"}, 'origin must be "subagent"'),
    ]
    for value, message in malformed:
        with pytest.raises(SessionFormatError) as caught:
            assert_released_v4_header(value)
        assert message in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects released wrapper rows before structural recovery can admit them"
)
def test_rejects_released_wrapper_rows() -> None:
    wrapper_row: dict[str, Any] = {
        "type": "tool/result",
        "seq": 0,
        "time": 1,
        "data": {"message": {"role": "user"}},
    }
    with pytest.raises(SessionFormatError, match=r"first-class message"):
        assert_v4_row_admission(wrapper_row)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > retains predecessor and future delivery generations without activating"
    " their watermarks"
)
def test_retains_predecessor_and_future_delivery_generations() -> None:
    for version in [None, 0, 1, 2, 3, 5]:
        marker = {
            **delivery(version, "foreign"),
            "data": {**delivery(version)["data"], "sessionId": "foreign", "throughSeq": 500},
        }
        assert restore([FACT, marker]).events[1] is marker
    current = delivery(4)
    assert restore([FACT, current]).events[1] is current
    with pytest.raises(SessionFormatError, match="wrong Session"):
        restore([FACT, delivery(4, "foreign")])


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects malformed active delivery coordinates and malformed generation"
    " identifiers"
)
def test_rejects_malformed_active_delivery_coordinates() -> None:
    marker = delivery(4)
    data = marker["data"]
    for through_seq in [-1, 1, 1.5]:
        with pytest.raises(SessionFormatError, match="throughSeq"):
            restore([FACT, {**marker, "data": {**data, "throughSeq": through_seq}}])
    for session_id in ["", None, 3]:
        with pytest.raises(SessionFormatError, match="nonempty"):
            restore([FACT, {**marker, "data": {**data, "sessionId": session_id}}])
    for session_format_version in [-1, -0.0, 1.5, "4", None]:
        with pytest.raises(SessionFormatError, match="sessionFormatVersion"):
            restore(
                [FACT, {**marker, "data": {**data, "sessionFormatVersion": session_format_version}}]
            )
    with pytest.raises(SessionFormatError, match="data must be an object"):
        restore([FACT, {**marker, "data": None}])
    reader = session_format_catalog.create_restore(
        {"type": "session", **HEADER},
        SessionFormatRestoreOptions(recovery="strict", validation="current"),
    )
    reader.decode_row(FACT)
    reader.decode_row({**marker, "data": {**data, "throughSeq": 1}})
    with pytest.raises(SessionFormatError, match="throughSeq"):
        reader.finish()


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > preserves inherited and local events and admits foreign active deliveries"
    " inside the seed only"
)
def test_preserves_inherited_and_local_events() -> None:
    inherited = delivery(4, "ancestor")
    artifact = SessionFormatArtifact(
        header={**HEADER, "isSeeded": True, "parentSession": "ancestor"},
        inherited_event_count=2,
        events=[
            FACT,
            inherited,
            {"type": "session/end-seed", "seq": 2, "time": 3, "data": {"inherited": True}},
            {**FACT, "seq": 3},
        ],
    )
    assert restore_released_v4_artifact(artifact, TYPES) is artifact
    with pytest.raises(SessionFormatError, match="marker"):
        restore_released_v4_artifact(replace(artifact, inherited_event_count=1), TYPES)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects invalid inherited cuts, sparse rows, and unseeded inherited markers"
)
def test_rejects_invalid_inherited_cuts_and_sparse_rows() -> None:
    with pytest.raises(SessionFormatError, match="exceeds its events"):
        restore_released_v4_artifact(
            SessionFormatArtifact(header=HEADER, inherited_event_count=1, events=[]), TYPES
        )
    with pytest.raises(SessionFormatError, match="unseeded format v4 Session has inherited events"):
        restore_released_v4_artifact(
            SessionFormatArtifact(header=HEADER, inherited_event_count=1, events=[FACT]), TYPES
        )
    with pytest.raises(SessionFormatError, match="not dense"):
        restore_released_v4_artifact(
            SessionFormatArtifact(
                header=HEADER, inherited_event_count=0, events=[{**FACT, "seq": 1}]
            ),
            TYPES,
        )
    with pytest.raises(
        SessionFormatError, match="unseeded Session contains an inherited end-seed marker"
    ):
        restore_released_v4_artifact(
            SessionFormatArtifact(
                header=HEADER,
                inherited_event_count=0,
                events=[
                    {"type": "session/end-seed", "seq": 0, "time": 1, "data": {"inherited": True}}
                ],
            ),
            TYPES,
        )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > preserves unknown ignorable values and requires installed vocabulary for"
    " unknown required events"
)
def test_preserves_unknown_ignorable_values() -> None:
    opaque = {
        "type": "external/opaque",
        "seq": 0,
        "time": 1,
        "data": {"untouched": ["a", 3]},
        "ignorable": True,
    }
    assert restore([opaque]).events[0] is opaque
    required: dict[str, Any] = {"type": "external/required", "seq": 0, "time": 1, "data": None}
    with pytest.raises(SessionFormatError, match="unknown event type"):
        restore([required])
    artifact = SessionFormatArtifact(header=HEADER, events=[required], inherited_event_count=0)
    assert restore_released_v4_artifact(artifact, frozenset([required["type"]])) is artifact


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses wrapper-shaped and malformed native tool results before recovery"
)
def test_refuses_wrapper_shaped_and_malformed_tool_results() -> None:
    wrapper = {
        "type": "tool/result",
        "seq": 0,
        "time": 1,
        "data": {
            "message": {
                "id": "result",
                "role": "user",
                "source": {"kind": "tool", "callId": "call"},
                "content": [],
            }
        },
    }
    malformed = {
        "type": "tool/result",
        "seq": 0,
        "time": 1,
        "data": {
            "message": {
                "id": "result",
                "role": "tool",
                "source": {"kind": "tool", "callId": "call"},
                "toolCallId": "call",
                "content": [],
                "isError": "yes",
            }
        },
    }
    for row in [wrapper, malformed]:
        with pytest.raises(SessionFormatError):
            assert_v4_row_admission(row)
        decoder = released_v4_session_format_codec.create_decoder(
            {"type": "session", **HEADER}, "recoverable"
        )
        output = SessionFormatEventCollector()
        decoder.decode_row(None, output)
        with pytest.raises(SessionFormatError):
            decoder.decode_row(row, output)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > leaves ordinary framing recovery to the released decoder")
def test_leaves_ordinary_framing_recovery_to_released_decoder() -> None:
    assert_v4_row_admission(None)
    assert_v4_row_admission(FACT)
    decoder = released_v4_session_format_codec.create_decoder(
        {"type": "session", **HEADER}, "recoverable"
    )
    output = SessionFormatEventCollector()
    decoder.decode_row(FACT, output)
    decoder.decode_row(None, output)
    assert decoder.finish(output) == 0
    assert output.values == [FACT]


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects duplicate own catalog membership")
def test_rejects_duplicate_own_catalog_membership() -> None:
    with pytest.raises(SessionFormatError, match="duplicate catalog child"):
        restore([CATALOG, {**CATALOG, "seq": 1}])


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps inherited catalog entries outside own uniqueness checks"
)
def test_keeps_inherited_catalog_entries_outside_uniqueness_checks() -> None:
    artifact = SessionFormatArtifact(
        header={**HEADER, "isSeeded": True, "parentSession": "ancestor"},
        inherited_event_count=1,
        events=[
            CATALOG,
            {"type": "session/end-seed", "seq": 1, "time": 2, "data": {"inherited": True}},
            {**CATALOG, "seq": 2},
        ],
    )
    assert restore_released_v4_artifact(artifact, TYPES) is artifact


@pytest.mark.upstream(
    f"{SPEC}::restores external membership and rejects duplicate catalog children"
)
def test_restores_external_membership() -> None:
    data = {**CATALOG["data"], "version": 2, "mode": "external", "label": "External"}
    entry = {**CATALOG, "data": data}
    assert restore([entry]).events == [entry]
    with pytest.raises(SessionFormatError, match="duplicate catalog child"):
        restore([entry, {**entry, "seq": 1}])


@pytest.mark.upstream(f"{SPEC}::rejects a catalog mode outside its payload version: %j")
@pytest.mark.parametrize(
    "descriptor",
    [
        {"version": 0, "mode": "external"},
        {"version": 1, "mode": "external"},
        {"version": 2, "mode": "one-shot"},
        {"version": 2, "mode": "continuable"},
        {"version": 2, "mode": "unknown"},
    ],
)
def test_rejects_catalog_mode_outside_payload_version(descriptor: dict[str, Any]) -> None:
    data = {**CATALOG["data"], **descriptor}
    with pytest.raises(SessionFormatError, match="supported versioned catalog fact"):
        restore([{**CATALOG, "data": data}])
