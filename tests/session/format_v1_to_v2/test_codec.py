"""Ported from packages/session/session-format-v1-to-v2/tests/codec.spec.ts.

The upstream Proxy-based row failures become a mapping whose iteration raises; a
thrown non-``Error`` primitive has no Python equivalent and is left out of that case.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from deskagents.core.session import decode_seq_ranges
from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEventCollector,
    SessionFormatRecovery,
)
from deskagents.session.format_v1_to_v2 import released_v2_session_format_codec as codec

SPEC = "packages/session/session-format-v1-to-v2/tests/codec.spec.ts"
HEADERS = "releasedV2SessionFormatCodec headers"
ROWS = "releasedV2SessionFormatCodec rows"

MINIMAL: dict[str, Any] = {
    "type": "session",
    "version": 2,
    "id": "codec",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}
FULL: dict[str, Any] = {
    **MINIMAL,
    "cwd": "/work",
    "parentSession": "parent",
    "origin": "subagent",
    "agentPreset": "default",
}
TEXT = {"type": "text", "text": "text"}


def feedback(seq: int) -> dict[str, Any]:
    return {
        "type": "feedback/record",
        "seq": seq,
        "time": seq + 1,
        "data": {"text": f"feedback-{seq}"},
    }


def user_message(seq: int, source_event_seqs: list[int] | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "user/message",
        "seq": seq,
        "time": seq + 1,
        "data": {
            "id": f"user-{seq}",
            "role": "user",
            "content": [TEXT],
            "source": {"kind": "user"},
        },
        "surfaceOp": "append",
    }
    if source_event_seqs is not None:
        event["sourceEventSeqs"] = source_event_seqs
    return event


def artifact(events: Sequence[Any], **overrides: Any) -> SessionFormatArtifact:
    header = {"version": 2, "id": "codec", "createdAt": 1, "isSeeded": False, "delegationDepth": 0}
    return SessionFormatArtifact(
        overrides.get("header", header), overrides.get("inherited_event_count", 0), list(events)
    )


def decode_v2(
    header: object, rows: Sequence[object], recovery: SessionFormatRecovery = "strict"
) -> SessionFormatArtifact:
    decoder = codec.create_decoder(header, recovery)
    context = SessionFormatEventCollector()
    for row in rows:
        decoder.decode_row(row, context)
    return SessionFormatArtifact(decoder.header, decoder.finish(context), context.values)


def encode_v2(source: SessionFormatArtifact) -> dict[str, Any]:
    return {
        "header": codec.encode_header(source.header, source.inherited_event_count),
        "rows": [codec.encode_event(event) for event in source.events],
    }


class Explosive(Mapping[str, Any]):
    """Stands in for upstream's Proxy whose ``ownKeys`` trap throws."""

    def __getitem__(self, key: str) -> Any:
        return feedback(1)[key]

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("proxy failure")

    def __len__(self) -> int:
        return 4

    def __contains__(self, key: object) -> bool:
        return key in feedback(1)


@pytest.mark.upstream(
    f"{SPEC}::{HEADERS} > round-trips the minimal and complete optional header images"
)
def test_round_trips_header_images() -> None:
    assert codec.version == 2
    assert codec.decode_header(MINIMAL) == {
        "version": 2,
        "id": "codec",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    full = codec.decode_header(FULL)
    assert list(full.items()) == [
        ("version", 2),
        ("id", "codec"),
        ("createdAt", 1),
        ("cwd", "/work"),
        ("parentSession", "parent"),
        ("isSeeded", False),
        ("origin", "subagent"),
        ("delegationDepth", 0),
        ("agentPreset", "default"),
    ]
    assert encode_v2(artifact([])) == {"header": MINIMAL, "rows": []}
    encoded = codec.encode_header(full, 0)
    assert list(encoded) == [
        "type",
        "version",
        "id",
        "createdAt",
        "cwd",
        "parentSession",
        "isSeeded",
        "origin",
        "delegationDepth",
        "agentPreset",
    ]
    assert encoded == FULL
    with pytest.raises(SessionFormatError, match="unseeded.*inherited events"):
        codec.encode_header(artifact([]).header, 1)


_WITHOUT_ID = {key: value for key, value in MINIMAL.items() if key != "id"}


@pytest.mark.upstream(f"{SPEC}::{HEADERS} > refuses a malformed physical header: %s")
@pytest.mark.parametrize(
    ("header", "message"),
    [
        pytest.param(None, "must be an object", id="null"),
        pytest.param([], "must be an object", id="array"),
        pytest.param("session", "must be an object", id="scalar"),
        pytest.param(_WITHOUT_ID, "lacks id", id="actually absent required member"),
        pytest.param(
            {**MINIMAL, "seedLength": 0}, "unexpected field seedLength", id="unexpected member"
        ),
        pytest.param({**MINIMAL, "type": "other"}, "expected released v2", id="wrong type tag"),
        pytest.param({**MINIMAL, "version": 1}, "expected released v2", id="wrong version"),
        pytest.param({**MINIMAL, "id": 1}, "id must be a string", id="non-string id"),
        pytest.param({**MINIMAL, "createdAt": -1}, "createdAt", id="negative creation time"),
        pytest.param(
            {**MINIMAL, "delegationDepth": -1}, "delegationDepth", id="negative delegation depth"
        ),
        pytest.param(
            {**MINIMAL, "isSeeded": 0}, "isSeeded must be boolean", id="non-boolean lineage"
        ),
        pytest.param({**MINIMAL, "cwd": 1}, "cwd must be a string", id="non-string cwd"),
        pytest.param({**MINIMAL, "cwd": "relative"}, "cwd must be absolute", id="relative cwd"),
        pytest.param(
            {**MINIMAL, "parentSession": 1},
            "parentSession must be a string",
            id="non-string parent",
        ),
        pytest.param(
            {**MINIMAL, "agentPreset": 1}, "agentPreset must be a string", id="non-string preset"
        ),
        pytest.param({**MINIMAL, "origin": "user"}, 'origin must be "subagent"', id="bad origin"),
    ],
)
def test_refuses_malformed_physical_header(header: object, message: str) -> None:
    with pytest.raises(SessionFormatError, match=message):
        codec.decode_header(header)


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > stores one event per row and compacts scalar, pair, and long-run "
    "source-event references exactly"
)
def test_compacts_source_event_references() -> None:
    source = artifact(
        [feedback(0), user_message(1, [0]), user_message(2, [0, 1]), user_message(3, [0, 1, 2])]
    )
    encoded = encode_v2(source)
    assert [row.get("sourceEventSeqs") for row in encoded["rows"]] == [None, [0], [0, 1], [[0, 2]]]
    assert "sourceEventSeqs" not in encoded["rows"][0]
    assert decode_v2(encoded["header"], encoded["rows"]) == source


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > keeps non-monotonic source-event references scalar-only for the current "
    "backend reader"
)
def test_keeps_non_monotonic_references_scalar() -> None:
    seqs = [4, 5, 1, 2, 3]
    source = artifact([*(feedback(seq) for seq in range(6)), user_message(6, seqs)])
    encoded = encode_v2(source)
    stored = encoded["rows"][6]["sourceEventSeqs"]
    assert stored == seqs
    assert list(decode_seq_ranges(stored, 6)) == seqs
    assert decode_v2(encoded["header"], encoded["rows"]) == source


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > keeps the v2 physical codec vocabulary-neutral for current growth and a "
    "future source freeze"
)
def test_codec_is_vocabulary_neutral() -> None:
    source = artifact(
        [
            {"type": "external/required", "seq": 0, "time": 1, "data": {"retained": True}},
            {
                "type": "external/ignorable",
                "seq": 1,
                "time": 2,
                "data": {"retained": True},
                "ignorable": True,
            },
            {
                "type": "turn/start",
                "seq": 2,
                "time": 3,
                "data": {"turn": 1, "postReleaseMember": True},
            },
        ]
    )
    encoded = encode_v2(source)
    assert decode_v2(encoded["header"], encoded["rows"]) == source


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > expands mixed stored source-event ranges and preserves unrelated rows"
)
def test_expands_mixed_ranges() -> None:
    rows = [
        *(feedback(seq) for seq in range(5)),
        {**user_message(5), "sourceEventSeqs": [[0, 2], 4]},
    ]
    decoded = decode_v2(MINIMAL, rows)
    assert decoded.events[0] == feedback(0)
    assert decoded.events[5]["sourceEventSeqs"] == [0, 1, 2, 4]
    descending = decode_v2(
        MINIMAL, [feedback(0), feedback(1), {**user_message(2), "sourceEventSeqs": [1, 0]}]
    )
    assert descending.events[2]["sourceEventSeqs"] == [1, 0]


@pytest.mark.upstream(f"{SPEC}::{ROWS} > refuses malformed stored source-event ranges: %s")
@pytest.mark.parametrize(
    ("source_event_seqs", "message"),
    [
        pytest.param("bad", "must be an array", id="not an array"),
        pytest.param([-1], "sourceEventSeqs member", id="invalid scalar"),
        pytest.param([[0]], r"must be a \[start, end\] pair", id="malformed range"),
        pytest.param([[-1, 0]], "range start", id="invalid range start"),
        pytest.param([[0, -1]], "range end", id="invalid range end"),
        pytest.param([[2, 1]], "range exceeds", id="reversed range"),
        pytest.param([[0, 4]], "range exceeds", id="range ending at the event"),
        pytest.param(
            [0, 1, 2, [0, 1]], "range exceeds", id="ranges with too many expanded members"
        ),
        pytest.param([4], "unique earlier", id="scalar at the event"),
        pytest.param([0, 0], "unique earlier", id="duplicate scalars"),
        pytest.param([[0, 1], 1], "unique earlier", id="overlapping range and scalar"),
        pytest.param([3, [0, 2]], "strictly increasing", id="non-monotonic range"),
    ],
)
def test_refuses_malformed_ranges(source_event_seqs: object, message: str) -> None:
    rows = [
        *(feedback(seq) for seq in range(4)),
        {**user_message(4), "sourceEventSeqs": source_event_seqs},
    ]
    with pytest.raises(SessionFormatError, match=message):
        decode_v2(MINIMAL, rows)


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > contains ordinary and non-SessionFormatError row failures in a "
    "recoverable tail"
)
def test_contains_row_failures_in_recoverable_tail() -> None:
    assert decode_v2(MINIMAL, [feedback(0), Explosive()], "recoverable").events == [feedback(0)]
    assert decode_v2(MINIMAL, [feedback(0), None], "recoverable").events == [feedback(0)]
    assert decode_v2(MINIMAL, [None, [], feedback(0)], "recoverable").events == []
    with pytest.raises(SessionFormatError, match="row 1 is malformed") as raised:
        decode_v2(MINIMAL, [feedback(0), Explosive()])
    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > refuses malformed strict rows, strict gaps, and terminal recoverable tails"
)
def test_refuses_malformed_strict_rows() -> None:
    turn_end = {"type": "turn/end", "time": 1, "data": {"turn": 1, "reason": {"kind": "completed"}}}
    with pytest.raises(SessionFormatError, match="row 0"):
        decode_v2(MINIMAL, [None])
    with pytest.raises(SessionFormatError, match="seq gap"):
        decode_v2(MINIMAL, [feedback(1)])
    with pytest.raises(SessionFormatError, match="seq gap"):
        decode_v2(MINIMAL, [{**turn_end, "seq": 1}], "recoverable")
    with pytest.raises(SessionFormatError, match="row 0"):
        decode_v2(MINIMAL, [None, {**turn_end, "seq": 0}], "recoverable")
    with pytest.raises(SessionFormatError, match="unexpected field extra"):
        decode_v2(MINIMAL, [{**feedback(0), "extra": True}])
    with pytest.raises(SessionFormatError, match="lacks required field data"):
        decode_v2(MINIMAL, [{"type": "feedback/record", "seq": 0, "time": 1}])
    with pytest.raises(SessionFormatError, match="type must be a string"):
        decode_v2(MINIMAL, [{**feedback(0), "type": 1}])
    with pytest.raises(SessionFormatError, match="ignorable must be true"):
        decode_v2(MINIMAL, [{**feedback(0), "ignorable": False}])


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > keeps the prefix before a recoverable non-terminal seq gap"
)
def test_keeps_prefix_before_gap() -> None:
    decoded = decode_v2(MINIMAL, [feedback(0), feedback(3), feedback(1)], "recoverable")
    assert decoded.events == [feedback(0)]


@pytest.mark.upstream(
    f"{SPEC}::{ROWS} > derives the last tagged seed marker and rejects lineage disagreements"
)
def test_derives_last_seed_marker() -> None:
    seeded = {**MINIMAL, "id": "seeded", "isSeeded": True}
    markers: list[dict[str, Any]] = [
        {"type": "session/end-seed", "seq": 0, "time": 1, "data": {"inherited": True}},
        {"type": "session/end-seed", "seq": 1, "time": 2, "data": {}},
        {"type": "session/end-seed", "seq": 2, "time": 3, "data": {"inherited": True}},
    ]
    assert decode_v2(seeded, markers).inherited_event_count == 2
    assert decode_v2(seeded, [markers[0]]).inherited_event_count == 0
    with pytest.raises(SessionFormatError, match="lacks an inherited end-seed marker"):
        decode_v2(seeded, [])
    with pytest.raises(
        SessionFormatError, match="unseeded Session contains an inherited end-seed marker"
    ):
        decode_v2(MINIMAL, [markers[0]])
    with pytest.raises(SessionFormatError, match="must be an object"):
        decode_v2(seeded, [{"type": "session/end-seed", "seq": 0, "time": 1, "data": None}])


def test_seq_comparison_never_treats_true_as_one() -> None:
    """Python-specific: JavaScript ``true !== 1``, so a boolean seq is a gap."""
    with pytest.raises(SessionFormatError, match="seq gap"):
        decode_v2(MINIMAL, [feedback(0), {**feedback(1), "seq": True}])
