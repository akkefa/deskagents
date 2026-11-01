"""Ported from packages/session/session-persistence-jsonl/tests/jsonl.spec.ts.

Covers the "scanLog unit" and "nested Assistant streams" describes. Not applicable:
``reads v%s rows and appends a current turn without changing predecessor bytes`` (it
migrates v2/v3 generations, which Python does not read).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from deepseek_cordis import Context

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
    SessionStore,
)
from deskagents.llm import create_message
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import SessionPersistence
from deskagents.session.persistence_jsonl.format import (
    SessionLogScanner,
    event_lines,
    log_path,
    scan_log,
    to_header_line,
)
from deskagents.util.values import MAX_SAFE_INTEGER, parse_json, stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/jsonl.spec.ts"
SCAN = f"{SPEC}::JsonlSessionPersistence: scanLog unit"
NESTED = f"{SPEC}::JsonlSessionPersistence: nested Assistant streams"


def meta(id_: str, cwd: str | None = None) -> dict[str, Any]:
    header: dict[str, Any] = {
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId(id_),
        "createdAt": 1000,
        "isSeeded": False,
    }
    if cwd is not None:
        header["cwd"] = cwd
    return header


def one_turn_log() -> list[dict[str, Any]]:
    return [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "user/message",
            "seq": 1,
            "time": 2,
            "data": {
                "id": "one-turn-user",
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {"type": "step/start", "seq": 2, "time": 3, "data": {"turn": 1, "step": 1}},
        {
            "type": "assistant/message",
            "seq": 3,
            "time": 4,
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "one-turn-assistant",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "source": {"kind": "model", "provider": "mock", "model": "mock"},
                },
                "stream": [
                    {
                        "type": "chunk",
                        "time": 3,
                        "chunk": {"type": "block-start", "index": 0, "blockType": "text"},
                    },
                    {"type": "text-chunks", "time0": 3, "index": 0, "dt": [], "texts": ["hello"]},
                    {
                        "type": "chunk",
                        "time": 4,
                        "chunk": {
                            "type": "block-end",
                            "index": 0,
                            "block": {"type": "text", "text": "hello"},
                        },
                    },
                    {
                        "type": "chunk",
                        "time": 4,
                        "chunk": {"type": "finish", "reason": {"kind": "stop"}},
                    },
                ],
            },
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 4, "time": 5, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 5,
            "time": 6,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]


def js(value: object) -> str:
    return stringify_json(value)


def header_bytes(header: SessionHeader, cut: int | None = None) -> bytes:
    return f"{js(to_header_line(header, cut))}\n".encode()


def current_header(id_: str) -> dict[str, Any]:
    return {
        "type": "session",
        "version": SESSION_FORMAT_VERSION,
        "id": id_,
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }


def rows(*values: object) -> bytes:
    return ("\n".join(v if isinstance(v, str) else js(v) for v in values) + "\n").encode()


def seqs(events: Sequence[SessionEvent]) -> list[int]:
    return [event["seq"] for event in events]


def matches(actual: object, expected: object) -> bool:
    """Vitest ``toMatchObject``: every expected key matches, recursively."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and matches(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, Sequence)
            and not isinstance(actual, str)
            and len(actual) == len(expected)
            and all(matches(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return actual == expected


# --- scanLog unit ------------------------------------------------------------------------


@pytest.mark.upstream(f"{SCAN} > requires exactly one newline-terminated header record")
def test_requires_exactly_one_newline_terminated_header_record() -> None:
    header = js(to_header_line(meta("scanner-header")))
    with pytest.raises(ValueError, match="header-less"):
        SessionLogScanner(b"")
    with pytest.raises(ValueError, match="header-less"):
        SessionLogScanner(header.encode())
    with pytest.raises(ValueError, match="header-less"):
        SessionLogScanner(f"{header}\n{header}\n".encode())


@pytest.mark.upstream(f"{SCAN} > fails immediately on an invalid row in strict scanner mode")
def test_fails_immediately_on_an_invalid_row_in_strict_scanner_mode() -> None:
    scanner = SessionLogScanner(header_bytes(meta("scanner-strict")), "strict")
    with pytest.raises(ValueError, match="invalid committed event"):
        scanner.write(b"null\n")


@pytest.mark.upstream(f"{SCAN} > expands valid stored source-event ranges")
def test_expands_valid_stored_source_event_ranges() -> None:
    log = "\n".join(
        [
            js(to_header_line(meta("scanner-source-ranges"))),
            js({"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}),
            js(one_turn_log()[1]),
            js({**one_turn_log()[1], "seq": 2, "time": 3, "sourceEventSeqs": [[0, 1]]}),
            "",
        ]
    )
    restored = scan_log(log.encode()).events[2]
    assert matches(
        restored, {"type": "user/message", "surfaceOp": "append", "sourceEventSeqs": [0, 1]}
    )


@pytest.mark.upstream(
    f"{SCAN} > requires the tagged inherited cut to agree with the current header lineage"
)
def test_requires_the_tagged_inherited_cut_to_agree_with_the_current_header_lineage() -> None:
    seeded = {**meta("scanner-seeded-cut"), "isSeeded": True}
    with pytest.raises(Exception, match="seeded Session lacks an inherited end-seed marker"):
        scan_log(header_bytes(seeded, SessionLogOffset(0)))

    unseeded = header_bytes(meta("scanner-unseeded-cut"))
    marker = js({"type": "session/end-seed", "seq": 0, "time": 1, "data": {"inherited": True}})
    with pytest.raises(Exception, match="unseeded Session contains an inherited end-seed marker"):
        scan_log(unseeded + f"{marker}\n".encode())


@pytest.mark.upstream(
    f"{SCAN} > handles empty writes, boundary newlines, torn fragments, and scanner completion"
)
def test_handles_empty_writes_boundary_newlines_torn_fragments_and_completion() -> None:
    header = header_bytes(meta("scanner-lifecycle"))
    event = js(one_turn_log()[0]).encode()
    scanner = SessionLogScanner(header)

    scanner.write(b"")
    scanner.write(event)
    scanner.write(b"\nignored torn tail")
    result = scanner.finish()

    assert result.events == [one_turn_log()[0]]
    assert result.committed_bytes == len(header) + len(event) + 1
    with pytest.raises(ValueError, match="finished"):
        scanner.write(b"\n")


@pytest.mark.upstream(
    f"{SCAN} > keeps scanning after a tolerable corrupt suffix until a committed turn end appears"
)
def test_keeps_scanning_after_a_tolerable_corrupt_suffix() -> None:
    header = header_bytes(meta("scanner-corrupt-suffix"))
    scanner = SessionLogScanner(header)
    scanner.write(
        "\n".join(
            [
                js(one_turn_log()[0]),
                "{not json",
                js({"type": "step/start", "seq": 1, "time": 2, "data": {"turn": 1, "step": 1}}),
                "",
            ]
        ).encode()
    )
    assert scanner.finish().events == [one_turn_log()[0]]

    committed = SessionLogScanner(header)
    with pytest.raises(ValueError, match="seq gap"):
        committed.write(rows(turn_end(1, 2)))


@pytest.mark.upstream(f"{SCAN} > incrementally scans records split across reusable decoder chunks")
def test_incrementally_scans_records_split_across_reusable_decoder_chunks() -> None:
    header = header_bytes(meta("incremental"))
    text = "\n".join(js(event) for event in one_turn_log()).replace('"hi"', '"你好"', 1)
    body = f"{text}\n".encode()
    split = body.index("你".encode()) + 1
    first_chunk = bytearray(body[:split])
    scanner = SessionLogScanner(header)

    scanner.write(bytes(first_chunk))
    checkpoint = scanner.checkpoint()
    first_chunk[:] = bytes(len(first_chunk))
    scanner.write(body[split:])

    assert checkpoint.input_bytes == len(header) + split
    assert checkpoint.event_count == 1
    assert scanner.finish() == scan_log(header + body)


@pytest.mark.upstream(f"{SCAN} > rejects a header-less / empty log")
def test_rejects_a_header_less_empty_log() -> None:
    with pytest.raises(ValueError):
        scan_log(b"")


@pytest.mark.upstream(f"{SCAN} > rejects a corrupt header line")
def test_rejects_a_corrupt_header_line() -> None:
    with pytest.raises(ValueError, match="header"):
        scan_log(b"not json\n")


@pytest.mark.upstream(f"{SCAN} > rejects a non-session first line")
def test_rejects_a_non_session_first_line() -> None:
    with pytest.raises(ValueError, match="session header"):
        scan_log(b'{"type":"event"}\n')


@pytest.mark.upstream(f"{SCAN} > rejects a session header with a %s createdAt")
@pytest.mark.parametrize(
    "created_at",
    [1.5, -1, MAX_SAFE_INTEGER + 1],
    ids=["fractional", "negative", "unsafe"],
)
def test_rejects_a_session_header_with_an_invalid_created_at(created_at: float) -> None:
    # json.dumps, not stringify_json: JSON.stringify writes an unsafe integer as-is.
    log = json.dumps(
        {
            "type": "session",
            "version": SESSION_FORMAT_VERSION,
            "id": "invalid-created-at",
            "createdAt": created_at,
            "delegationDepth": 0,
        },
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="session header"):
        scan_log(f"{log}\n".encode())


@pytest.mark.upstream(f"{SCAN} > rejects a session header with negative-zero createdAt")
def test_rejects_a_session_header_with_negative_zero_created_at() -> None:
    log = (
        f'{{"type":"session","version":{SESSION_FORMAT_VERSION},"id":"invalid-created-at",'
        '"createdAt":-0,"delegationDepth":0}\n'
    )
    with pytest.raises(ValueError, match="session header"):
        scan_log(log.encode())


_ABSENT = object()


@pytest.mark.upstream(f"{SCAN} > rejects a session header with %s delegationDepth")
@pytest.mark.parametrize(
    "delegation_depth",
    [_ABSENT, "1", 1.5, -1],
    ids=["missing", "a string", "fractional", "negative"],
)
def test_rejects_a_session_header_with_invalid_delegation_depth(delegation_depth: object) -> None:
    line: dict[str, Any] = {
        "type": "session",
        "version": SESSION_FORMAT_VERSION,
        "id": "invalid-depth",
        "createdAt": 1,
    }
    if delegation_depth is not _ABSENT:
        line["delegationDepth"] = delegation_depth
    with pytest.raises(ValueError, match="session header"):
        scan_log(f"{js(line)}\n".encode())


@pytest.mark.upstream(f"{SCAN} > rejects a session header with negative-zero delegationDepth")
def test_rejects_a_session_header_with_negative_zero_delegation_depth() -> None:
    log = (
        f'{{"type":"session","version":{SESSION_FORMAT_VERSION},"id":"invalid-depth",'
        '"createdAt":1,"delegationDepth":-0}\n'
    )
    with pytest.raises(ValueError, match="session header"):
        scan_log(log.encode())


@pytest.mark.upstream(f"{SCAN} > round-trips the agent preset a session was composed from")
def test_round_trips_the_agent_preset_a_session_was_composed_from() -> None:
    line = to_header_line(
        {
            "version": SESSION_FORMAT_VERSION,
            "id": SessionId("composed"),
            "isSeeded": False,
            "createdAt": 1,
            "delegationDepth": 0,
            "agentPreset": "minimal",
        }
    )
    assert scan_log(f"{js(line)}\n".encode()).meta["agentPreset"] == "minimal"


@pytest.mark.upstream(f"{SCAN} > rejects a session header whose agentPreset is not a string")
def test_rejects_a_session_header_whose_agent_preset_is_not_a_string() -> None:
    line = {
        "type": "session",
        "version": SESSION_FORMAT_VERSION,
        "id": "bad-preset",
        "createdAt": 1,
        "delegationDepth": 0,
        "agentPreset": 7,
    }
    with pytest.raises(ValueError, match="session header"):
        scan_log(f"{js(line)}\n".encode())


TURN_START = {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}


def turn_end(seq: int, time: int) -> dict[str, Any]:
    return {
        "type": "turn/end",
        "seq": seq,
        "time": time,
        "data": {"turn": 1, "reason": {"kind": "completed"}},
    }


@pytest.mark.upstream(
    f"{SCAN} > a seq gap after the last turn/end bounds the preserved tail "
    "(torn fragment tolerated)"
)
def test_a_seq_gap_after_the_last_turn_end_bounds_the_preserved_tail() -> None:
    log = rows(
        current_header("g"),
        TURN_START,
        {"type": "step/start", "seq": 2, "time": 2, "data": {"turn": 1, "step": 1}},
    )
    assert seqs(scan_log(log).events) == [0]


@pytest.mark.upstream(
    f"{SCAN} > rejects a seq gap BEFORE a later committed turn/end (committed data damaged)"
)
def test_rejects_a_seq_gap_before_a_later_committed_turn_end() -> None:
    log = rows(
        current_header("g2"),
        TURN_START,
        {"type": "step/start", "seq": 2, "time": 2, "data": {"turn": 1, "step": 1}},
        turn_end(3, 3),
    )
    with pytest.raises(ValueError, match="seq gap"):
        scan_log(log)


@pytest.mark.upstream(f"{SCAN} > rejects malformed %s message slots before exposing a native scan")
@pytest.mark.parametrize(
    ("type_", "data", "error"),
    [
        ("assistant/message", {}, r"assistant/message requires a message"),
        ("agent/inbox/spliced", {"inserted": None}, r"agent/inbox/spliced requires message array"),
        (
            "session/title-llm-request",
            {"messages": [None]},
            r"session/title-llm-request requires message objects",
        ),
    ],
)
def test_rejects_malformed_message_slots_before_exposing_a_native_scan(
    type_: str, data: dict[str, Any], error: str
) -> None:
    header = {**current_header("malformed-tail")}
    prefix = rows(header, TURN_START)
    malformed = rows({"type": type_, "seq": 1, "time": 2, "data": data})
    with pytest.raises(Exception, match=error):
        scan_log(prefix + malformed)
    committed = rows(turn_end(2, 3))
    with pytest.raises(Exception, match=error):
        scan_log(prefix + malformed + committed)


@pytest.mark.upstream(f"{SCAN} > rejects malformed native tool results before recovery (sealed=%s)")
@pytest.mark.parametrize("sealed", [False, True])
def test_rejects_malformed_native_tool_results_before_recovery(sealed: bool) -> None:
    values: list[object] = [
        current_header("malformed-tool"),
        TURN_START,
        {"type": "tool/result", "seq": 1, "time": 2, "data": {"message": []}},
    ]
    if sealed:
        values.append(turn_end(2, 3))
    with pytest.raises(Exception, match=r"tool/result.*message must be an object"):
        scan_log(rows(*values))


@pytest.mark.upstream(f"{SCAN} > rejects malformed records before a later committed turn/end")
def test_rejects_malformed_records_before_a_later_committed_turn_end() -> None:
    corrupt_records = [
        ("{not json", "unparsable committed event"),
        ("null", "invalid committed event"),
        (
            js({"type": "assistant/message", "sourceEventSeqs": [0], "data": {}}),
            "invalid committed event",
        ),
    ]
    for record, message in corrupt_records:
        log = rows(current_header("c"), record, turn_end(1, 2))
        with pytest.raises(ValueError, match=message):
            scan_log(log)


@pytest.mark.upstream(
    f"{SCAN} > a header-only log (no event lines at all) preserves nothing — "
    "committedBytes is the header"
)
def test_a_header_only_log_preserves_nothing() -> None:
    log = rows(current_header("h0"))
    scanned = scan_log(log)
    assert scanned.events == []
    assert scanned.committed_bytes == len(log)


@pytest.mark.upstream(f"{SCAN} > a corrupt line after the last turn/end bounds the preserved tail")
def test_a_corrupt_line_after_the_last_turn_end_bounds_the_preserved_tail() -> None:
    log = rows(current_header("c2"), TURN_START, "{not json")
    assert seqs(scan_log(log).events) == [0]


@pytest.mark.upstream(f"{SCAN} > tolerates a seq gap AFTER a turn/end (uncommitted tail)")
def test_tolerates_a_seq_gap_after_a_turn_end() -> None:
    log = rows(
        current_header("t"),
        TURN_START,
        turn_end(1, 2),
        {"type": "step/start", "seq": 9, "time": 3, "data": {"turn": 2, "step": 1}},
    )
    assert seqs(scan_log(log).events) == [0, 1]


# --- nested Assistant streams ------------------------------------------------------------


@pytest_asyncio.fixture
async def persistence(tmp_path: Path) -> AsyncIterator[SessionPersistence]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(jsonl.apply, {"root": str(tmp_path), "compression": "none"})
    service = ctx.get("sessionPersistence")
    assert isinstance(service, SessionPersistence)
    yield service
    await ctx.fiber.dispose()


def chunk_run_log() -> list[dict[str, Any]]:
    """A one-turn log whose message embeds a five-member text-delta run."""
    return [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": 1, "time": 2, "data": {"turn": 1, "step": 1}},
        {
            "type": "assistant/message",
            "seq": 2,
            "time": 8,
            "data": {
                "turn": 1,
                "step": 1,
                "message": create_message(
                    role="assistant",
                    content=[{"type": "text", "text": "t0t1t2t3t4"}],
                    source={"kind": "model", "provider": "mock", "model": "mock"},
                ),
                "stream": [
                    {
                        "type": "text-chunks",
                        "time0": 3,
                        "index": 0,
                        "dt": [1, 1, 1, 1],
                        "texts": ["t0", "t1", "t2", "t3", "t4"],
                    },
                    {
                        "type": "chunk",
                        "time": 8,
                        "chunk": {"type": "finish", "reason": {"kind": "stop"}},
                    },
                ],
            },
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": 3, "time": 9, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": 4,
            "time": 10,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]


def read_text(path: str) -> str:
    return Path(path).read_text("utf-8")


async def write_log(
    persistence: SessionPersistence, header: SessionHeader, events: Sequence[SessionEvent]
) -> None:
    handle = await persistence.create(header)
    try:
        await handle.append(events)
    finally:
        await handle.close()


async def read_all(persistence: SessionPersistence, id_: str) -> list[SessionEvent]:
    handle = await persistence.open(id_, "read")
    try:
        return list((await handle.read()).events)
    finally:
        await handle.close()


@pytest.mark.upstream(
    f"{NESTED} > writes one row per event with compact stream records nested in the message"
)
@pytest.mark.asyncio
async def test_writes_one_row_per_event_with_compact_stream_records_nested(
    persistence: SessionPersistence, tmp_path: Path
) -> None:
    m = meta("packed", "/work")
    log = chunk_run_log()
    await write_log(persistence, m, log)

    path = log_path(str(tmp_path), "/work", m["id"], "none")
    raw = [line for line in read_text(path).split("\n") if line]
    tags = [parse_json(line)["type"] for line in raw[1:]]  # type: ignore[index,call-overload]
    assert tags == ["turn/start", "step/start", "assistant/message", "step/end", "turn/end"]
    message: Any = parse_json(raw[3])
    assert [record["type"] for record in message["data"]["stream"]] == ["text-chunks", "chunk"]

    assert await read_all(persistence, m["id"]) == log


@pytest.mark.upstream(
    f"{NESTED} > scanLog rejects a removed top-level packed row before a committed boundary"
)
def test_scan_log_rejects_a_removed_top_level_packed_row() -> None:
    log = rows(
        current_header("rows"),
        TURN_START,
        {
            "type": "text-chunks",
            "seq0": 1,
            "time0": 2,
            "data": {"turn": 1, "step": 1, "index": 0, "dt": [1, 1], "texts": ["a", "b", "c"]},
        },
        turn_end(4, 5),
    )
    with pytest.raises(Exception, match=r"lacks .*seq"):
        scan_log(log)


@pytest.mark.upstream(
    f"{NESTED} > scanLog treats a malformed removed packed row as a committed seq hole"
)
def test_scan_log_treats_a_malformed_removed_packed_row_as_a_committed_seq_hole() -> None:
    log = rows(
        current_header("bad-row"),
        {
            "type": "text-chunks",
            "seq0": 0,
            "time0": 1,
            "data": {"turn": 1, "step": 1, "index": 0, "dt": [], "texts": ["a", "b"]},
        },
        turn_end(2, 3),
    )
    with pytest.raises(Exception, match=r"lacks .*seq"):
        scan_log(log)


@pytest.mark.upstream(
    f"{NESTED} > scanLog: a packed row with a mid-run seq gap after the last turn/end drops "
    "the whole row"
)
def test_scan_log_a_packed_row_with_a_mid_run_seq_gap_drops_the_whole_row() -> None:
    log = rows(
        current_header("row-gap"),
        TURN_START,
        {
            "type": "text-chunks",
            "seq0": 2,
            "time0": 2,
            "data": {"turn": 1, "step": 1, "index": 0, "dt": [1, 1], "texts": ["a", "b", "c"]},
        },
    )
    scanned = scan_log(log)
    assert seqs(scanned.events) == [0]
    header_and_turn = "\n".join(log.decode().split("\n")[:2]) + "\n"
    assert scanned.committed_bytes == len(header_and_turn.encode())


@pytest.mark.upstream(
    f"{NESTED} > eventLines keeps one event per line with nested compact stream records"
)
def test_event_lines_keeps_one_event_per_line_with_nested_compact_stream_records() -> None:
    log = chunk_run_log()
    text = event_lines(log)
    lines = text.split("\n")
    assert len(lines) == len(log)
    for line in lines:
        value: Any = parse_json(line)
        assert not re.search(r"-chunks$", value["type"])
    header = js(to_header_line(meta("packed", "/work"))) + "\n"
    assert scan_log((header + text + "\n").encode()).events == log
