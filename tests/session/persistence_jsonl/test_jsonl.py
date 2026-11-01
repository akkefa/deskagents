"""Ported from packages/session/session-persistence-jsonl/tests/jsonl.spec.ts (part 1).

Covers the "format helpers", "stored-format refusals" and "immutable format generations"
describes. Python reads and writes only format v4 and refuses v0-v3 generations instead of
migrating them, and it has no Zstandard encoding yet, so every generation case whose point
is a v0-v3 migration, a historical preparation/publication or a ``.zstd`` artifact is not
ported. ``runPersistenceContract`` is ported with the shared contract suite.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Callable
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
)
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.format import SessionFormatError
from deskagents.session.persistence import (
    SessionAlreadyExistsError,
    SessionFormatUnsupportedError,
    SessionPersistence,
    SessionPersistenceCorruptionError,
)
from deskagents.session.persistence_jsonl import JsonlSessionPersistence
from deskagents.session.persistence_jsonl.format import (
    assert_no_retired_header_fields,
    encode_segment,
    generation_log_filename,
    generation_log_path,
    log_path,
    parse_generation_log_filename,
    project_key,
    scan_log,
    session_dir,
    to_header_line,
)
from deskagents.util.values import FrozenJsonArray, FrozenJsonObject

SPEC = "packages/session/session-persistence-jsonl/tests/jsonl.spec.ts"
FORMAT = "JsonlSessionPersistence: format helpers"
REFUSALS = "JsonlSessionPersistence: stored-format refusals"
GENERATIONS = "JsonlSessionPersistence: immutable format generations"


# --- helpers (local copies of the upstream contract fixtures) ----------------------------


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


def released_v0_header(header: SessionHeader) -> dict[str, Any]:
    line: dict[str, Any] = {
        "type": "session",
        "version": 0,
        "id": header["id"],
        "createdAt": header["createdAt"],
    }
    if "cwd" in header:
        line["cwd"] = header["cwd"]
    if "parentSession" in header:
        line["parentSession"] = header["parentSession"]
    if header["isSeeded"]:
        line["seedLength"] = 0
    if "origin" in header:
        line["origin"] = header["origin"]
    line["delegationDepth"] = header.get("delegationDepth", 0)
    if "agentPreset" in header:
        line["agentPreset"] = header["agentPreset"]
    return line


def dumps(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def raw_log_path(root: str, cwd: str | None, id_: str) -> str:
    return log_path(root, cwd, id_, "none")


def historical_log_path(root: str, cwd: str | None, id_: str) -> str:
    return generation_log_path(root, cwd, id_, 0, "none")


def read_text(path: str) -> str:
    return Path(path).read_text()


def exists(path: str) -> bool:
    return os.path.exists(path)


def is_file(path: str) -> bool:
    return os.path.isfile(path)


def relative_to_cwd(path: str) -> str:
    return os.path.relpath(path, os.getcwd())


def write_file(path: str, content: str | bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = content.encode() if isinstance(content, str) else content
    Path(path).write_bytes(data)


async def write_log(
    persistence: SessionPersistence, header: SessionHeader, events: list[dict[str, Any]]
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


def is_frozen(value: object) -> bool:
    return isinstance(value, FrozenJsonObject | FrozenJsonArray)


async def mount(root: str) -> Context:
    ctx = Context()
    await ctx.plugin(jsonl.apply, {"root": root, "compression": "none"})
    return ctx


@pytest_asyncio.fixture
async def root(tmp_path: Path) -> str:
    return str(tmp_path / "root")


@pytest_asyncio.fixture
async def ctx(root: str) -> AsyncIterator[Context]:
    context = await mount(root)
    yield context
    await context.fiber.dispose()


@pytest_asyncio.fixture
async def persistence(ctx: Context) -> JsonlSessionPersistence:
    service = ctx.get("sessionPersistence")
    assert isinstance(service, JsonlSessionPersistence)
    return service


# --- format helpers ----------------------------------------------------------------------


@pytest.mark.upstream(f"{SPEC}::{FORMAT} > names and parses only canonical immutable generations")
def test_names_and_parses_only_canonical_generations() -> None:
    assert generation_log_filename(0, "none") == "session.jsonl"
    assert generation_log_filename(1, "none") == "session.v1.jsonl"
    assert generation_log_filename(27, "zstd") == "session.v27.jsonl.zstd"
    for invalid in [-1, -0.0, 1.5, 2**53]:
        with pytest.raises(SessionFormatError, match="non-negative safe integer"):
            generation_log_filename(invalid, "none")  # type: ignore[arg-type]
    assert parse_generation_log_filename("session.jsonl", "none") == 0
    assert parse_generation_log_filename("session.v1.jsonl", "none") == 1
    assert parse_generation_log_filename("session.v9007199254740992.jsonl", "none") is None
    for name in [
        "session.v0.jsonl",
        "session.v01.jsonl",
        "session.V1.jsonl",
        "session.v1.backup.jsonl",
        "session.migration.deadbeef.tmp.jsonl",
        "session.v1.jsonl.zstd",
    ]:
        assert parse_generation_log_filename(name, "none") is None


@pytest.mark.upstream(f"{SPEC}::{FORMAT} > ignores retired-header checks for non-object values")
def test_ignores_retired_header_checks_for_non_objects() -> None:
    assert_no_retired_header_fields(None)
    assert_no_retired_header_fields("header")
    with pytest.raises(ValueError, match="first line is not a JSON object"):
        scan_log(b"42\n")
    with pytest.raises(ValueError, match="first line is not a JSON object"):
        scan_log(b"null\n")
    with pytest.raises(ValueError, match="first line is not a session header"):
        scan_log(
            f"{dumps({'type': 'session', 'version': SESSION_FORMAT_VERSION, 'id': 123})}\n".encode()
        )
    future = {"type": "session", "version": SESSION_FORMAT_VERSION + 1, "id": 123}
    with pytest.raises(Exception) as caught:
        scan_log(f"{dumps(future)}\n".encode())
    assert f'session "123" uses log format v{SESSION_FORMAT_VERSION + 1}' in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{FORMAT} > encodeSegment neutralizes traversal, separators, and absolute paths"
)
def test_encode_segment_neutralizes_traversal() -> None:
    assert encode_segment("..") == "~002E~002E"
    assert encode_segment(".") == "~002E"
    assert encode_segment("a/b") == "a~002Fb"
    assert encode_segment("/etc/passwd") == "~002Fetc~002Fpasswd"
    assert encode_segment("a\u0000b") == "a~0000b"
    assert encode_segment("plain-ID_1.2") == "plain-ID_1.2"
    assert encode_segment("a~b") == "a~007Eb"


@pytest.mark.upstream(
    f"{SPEC}::{FORMAT} > encodeSegment is injective over UTF-16, incl. lone surrogates"
)
def test_encode_segment_is_injective_over_utf16() -> None:
    hi = encode_segment(chr(0xD800))
    lo = encode_segment(chr(0xDC00))
    assert hi == "~D800"
    assert lo == "~DC00"
    assert hi != lo
    assert encode_segment("~002F") != encode_segment("/")


@pytest.mark.upstream(f"{SPEC}::{FORMAT} > encodeSegment rejects an empty id")
def test_encode_segment_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_segment("")


def _seed_rows(count: int) -> list[dict[str, Any]]:
    return [
        {"type": "feedback/record", "seq": seq, "time": seq + 1, "data": {"text": f"prefix-{seq}"}}
        for seq in range(count)
    ]


@pytest.mark.upstream(
    f"{SPEC}::{FORMAT} > round-trips every optional header field through the header line"
)
def test_round_trips_every_optional_header_field() -> None:
    full: dict[str, Any] = {
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId("full-header"),
        "createdAt": 5,
        "cwd": "/w",
        "parentSession": SessionId("parent"),
        "isSeeded": True,
        "origin": "subagent",
        "delegationDepth": 2,
        "agentPreset": "minimal",
    }
    events = [
        *_seed_rows(3),
        {"type": "session/end-seed", "seq": 3, "time": 4, "data": {"inherited": True}},
    ]
    rows = [to_header_line(full, SessionLogOffset(3)), *events]
    scan = scan_log(("\n".join(dumps(row) for row in rows) + "\n").encode())
    assert scan.meta == full
    assert scan.inherited_event_count == 3


@pytest.mark.upstream(f"{SPEC}::{FORMAT} > round-trips current lineage markers for %s")
@pytest.mark.parametrize(
    ("case", "is_seeded", "inherited_event_count"),
    [("unseeded", False, 0), ("empty-seed", True, 0), ("nonempty-seed", True, 3)],
)
def test_round_trips_current_lineage_markers(
    case: str, is_seeded: bool, inherited_event_count: int
) -> None:
    events: list[dict[str, Any]] = (
        [
            *_seed_rows(inherited_event_count),
            {
                "type": "session/end-seed",
                "seq": inherited_event_count,
                "time": inherited_event_count + 1,
                "data": {"inherited": True},
            },
        ]
        if is_seeded
        else []
    )
    line = {
        "type": "session",
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId(f"physical-seed-{case}"),
        "createdAt": 1000,
        "isSeeded": is_seeded,
        "delegationDepth": 0,
    }
    data = ("\n".join(dumps(row) for row in [line, *events]) + "\n").encode()

    scanned = scan_log(data)

    assert scanned.meta["isSeeded"] is is_seeded
    assert scanned.inherited_event_count == SessionLogOffset(inherited_event_count)
    header_line = to_header_line(scanned.meta, scanned.inherited_event_count)
    assert dict(header_line) == line
    assert list(header_line) == list(line)


@pytest.mark.upstream(
    f"{SPEC}::{FORMAT} > rejects a current physical header without explicit isSeeded"
)
def test_rejects_current_header_without_is_seeded() -> None:
    line = dict(to_header_line(meta("missing-is-seeded")))
    del line["isSeeded"]
    with pytest.raises(ValueError, match="session header"):
        scan_log(f"{dumps(line)}\n".encode())


@pytest.mark.upstream(
    f"{SPEC}::{FORMAT} > requires logical lineage and the physical inherited cut to agree"
)
def test_requires_lineage_and_cut_to_agree() -> None:
    unseeded = meta("lineage-cut")
    with pytest.raises(ValueError, match="seeded session header requires an inherited event count"):
        to_header_line({**unseeded, "isSeeded": True})
    with pytest.raises(ValueError, match="unseeded session header inherited event count must be 0"):
        to_header_line(unseeded, SessionLogOffset(1))


@pytest.mark.upstream(
    f"{SPEC}::{FORMAT} > projectKey normalizes project paths into bounded readable names"
)
def test_project_key_normalizes_paths() -> None:
    assert project_key("/Users/qyj/work/deepseek-harness") == "--Users-qyj-work-deepseek-harness--"
    assert project_key("/a/b-c") == project_key("/a-b/c")
    assert project_key("C:\\work\\agent") == "--C-work-agent--"
    assert project_key("/开发/~agent") == "--~5F00~53D1-~007Eagent--"
    assert project_key("/") == "--root--"
    assert len(project_key("/" + "x" * 1_000)) == 255
    with pytest.raises(ValueError, match="empty project path"):
        project_key("")


@pytest.mark.upstream(
    f"{SPEC}::{FORMAT} > resolves a relative custom root before storing a session"
)
@pytest.mark.asyncio
async def test_resolves_relative_custom_root(tmp_path: Path) -> None:
    absolute_root = str(tmp_path / "relative-root")
    context = Context()
    fiber = await context.plugin(
        jsonl.apply, {"root": relative_to_cwd(absolute_root), "compression": "none"}
    )
    try:
        m = meta("relative-location", "/work")
        service = context.get("sessionPersistence")
        assert isinstance(service, JsonlSessionPersistence)
        await write_log(service, m, one_turn_log())
        assert is_file(raw_log_path(absolute_root, "/work", m["id"]))
    finally:
        await fiber.dispose()


# --- stored-format refusals --------------------------------------------------------------


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > propagates a non-format header failure from stat and list unchanged"
)
@pytest.mark.asyncio
async def test_propagates_non_format_header_failure(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    id_ = SessionId("retired-policy-header")
    line = {
        "type": "session",
        "version": SESSION_FORMAT_VERSION,
        "id": id_,
        "createdAt": 1,
        "delegationDepth": 0,
        "sandboxMode": "strict",
    }
    write_file(raw_log_path(root, "/work", id_), f"{dumps(line)}\n")
    with pytest.raises(ValueError, match="retired policy baseline fields"):
        await persistence.stat(id_)
    with pytest.raises(ValueError, match="retired policy baseline fields"):
        await persistence.list()


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > refuses a structurally foreign future header as unsupported, "
    "not corrupt or absent"
)
@pytest.mark.asyncio
async def test_refuses_structurally_foreign_future_header(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    id_ = SessionId("future-shape")
    path = generation_log_path(root, "/work", id_, 42, "none")
    header = {"type": "session", "version": 42, "id": id_, "futureOnly": True}
    write_file(path, f'{dumps(header)}\n{{"future":"row"}}\n')
    for access in ("read", "write"):
        with pytest.raises(SessionFormatUnsupportedError) as caught:
            await persistence.open(id_, access)
        assert caught.type is SessionFormatUnsupportedError
        message = str(caught.value)
        assert re.search(r"written by a newer harness.*upgrade the harness", message)
        assert f"(raw log: {path})" in message
    assert list(await persistence.list()) == []


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > classifies an unparsable future generation header as corruption"
)
@pytest.mark.asyncio
async def test_unparsable_future_header_is_corruption(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    id_ = SessionId("future-malformed")
    write_file(generation_log_path(root, "/work", id_, 42, "none"), "{not-json}\n")
    with pytest.raises(SessionPersistenceCorruptionError):
        await persistence.open(id_, "read")


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > refuses a well-shaped newer-version header at read open with the "
    "upgrade direction"
)
@pytest.mark.asyncio
async def test_refuses_well_shaped_newer_header(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    id_ = SessionId("future-version")
    path = generation_log_path(root, "/work", id_, 42, "none")
    header = {
        "type": "session",
        "version": 42,
        "id": id_,
        "createdAt": 1,
        "cwd": "/work",
        "delegationDepth": 0,
    }
    write_file(path, f"{dumps(header)}\n")
    with pytest.raises(SessionFormatUnsupportedError, match="upgrade the harness") as caught:
        await persistence.open(id_, "read")
    assert f"(raw log: {path})" in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > keeps a non-object header line a corruption, not a format refusal"
)
@pytest.mark.asyncio
async def test_non_object_header_is_corruption(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    id_ = SessionId("scalar-header")
    path = raw_log_path(root, "/work", id_)
    write_file(path, "42\n")
    with pytest.raises(SessionPersistenceCorruptionError) as caught:
        await persistence.open(id_, "read")
    assert "first line is not a JSON object" in str(caught.value)
    assert f"(raw log: {path})" in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > classifies a %s current log as corruption with its raw path"
)
@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("invalid JSON", "not json\n", "not valid JSON"),
        ("missing header newline", '{"type":"session"}', "header-less session log"),
    ],
)
@pytest.mark.asyncio
async def test_classifies_corrupt_current_log(
    root: str, persistence: JsonlSessionPersistence, name: str, content: str, reason: str
) -> None:
    id_ = SessionId(f"corrupt-{name.replace(' ', '-')}")
    path = raw_log_path(root, "/work", id_)
    write_file(path, content)
    with pytest.raises(SessionPersistenceCorruptionError, match=reason) as caught:
        await persistence.open(id_, "read")
    assert f"(raw log: {path})" in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > names a foreign-version header by its stringified non-string id"
)
@pytest.mark.asyncio
async def test_names_foreign_header_by_stringified_id(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    id_ = SessionId("numeric-id")
    path = generation_log_path(root, "/work", id_, 42, "none")
    write_file(path, f"{dumps({'type': 'session', 'version': 42, 'id': 123})}\n")
    with pytest.raises(SessionFormatUnsupportedError) as caught:
        await persistence.open(id_, "read")
    assert 'session "123" uses log format v42' in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > points a future-generation refusal at the selected raw log path"
)
@pytest.mark.asyncio
async def test_future_refusal_points_at_raw_log(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    m = meta("newer-format", "/work")
    path = generation_log_path(root, m["cwd"], m["id"], 7, "none")
    write_file(path, f"{dumps({**to_header_line(m), 'version': 7})}\n")
    with pytest.raises(SessionFormatUnsupportedError) as caught:
        await persistence.open(m["id"], "read")
    assert f"(raw log: {path})" in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > serves a read open through the full log read when the header-only "
    "read races a writer"
)
@pytest.mark.asyncio
async def test_read_open_survives_header_only_race(
    persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = meta("stat-race-open", "/work")
    await write_log(persistence, m, one_turn_log())

    async def no_stat(_id: object) -> None:
        return None

    monkeypatch.setattr(persistence, "stat", no_stat)
    handle = await persistence.open(m["id"], "read")
    try:
        assert handle.header["id"] == m["id"]
        assert handle.header["cwd"] == "/work"
        read = await handle.read()
        assert read.event_state == "shared-frozen"
        assert list(read.events) == one_turn_log()
        assert all(is_frozen(event) and is_frozen(event["data"]) for event in read.events)
        reread = await handle.read()
        assert reread.events is not read.events
        assert reread.events[0] is read.events[0]
        assert (await handle.read(len(read.events))).event_state == "shared-frozen"
    finally:
        await handle.close()


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > rejects a stored v0 log containing a legacy request/header-delta event"
)
@pytest.mark.asyncio
async def test_rejects_legacy_header_delta(root: str, persistence: JsonlSessionPersistence) -> None:
    m = meta("legacy-header-delta", "/legacy")
    rows = [
        to_header_line(m),
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "request/header-delta",
            "seq": 1,
            "time": 2,
            "data": {"config": {"model": "legacy"}},
        },
        {
            "type": "turn/end",
            "seq": 2,
            "time": 3,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    write_file(raw_log_path(root, m["cwd"], m["id"]), "\n".join(dumps(r) for r in rows) + "\n")
    with pytest.raises(
        Exception,
        match=r'contains event type "request/header-delta" \(seq 1\) unknown to this harness',
    ):
        await read_all(persistence, m["id"])


@pytest.mark.upstream(
    f"{SPEC}::{REFUSALS} > rejects a stored full header carrying the legacy fallback reason"
)
@pytest.mark.asyncio
async def test_rejects_legacy_fallback_reason(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    m = meta("legacy-header-fallback", "/legacy")
    rows = [
        to_header_line(m),
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "request/header",
            "seq": 1,
            "time": 2,
            "data": {
                "header": {"config": {"provider": "mock", "model": "legacy"}},
                "reason": "fallback",
            },
        },
    ]
    write_file(raw_log_path(root, m["cwd"], m["id"]), "\n".join(dumps(r) for r in rows) + "\n")
    with pytest.raises(Exception, match='unsupported legacy reason "fallback"'):
        await read_all(persistence, m["id"])


# --- immutable format generations --------------------------------------------------------


@pytest.mark.upstream(f"{SPEC}::{GENERATIONS} > shares deeply frozen opaque JSON from format v%s")
@pytest.mark.parametrize("version", [SESSION_FORMAT_VERSION])  # the v3 row needs migration
@pytest.mark.asyncio
async def test_shares_deeply_frozen_opaque_json(
    root: str, persistence: JsonlSessionPersistence, version: int
) -> None:
    raw = (
        '{"nested":[[{"values":[null,true,7,"text",[]],'
        '"__proto__":{"leaf":1},"constructor":{"leaf":2}}]]}'
    )
    data = json.loads(raw)
    header = meta(f"frozen-json-v{version}", "/work")
    path = generation_log_path(root, header["cwd"], header["id"], version, "none")
    rows = [
        {**to_header_line(header), "version": version},
        {"type": "external/frozen-json", "seq": 0, "time": 1, "ignorable": True, "data": data},
    ]
    write_file(path, "".join(dumps(row) + "\n" for row in rows))

    handle = await persistence.open(header["id"], "read")
    try:
        read = await handle.read()
        assert read.event_state == "shared-frozen"
        assert len(read.events) == 1
        event = read.events[0]
        actual = event["data"]
        nested = actual["nested"][0][0]
        assert actual == data
        assert "__proto__" in nested
        assert "constructor" in nested
        assert all(
            is_frozen(value)
            for value in [
                event,
                actual,
                actual["nested"],
                actual["nested"][0],
                nested,
                nested["values"],
                nested["values"][4],
                nested["__proto__"],
                nested["constructor"],
            ]
        )
        with pytest.raises(TypeError):
            nested["__proto__"]["leaf"] = 9
        with pytest.raises(TypeError):
            nested["constructor"]["leaf"] = 9
        with pytest.raises(TypeError):
            nested["values"][4].append("changed")

        reread = await handle.read()
        assert reread.events is not read.events
        assert reread.events[0] is event
        assert reread.events[0]["data"] is actual
        assert actual == data
    finally:
        await handle.close()


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > resolves absent, current, and historical current-generation paths"
)
@pytest.mark.asyncio
async def test_resolves_current_generation_paths(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    assert await persistence.resolve_current_log(SessionId("missing-generation")) is None

    current = meta("resolved-current", "/work")
    current_handle = await persistence.create(current)
    await current_handle.flush()
    await current_handle.close()
    assert await persistence.resolve_current_log(current["id"]) == raw_log_path(
        root, current["cwd"], current["id"]
    )

    # Python difference: upstream resolves a historical (v0) generation to `undefined`
    # because it migrates on open; Python ships no v0-v3 migration and refuses it.
    historical = meta("resolved-historical", "/work")
    write_file(
        historical_log_path(root, historical["cwd"], historical["id"]),
        f"{dumps(released_v0_header(historical))}\n",
    )
    with pytest.raises(SessionFormatUnsupportedError):
        await persistence.resolve_current_log(historical["id"])

    future = meta("resolved-future", "/work")
    future_path = generation_log_path(
        root, future["cwd"], future["id"], SESSION_FORMAT_VERSION + 1, "none"
    )
    write_file(
        future_path, f"{dumps({**to_header_line(future), 'version': SESSION_FORMAT_VERSION + 1})}\n"
    )
    with pytest.raises(SessionFormatUnsupportedError):
        await persistence.resolve_current_log(future["id"])


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > treats a historical generation as an existing id at create"
)
@pytest.mark.asyncio
async def test_historical_generation_is_existing_id(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    header = meta("released-v0-collision", "/work")
    write_file(
        historical_log_path(root, header["cwd"], header["id"]),
        f"{dumps(released_v0_header(header))}\n",
    )
    with pytest.raises(SessionAlreadyExistsError):
        await persistence.create(header)


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > materializes a seeded current header at its exact inherited cut"
)
@pytest.mark.asyncio
async def test_materializes_seeded_header_at_cut(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    header = {
        **meta("seeded-current", "/work"),
        "parentSession": SessionId("seed-parent"),
        "isSeeded": True,
        "origin": "subagent",
    }
    handle = await persistence.create(header, inherited_event_count=SessionLogOffset(0))
    assert handle.inherited_event_count == SessionLogOffset(0)
    await handle.append(
        [{"type": "session/end-seed", "seq": 0, "time": 1, "data": {"inherited": True}}]
    )
    await handle.flush()
    await handle.close()

    text = read_text(raw_log_path(root, header["cwd"], header["id"]))
    rows = [json.loads(line) for line in text.strip().split("\n")]
    assert rows[0]["version"] == SESSION_FORMAT_VERSION
    assert rows[0]["isSeeded"] is True
    assert rows[1]["type"] == "session/end-seed"
    assert rows[1]["seq"] == 0
    assert rows[1]["data"] == {"inherited": True}
    snapshot = await persistence.stat(header["id"])
    assert snapshot is not None
    assert snapshot.header["parentSession"] == header["parentSession"]
    assert snapshot.header["origin"] == "subagent"


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > selects a retained future generation above readable historical bytes"
)
@pytest.mark.asyncio
async def test_selects_future_generation_above_historical(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    header = meta("future-wins", "/work")
    source_path = historical_log_path(root, header["cwd"], header["id"])
    future_path = generation_log_path(
        root, header["cwd"], header["id"], SESSION_FORMAT_VERSION + 1, "none"
    )
    source = f"{dumps(released_v0_header(header))}\n"
    write_file(source_path, source)
    future = {"type": "session", "version": SESSION_FORMAT_VERSION + 1, "id": header["id"]}
    write_file(future_path, f"{dumps(future)}\n")

    with pytest.raises(SessionFormatUnsupportedError):
        await persistence.open(header["id"], "read")
    with pytest.raises(SessionFormatUnsupportedError):
        await persistence.stat(header["id"])
    assert list(await persistence.list()) == []
    assert read_text(source_path) == source
    assert not exists(raw_log_path(root, header["cwd"], header["id"]))


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > refuses a selected generation whose filename and header versions "
    "disagree"
)
@pytest.mark.asyncio
async def test_refuses_filename_header_version_mismatch(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    header = meta("generation-header-mismatch", "/work")
    path = generation_log_path(root, header["cwd"], header["id"], 3, "none")
    write_file(path, f"{dumps({**released_v0_header(header), 'version': 2})}\n")
    with pytest.raises(Exception, match=r"filename identifies v3.*header identifies v2"):
        await persistence.stat(header["id"])


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > selects the highest opposite-encoding generation for its refusal"
)
@pytest.mark.asyncio
async def test_selects_highest_opposite_generation(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    header = meta("opposite-generations", "/work")
    write_file(generation_log_path(root, header["cwd"], header["id"], 2, "zstd"), "older")
    highest = generation_log_path(root, header["cwd"], header["id"], 4, "zstd")
    write_file(highest, "newer")
    with pytest.raises(ValueError) as caught:
        await persistence.list()
    assert dumps(highest) in str(caught.value)


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > propagates a non-ENOENT opposite-generation scan failure during "
    "materialization"
)
@pytest.mark.asyncio
async def test_propagates_opposite_generation_scan_failure(
    root: str, persistence: JsonlSessionPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = meta("opposite-generation-scan-failure", "/work")
    handle = await persistence.create(header)
    failure = PermissionError(13, "opposite-generation scan denied")
    target = session_dir(root, header["cwd"], header["id"])
    real_listdir: Callable[..., list[str]] = os.listdir

    def listdir(path: Any = ".") -> list[str]:
        if os.fspath(path) == target:
            raise failure
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", listdir)
    try:
        with pytest.raises(PermissionError) as caught:
            await handle.flush()
        assert caught.value is failure
    finally:
        await handle.close()


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > surfaces a file occupying the targeted Session-directory path"
)
@pytest.mark.asyncio
async def test_surfaces_file_occupying_session_directory(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    header = meta("blocked-generation-directory", "/work")
    write_file(session_dir(root, header["cwd"], header["id"]), "not a directory")
    with pytest.raises(NotADirectoryError):
        await persistence.open(header["id"], "read")


@pytest.mark.upstream(
    f"{SPEC}::{GENERATIONS} > attaches the current path when a direct handle read finds a "
    "foreign header"
)
@pytest.mark.asyncio
async def test_direct_read_attaches_current_path(
    root: str, persistence: JsonlSessionPersistence
) -> None:
    header = meta("foreign-direct-read", "/work")
    path = raw_log_path(root, header["cwd"], header["id"])
    write_file(
        path, f"{dumps({**to_header_line(header), 'version': SESSION_FORMAT_VERSION + 1})}\n"
    )
    with pytest.raises(SessionFormatUnsupportedError) as caught:
        await persistence.read_stored_log(path, header["id"])
    assert caught.type is SessionFormatUnsupportedError
    assert f"(raw log: {path})" in str(caught.value)
