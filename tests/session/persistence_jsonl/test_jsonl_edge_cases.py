"""Ported from packages/session/session-persistence-jsonl/tests/jsonl.spec.ts.

Covers the "edge cases" and "listing concurrency" describes. Upstream's ``vi.mock`` of
``readdir``/``stat`` becomes a patch of the backend's own directory-listing and stat
helpers, which run in worker threads.
"""

from __future__ import annotations

import asyncio
import errno
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydantic
import pytest
import pytest_asyncio
from deepseek_cordis import Context

import deskagents.session.persistence_jsonl as jsonl_module
from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionStore,
)
from deskagents.session import persistence_jsonl as jsonl
from deskagents.session.persistence import (
    SessionAlreadyExistsError,
    SessionHandleClosedError,
    SessionPersistence,
)
from deskagents.session.persistence_jsonl import JsonlConfig, JsonlSessionPersistence
from deskagents.session.persistence_jsonl.format import (
    encode_segment,
    generation_log_filename,
    log_path,
    project_dir,
    scan_log,
    session_dir,
    to_header_line,
)
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/jsonl.spec.ts"
EDGE = f"{SPEC}::JsonlSessionPersistence: edge cases"
LISTING = f"{SPEC}::JsonlSessionPersistence listing concurrency"


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


def raw_log_path(root: Path, cwd: str | None, id_: str) -> str:
    return log_path(str(root), cwd, id_, "none")


def log_text(header: SessionHeader, events: Sequence[SessionEvent]) -> str:
    lines = [stringify_json(to_header_line(header))] + [stringify_json(e) for e in events]
    return "\n".join(lines) + "\n"


def write_text(*parts: str | Path) -> None:
    """Write the last argument as UTF-8 text to the path joined from the others."""
    *path, content = parts
    Path(*path).write_text(str(content), "utf-8")


def is_file(path: str) -> bool:
    return os.path.isfile(path)


def rewrite_header(path: str, update: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite only a stored header while preserving every event byte below it."""
    lines = Path(path).read_text("utf-8").split("\n")
    header: Any = dict(parse_json(lines[0]))  # type: ignore[arg-type]
    update(header)
    lines[0] = stringify_json(header)
    Path(path).write_text("\n".join(lines), "utf-8")


async def mount(root: Path | str, **config: Any) -> tuple[Context, SessionPersistence]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    await ctx.plugin(jsonl.apply, {"root": str(root), "compression": "none", **config})
    service = ctx.get("sessionPersistence")
    assert isinstance(service, SessionPersistence)
    return ctx, service


async def write_log(
    persistence: SessionPersistence, header: SessionHeader, events: Sequence[SessionEvent]
) -> None:
    handle = await persistence.create(header)
    try:
        await handle.append(events)
    finally:
        await handle.close()


async def read_all(
    persistence: SessionPersistence, id_: str
) -> tuple[SessionHeader, list[SessionEvent]]:
    handle = await persistence.open(id_, "read")
    try:
        return handle.header, list((await handle.read()).events)
    finally:
        await handle.close()


async def listed_ids(persistence: SessionPersistence) -> list[str]:
    return sorted(snapshot.header["id"] for snapshot in await persistence.list())


@dataclass
class Backend:
    root: Path
    ctx: Context
    persistence: SessionPersistence


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncIterator[Backend]:
    root = tmp_path / "root"
    root.mkdir()
    ctx, persistence = await mount(root)
    state = Backend(root, ctx, persistence)
    yield state
    await state.ctx.fiber.dispose()


# --- edge cases --------------------------------------------------------------------------


@pytest.mark.upstream(
    f"{EDGE} > create snapshots its meta: mutating the caller object after the call is ignored"
)
@pytest.mark.asyncio
async def test_create_snapshots_its_meta(backend: Backend) -> None:
    m = meta("create-snap", "/orig")
    # Python difference: a coroutine runs nothing until awaited, so the caller object
    # is mutated once create returns rather than between the call and its settlement.
    handle = await backend.persistence.create(m)
    m["cwd"] = "/mutated"
    await handle.append(one_turn_log())
    await handle.close()
    assert is_file(raw_log_path(backend.root, "/orig", "create-snap"))
    with pytest.raises(OSError):
        os.stat(raw_log_path(backend.root, "/mutated", "create-snap"))


@pytest.mark.upstream(
    f"{EDGE} > create rejects non-JSON metadata and a fractional creation timestamp without "
    "reserving the id"
)
@pytest.mark.asyncio
async def test_create_rejects_non_json_metadata_without_reserving_the_id(
    backend: Backend,
) -> None:
    with pytest.raises(Exception, match="session metadata must be losslessly JSON-serializable"):
        await backend.persistence.create({**meta("bad-meta"), "extra": object()})
    with pytest.raises(
        Exception, match="session metadata createdAt must be a non-negative safe integer"
    ):
        await backend.persistence.create({**meta("fractional-created-at"), "createdAt": 1.5})

    valid = meta("fractional-created-at")
    await write_log(backend.persistence, valid, one_turn_log())
    header, _ = await read_all(backend.persistence, valid["id"])
    assert header["createdAt"] == valid["createdAt"]


@pytest.mark.upstream(f"{EDGE} > list discovers sessions across multiple project directories")
@pytest.mark.asyncio
async def test_list_discovers_sessions_across_multiple_project_directories(
    backend: Backend,
) -> None:
    await write_log(backend.persistence, meta("p1", "/projA"), one_turn_log())
    await write_log(backend.persistence, meta("p2", "/projB"), one_turn_log())
    await write_log(backend.persistence, meta("p3"), one_turn_log())
    assert await listed_ids(backend.persistence) == ["p1", "p2", "p3"]


@pytest.mark.upstream(
    f"{EDGE} > groups sessions whose cwd paths normalize to the same project directory"
)
@pytest.mark.asyncio
async def test_groups_sessions_whose_cwd_paths_normalize_to_the_same_project_directory(
    backend: Backend,
) -> None:
    first = meta("normalized-first", "/a/b-c")
    second = meta("normalized-second", "/a-b/c")
    await write_log(backend.persistence, first, one_turn_log())
    await write_log(backend.persistence, second, one_turn_log())

    root = str(backend.root)
    assert project_dir(root, first["cwd"]) == project_dir(root, second["cwd"])
    entries = os.listdir(project_dir(root, first["cwd"]))
    assert encode_segment(first["id"]) in entries
    assert encode_segment(second["id"]) in entries
    assert await listed_ids(backend.persistence) == sorted([first["id"], second["id"]])


@pytest.mark.upstream(f"{EDGE} > list on an empty root returns nothing")
@pytest.mark.asyncio
async def test_list_on_an_empty_root_returns_nothing(backend: Backend) -> None:
    assert list(await backend.persistence.list()) == []


@pytest.mark.upstream(f"{EDGE} > keeps the transcript in an extensible session-owned directory")
@pytest.mark.asyncio
async def test_keeps_the_transcript_in_an_extensible_session_owned_directory(
    backend: Backend,
) -> None:
    root = str(backend.root)
    m = meta("owned-directory", "/project")
    await write_log(backend.persistence, m, one_turn_log())
    directory = session_dir(root, m["cwd"], m["id"])
    write_text(directory, "metadata.json", "{}\n")
    write_text(project_dir(root, m["cwd"]), "README", "project metadata\n")
    os.makedirs(os.path.join(project_dir(root, m["cwd"]), "reserved-session"), exist_ok=True)

    entries = os.listdir(directory)
    assert "metadata.json" in entries
    assert generation_log_filename(SESSION_FORMAT_VERSION, "none") in entries
    assert m["id"] in await listed_ids(backend.persistence)
    _, events = await read_all(backend.persistence, m["id"])
    assert events == one_turn_log()


@pytest.mark.upstream(
    f"{EDGE} > rejects the obsolete flat-file layout instead of ignoring stored sessions"
)
@pytest.mark.asyncio
async def test_rejects_the_obsolete_flat_file_layout(backend: Backend) -> None:
    m = meta("legacy-flat", "/legacy")
    project = project_dir(str(backend.root), m["cwd"])
    os.makedirs(project)
    write_text(project, f"{encode_segment(m['id'])}.jsonl", log_text(m, one_turn_log()))

    with pytest.raises(ValueError, match="unsupported flat-file layout"):
        await backend.persistence.open(m["id"], "read")
    with pytest.raises(ValueError, match="unsupported flat-file layout"):
        await backend.persistence.list()


@pytest.mark.upstream(
    f"{EDGE} > rejects a compressed obsolete flat-file artifact during targeted lookup"
)
@pytest.mark.asyncio
async def test_rejects_a_compressed_obsolete_flat_file_artifact(backend: Backend) -> None:
    m = meta("legacy-compressed-flat", "/legacy")
    project = project_dir(str(backend.root), m["cwd"])
    assert list(await backend.persistence.list()) == []
    os.makedirs(project)
    write_text(project, f"{encode_segment(m['id'])}.jsonl.zstd", "legacy")

    with pytest.raises(ValueError, match="unsupported flat-file layout"):
        await backend.persistence.open(m["id"], "read")


@pytest.mark.upstream(f"{EDGE} > list skips empty and non-header session logs (metadata-only read)")
@pytest.mark.asyncio
async def test_list_skips_empty_and_non_header_session_logs(backend: Backend) -> None:
    root = str(backend.root)
    await write_log(backend.persistence, meta("real", "/p"), one_turn_log())
    for id_, content in [
        ("empty", ""),
        ("notheader", '{"type":"turn/start"}\n'),
        ("badjson", "not json at all\n"),
    ]:
        os.makedirs(session_dir(root, None, id_), exist_ok=True)
        write_text(raw_log_path(backend.root, None, id_), content)

    assert await listed_ids(backend.persistence) == ["real"]


@pytest.mark.upstream(f"{EDGE} > list reads a header line longer than the 8KB read chunk")
@pytest.mark.asyncio
async def test_list_reads_a_header_line_longer_than_the_8kb_read_chunk(
    backend: Backend,
) -> None:
    os.makedirs(session_dir(str(backend.root), None, "big"))
    big_header = stringify_json(
        {
            "type": "session",
            "version": SESSION_FORMAT_VERSION,
            "id": "big",
            "createdAt": 1,
            "isSeeded": False,
            "delegationDepth": 0,
            "agentPreset": "x" * 9000,
        }
    )
    write_text(raw_log_path(backend.root, None, "big"), big_header + "\n")
    assert "big" in await listed_ids(backend.persistence)


@pytest.mark.upstream(f"{EDGE} > rejects the retired %s header field")
@pytest.mark.parametrize("field_name", ["sandboxMode", "approvalPolicy"])
def test_rejects_the_retired_header_field(field_name: str) -> None:
    line = {**to_header_line(meta("retired-policy-header")), field_name: "read-only"}
    with pytest.raises(ValueError, match="retired policy baseline fields"):
        scan_log(f"{stringify_json(line)}\n".encode())


@pytest.mark.upstream(
    f"{EDGE} > list rejects a header whose cwd does not identify its physical log"
)
@pytest.mark.asyncio
async def test_list_rejects_a_header_whose_cwd_does_not_identify_its_physical_log(
    backend: Backend,
) -> None:
    m = meta("misplaced", "/stored")
    await write_log(backend.persistence, m, one_turn_log())

    def move(header: dict[str, Any]) -> None:
        header["cwd"] = "/elsewhere"

    rewrite_header(raw_log_path(backend.root, m["cwd"], m["id"]), move)
    with pytest.raises(ValueError, match="and cwd identify"):
        await backend.persistence.list()


@pytest.mark.upstream(
    f"{EDGE} > accepts an alternate project path only when it identifies the same physical log"
)
@pytest.mark.asyncio
async def test_accepts_an_alternate_project_path_only_when_it_identifies_the_same_log(
    backend: Backend,
) -> None:
    root = str(backend.root)
    m = meta("physical-alias", "/stored")
    await write_log(backend.persistence, m, one_turn_log())
    path = raw_log_path(backend.root, m["cwd"], m["id"])
    alias_cwd = "/alias"
    os.symlink(project_dir(root, m["cwd"]), project_dir(root, alias_cwd), target_is_directory=True)

    def alias(header: dict[str, Any]) -> None:
        header["cwd"] = alias_cwd

    rewrite_header(path, alias)

    header, _ = await read_all(backend.persistence, m["id"])
    assert header["cwd"] == alias_cwd
    assert m["id"] in [s.header["id"] for s in await backend.persistence.list()]


@pytest.mark.upstream(f"{EDGE} > list rejects a session header whose id cannot name a storage path")
@pytest.mark.asyncio
async def test_list_rejects_a_session_header_whose_id_cannot_name_a_storage_path(
    backend: Backend,
) -> None:
    directory = os.path.join(project_dir(str(backend.root), None), "invalid-id")
    os.makedirs(directory)
    header = {
        "type": "session",
        "version": SESSION_FORMAT_VERSION,
        "id": "",
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    write_text(
        directory,
        generation_log_filename(SESSION_FORMAT_VERSION, "none"),
        stringify_json(header) + "\n",
    )
    with pytest.raises(ValueError, match="header id cannot name a storage path"):
        await backend.persistence.list()


@pytest.mark.upstream(
    f"{EDGE} > open and list reject one id materialized in multiple project directories"
)
@pytest.mark.asyncio
async def test_open_and_list_reject_one_id_materialized_in_multiple_project_directories(
    backend: Backend,
) -> None:
    id_ = "duplicate"
    for cwd in ["/a", "/b"]:
        os.makedirs(session_dir(str(backend.root), cwd, id_))
        write_text(raw_log_path(backend.root, cwd, id_), log_text(meta(id_, cwd), one_turn_log()))

    with pytest.raises(ValueError, match="appears in multiple project directories"):
        await backend.persistence.open(id_, "read")
    with pytest.raises(ValueError, match="appears in multiple project directories"):
        await backend.persistence.list()


@pytest.mark.upstream(
    f"{EDGE} > create rejects an id already on disk under a different project directory"
)
@pytest.mark.asyncio
async def test_create_rejects_an_id_already_on_disk_under_a_different_project_directory(
    backend: Backend,
) -> None:
    await write_log(backend.persistence, meta("dup-id", "/projA"), one_turn_log())
    ctx2, persistence2 = await mount(backend.root)
    try:
        with pytest.raises(SessionAlreadyExistsError, match="already exists"):
            await persistence2.create(meta("dup-id", "/projB"))
    finally:
        await ctx2.fiber.dispose()


@pytest.mark.upstream(f"{EDGE} > list returns nothing when the root directory does not exist")
@pytest.mark.asyncio
async def test_list_returns_nothing_when_the_root_directory_does_not_exist(
    backend: Backend,
) -> None:
    ctx2, persistence2 = await mount(backend.root / "does-not-exist-yet")
    try:
        assert list(await persistence2.list()) == []
    finally:
        await ctx2.fiber.dispose()


@pytest.mark.upstream(f"{EDGE} > plugin load rejects an existing root that is not a directory")
@pytest.mark.asyncio
async def test_plugin_load_rejects_an_existing_root_that_is_not_a_directory(
    backend: Backend,
) -> None:
    file_path = backend.root / "not-a-dir"
    file_path.write_text("x")
    ctx2 = Context()
    try:
        with pytest.raises(NotADirectoryError) as caught:
            await ctx2.plugin(jsonl.apply, {"root": str(file_path), "compression": "none"})
        assert caught.value.errno == errno.ENOTDIR
    finally:
        await ctx2.fiber.dispose()


@pytest.mark.upstream(f"{EDGE} > list surfaces a root that becomes unusable after plugin load")
@pytest.mark.asyncio
async def test_list_surfaces_a_root_that_becomes_unusable_after_plugin_load(
    backend: Backend,
) -> None:
    os.rmdir(backend.root)
    backend.root.write_text("not a directory")
    with pytest.raises(NotADirectoryError) as caught:
        await backend.persistence.list()
    assert caught.value.errno == errno.ENOTDIR


@pytest.mark.upstream(f"{EDGE} > per-id lookup surfaces non-ENOENT storage errors")
@pytest.mark.asyncio
async def test_per_id_lookup_surfaces_non_enoent_storage_errors(backend: Backend) -> None:
    blocker = backend.root / "not-a-directory"
    blocker.write_text("x")
    service = backend.persistence
    assert isinstance(service, JsonlSessionPersistence)
    # Upstream probes the private exists() on `blocker/child.jsonl`; Python's probe of that
    # path is the legacy flat-artifact check for id "child" in project `blocker`.
    with pytest.raises(NotADirectoryError):
        service._reject_legacy_flat_artifact(str(blocker), SessionId("child"))


@pytest.mark.upstream(
    f"{EDGE} > a project-directory storage fault surfaces at the first materializing write"
)
@pytest.mark.asyncio
async def test_a_project_directory_storage_fault_surfaces_at_the_first_materializing_write(
    backend: Backend,
) -> None:
    cwd = "/x"
    write_text(project_dir(str(backend.root), cwd), "x")
    handle = await backend.persistence.create(meta("exists-fault", cwd))
    with pytest.raises(OSError) as caught:
        await handle.append([{"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}])
    assert caught.value.errno in (errno.EEXIST, errno.ENOTDIR)
    await handle.close()


@pytest.mark.upstream(
    f"{EDGE} > backend teardown closes handles left open and fails later operations loudly"
)
@pytest.mark.asyncio
async def test_backend_teardown_closes_handles_left_open(backend: Backend) -> None:
    handle = await backend.persistence.create(meta("teardown"))
    await handle.append(one_turn_log())
    await backend.ctx.fiber.dispose()
    with pytest.raises(SessionHandleClosedError, match="on a closed handle"):
        await handle.append([{"type": "turn/start", "seq": 6, "time": 9, "data": {"turn": 2}}])
    # Reload the backend so the fixture's dispose stays valid.
    backend.ctx, backend.persistence = await mount(backend.root)


@pytest.mark.upstream(
    f"{EDGE} > accepts well-formed JSON values (null, booleans, nested arrays/objects)"
)
@pytest.mark.asyncio
async def test_accepts_well_formed_json_values(backend: Backend) -> None:
    m = meta("json-ok")
    events: list[dict[str, Any]] = [
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": {
                "id": "json-ok",
                "role": "user",
                "content": [{"type": "text", "text": "x"}],
                "source": {"kind": "user"},
                "extra": {"a": None, "b": True, "c": [1, 2, {"d": "nested"}]},
            },
            "surfaceOp": "append",
        }
    ]
    await write_log(backend.persistence, m, events)
    _, loaded = await read_all(backend.persistence, m["id"])
    assert loaded == events


# --- listing concurrency -----------------------------------------------------------------


@dataclass
class ListingProbe:
    """While active, every directory read and stat waits, counts overlap, and may fail."""

    active: bool = False
    delay_s: float = 0.0
    in_flight: int = 0
    max_in_flight: int = 0
    failures: dict[str, Exception] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, path: str, read: Callable[[], Any]) -> Any:
        if not self.active:
            return read()
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay_s)
            failure = self.failures.get(path)
            if failure is not None:
                raise failure
            return read()
        finally:
            with self.lock:
                self.in_flight -= 1


@dataclass
class Listing:
    root: Path
    persistence: SessionPersistence
    probe: ListingProbe


@pytest_asyncio.fixture
async def listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Listing]:
    probe = ListingProbe()
    cls = JsonlSessionPersistence
    list_projects = cls._list_project_dirs
    list_sessions = cls._list_session_dirs
    resolve = cls._resolve_generation_in_directory
    opposite = cls._find_opposite_generation
    stat_identity = jsonl_module._stat_identity

    def patched_list_projects(self: JsonlSessionPersistence) -> list[str]:
        return probe.run(self.root, lambda: list_projects(self))  # type: ignore[no-any-return]

    def patched_list_sessions(self: JsonlSessionPersistence, project: str) -> list[str]:
        return probe.run(project, lambda: list_sessions(self, project))  # type: ignore[no-any-return]

    def patched_resolve(self: JsonlSessionPersistence, directory: str) -> Any:
        return probe.run(directory, lambda: resolve(self, directory))

    def patched_opposite(self: JsonlSessionPersistence, directory: str) -> Any:
        return probe.run(directory, lambda: opposite(self, directory))

    def patched_stat(path: str) -> Any:
        return probe.run(path, lambda: stat_identity(path))

    monkeypatch.setattr(cls, "_list_project_dirs", patched_list_projects)
    monkeypatch.setattr(cls, "_list_session_dirs", patched_list_sessions)
    monkeypatch.setattr(cls, "_resolve_generation_in_directory", patched_resolve)
    monkeypatch.setattr(cls, "_find_opposite_generation", patched_opposite)
    monkeypatch.setattr(jsonl_module, "_stat_identity", patched_stat)

    root = tmp_path / "listing"
    root.mkdir()
    ctx, persistence = await mount(root, listConcurrency=2)
    yield Listing(root, persistence, probe)
    probe.active = False
    await ctx.fiber.dispose()


async def seed(persistence: SessionPersistence, cwd: str, ids: Sequence[str]) -> None:
    for id_ in ids:
        await write_log(persistence, meta(id_, cwd), one_turn_log())


def directories_of(directory: Path | str) -> list[str]:
    """Directory entries in the order the backend's listing reads them."""
    return [os.path.join(directory, e.name) for e in os.scandir(directory) if e.is_dir()]


@pytest.mark.upstream(
    f"{LISTING} > validates listConcurrency as a positive integer that defaults to 3"
)
def test_validates_list_concurrency_as_a_positive_integer_that_defaults_to_3(
    tmp_path: Path,
) -> None:
    root = str(tmp_path)
    assert JsonlConfig.model_validate({"root": root}).list_concurrency == 3
    assert JsonlConfig.model_validate({"root": root, "listConcurrency": 1}).list_concurrency == 1
    with pytest.raises(pydantic.ValidationError):
        JsonlConfig.model_validate({"root": root, "listConcurrency": 0})
    with pytest.raises(pydantic.ValidationError):
        JsonlConfig.model_validate({"root": root, "listConcurrency": 1.5})


@pytest.mark.upstream(
    f"{LISTING} > bounds directory and stat reads across concurrent listings of one backend"
)
@pytest.mark.asyncio
async def test_bounds_directory_and_stat_reads_across_concurrent_listings(
    listing: Listing,
) -> None:
    persistence = listing.persistence
    await seed(persistence, "/work-a", ["bound-a1", "bound-a2", "bound-a3"])
    await seed(persistence, "/work-b", ["bound-b1", "bound-b2", "bound-b3"])
    expected = ["bound-a1", "bound-a2", "bound-a3", "bound-b1", "bound-b2", "bound-b3"]
    assert await listed_ids(persistence) == expected

    listing.probe.delay_s = 0.05
    listing.probe.active = True
    # Python addition: start the second listing once the first holds both limiter slots,
    # so an unbounded read in the second one overlaps them regardless of thread timing.
    first = asyncio.ensure_future(persistence.list())
    for _ in range(1000):
        if listing.probe.in_flight >= 2:
            break
        await asyncio.sleep(0.001)
    second = asyncio.ensure_future(persistence.list())
    listings = await asyncio.gather(first, second)
    listing.probe.active = False

    assert listing.probe.max_in_flight == 2
    for result in listings:
        assert sorted(s.header["id"] for s in result) == expected


@pytest.mark.upstream(f"{LISTING} > reports the failure a sequential traversal would reach first")
@pytest.mark.asyncio
async def test_reports_the_failure_a_sequential_traversal_would_reach_first(
    listing: Listing,
) -> None:
    persistence = listing.persistence
    await seed(persistence, "/work-a", ["order-a1", "order-a2"])
    await seed(persistence, "/work-b", ["order-b1", "order-b2"])
    await persistence.list()
    earlier_project, later_project = directories_of(listing.root)
    earlier_session, later_session = directories_of(earlier_project)
    project_failure = PermissionError(errno.EACCES, "later project denied")
    later_session_failure = PermissionError(errno.EACCES, "later session denied")
    earlier_session_failure = PermissionError(errno.EACCES, "earlier session denied")
    listing.probe.active = True

    listing.probe.failures[later_project] = project_failure
    with pytest.raises(PermissionError) as first:
        await persistence.list()
    assert first.value is project_failure
    listing.probe.failures[later_session] = later_session_failure
    with pytest.raises(PermissionError) as second:
        await persistence.list()
    assert second.value is later_session_failure
    listing.probe.failures[earlier_session] = earlier_session_failure
    with pytest.raises(PermissionError) as third:
        await persistence.list()
    assert third.value is earlier_session_failure
