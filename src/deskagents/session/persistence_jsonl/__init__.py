"""JSONL session persistence: one immutable generation file per Session format version.

Upstream: packages/session/session-persistence-jsonl
(``@deepseek-ai/dsh-session-persistence-jsonl``) at ce7073d.

Layout: ``<root>/<project key>/<encoded session id>/session.v4.jsonl`` with a
``session.lock`` single-writer lease beside it. The first record is the header line;
each event is one record. Materialization writes a synced temporary file and publishes it
with a hard link (POSIX, then a directory fsync) or a no-overwrite rename (Windows), so a
generation is never partially visible.

Python differences:

- Only format v4 is read and written. A directory whose newest generation is v0–v3 is
  reported as unsupported (upstream migrates it to a new v4 generation); ``stat`` raises
  ``SessionFormatUnsupportedError`` for it and ``list`` skips it, as upstream skips
  unreadable artifacts.
- The service registers as ``sessionPersistence`` (deepseek-cordis registers a service
  under its class ``name``); upstream's diagnostic name is :attr:`backend_name`.
- Blocking file I/O runs in worker threads; there is no ``AbortSignal`` (cancel the task).
- ``AggregateError`` becomes :class:`ExceptionGroup`; ``ctx.logger`` becomes
  :mod:`logging`.
"""

from __future__ import annotations

import asyncio
import builtins
import logging
import os
import secrets
import sys
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from deepseek_cordis import Context, Fiber
from pydantic import BaseModel, ConfigDict, Field

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
    session_log_offset,
)
from deskagents.session.format_catalog import session_format_catalog
from deskagents.session.persistence import (
    SessionAccess,
    SessionAlreadyExistsError,
    SessionFormatUnsupportedError,
    SessionHandle,
    SessionHandleReadResult,
    SessionLocation,
    SessionPersistence,
    SessionPersistenceCorruptionError,
    SessionPersistenceNotFoundError,
    SessionPersistenceRevision,
    SessionPersistenceSnapshot,
    assert_stored_id,
    materialize_create_header,
    session_format_version_refusal,
    validate_stored_events,
)
from deskagents.util.values import parse_json, stringify_json

from .format import (
    JsonlCompression,
    SessionLogScanner,
    assert_no_retired_header_fields,
    encode_segment,
    event_lines,
    generation_log_filename,
    generation_log_path,
    log_path,
    log_suffix,
    parse_generation_log_filename,
    project_dir,
    scan_log,
    session_dir,
    to_header_line,
)
from .generation import JsonlPhysicalIdentity, read_stable_jsonl_file
from .lease import LEASE_FILENAME, SessionWriteLease
from .storage import (
    LIVE_WRITE_BATCH_MAX_DELAY_S,
    JsonlBackendTracker,
    JsonlSessionHandle,
    StorageHandleState,
)
from .zstd import (
    compress_zstd_frame,
    decode_zstd_frames,
    decompress_zstd_frame,
    decompress_zstd_prefix,
    scan_zstd_frames,
)

__all__ = [
    "LEASE_FILENAME",
    "LIVE_WRITE_BATCH_MAX_DELAY_S",
    "NAME",
    "Config",
    "JsonlCompression",
    "JsonlConfig",
    "JsonlSessionPersistence",
    "apply",
    "encode_segment",
    "generation_log_path",
    "log_path",
    "project_dir",
    "session_dir",
]

logger = logging.getLogger(__name__)

NAME = "session-persistence-jsonl"
_COLD_LOG_MEMO_MAX_ENTRIES = 2
_DEFAULT_LIST_CONCURRENCY = 3
T = TypeVar("T")


class JsonlConfig(BaseModel):
    """Plugin configuration (upstream keys ``root``, ``compression``, ``listConcurrency``)."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    root: str
    compression: Literal["zstd", "none"] = "zstd"
    list_concurrency: int = Field(default=_DEFAULT_LIST_CONCURRENCY, ge=1, alias="listConcurrency")


@dataclass(frozen=True)
class _StoredLog(SessionHandleReadResult):
    meta: SessionHeader = None  # type: ignore[assignment]
    torn_truncate_to: int | None = None
    recovered_tail: tuple[SessionEvent, ...] = ()
    inherited_event_count: SessionLogOffset = SessionLogOffset(0)
    revision: SessionPersistenceRevision = SessionPersistenceRevision("")


@dataclass(frozen=True)
class _ResolvedGeneration:
    source_path: str
    source_version: int
    current_path: str


@dataclass(frozen=True)
class _GenerationHeader:
    header: SessionHeader
    format_status: Literal["current", "migration-required"]


def _revision(identity: JsonlPhysicalIdentity) -> SessionPersistenceRevision:
    return SessionPersistenceRevision(identity.token())


def _stat_identity(path: str) -> JsonlPhysicalIdentity:
    return JsonlPhysicalIdentity.of(os.stat(path))


def _fsync_dir(path: str) -> None:
    if sys.platform == "win32":
        return  # Windows namespace operations are write-through
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_synced_temp(final_path: str, content: bytes) -> str:
    tmp = f"{final_path}.{secrets.token_hex(6)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return tmp


def _publish_new(tmp: str, final_path: str) -> None:
    """Publish ``tmp`` at ``final_path`` without ever replacing an existing file."""
    if sys.platform == "win32":
        try:
            os.rename(tmp, final_path)  # refuses an existing target on Windows
        except BaseException:
            _remove(tmp)
            raise
        return
    linked = False
    try:
        os.link(tmp, final_path)
        linked = True
    finally:
        if not linked:
            _remove(tmp)
    _fsync_dir(os.path.dirname(final_path))
    _remove(tmp)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _append_synced(path: str, content: bytes) -> None:
    with open(path, "ab") as handle:
        before = os.fstat(handle.fileno()).st_size
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as error:
            try:
                handle.close()
                with open(path, "r+b") as rollback:
                    rollback.truncate(before)
                    rollback.flush()
                    os.fsync(rollback.fileno())
            except BaseException as rollback_error:
                raise ExceptionGroup(
                    f'failed to roll back append to "{path}"',
                    [e for e in (error, rollback_error) if isinstance(e, Exception)],
                ) from None
            raise


def _truncate_synced(path: str, offset: int) -> None:
    os.truncate(path, offset)
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


def _read_first_line(path: str) -> str | None:
    chunks = bytearray()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8192)
            if not block:
                return None  # EOF with no newline: no complete line
            newline = block.find(0x0A)
            if newline != -1:
                chunks += block[:newline]
                return chunks.decode("utf-8", errors="replace")
            chunks += block


def _assert_zstd_header_frame(plaintext: bytes) -> None:
    if not plaintext or plaintext.find(b"\n") != len(plaintext) - 1:
        raise ValueError(
            "corrupt Zstandard session log: first frame is not exactly one header line"
        )


def _read_first_zstd_line(path: str) -> str | None:
    content = bytearray()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8192)
            if not block:
                return None
            content += block
            scan = scan_zstd_frames(bytes(content), 1)
            if not scan.frames:
                continue
            first = scan.frames[0]
            try:
                plaintext = decompress_zstd_frame(bytes(content[first.start : first.end]))
                _assert_zstd_header_frame(plaintext)
            except Exception as error:
                raise SessionPersistenceCorruptionError(
                    "corrupt Zstandard session log: header frame failed validation: "
                    f"{error} (raw log: {path})"
                ) from error
            return plaintext[:-1].decode("utf-8", errors="replace")


async def _gather_limited(
    limit: asyncio.Semaphore, items: Sequence[Any], task: Callable[[Any], Awaitable[T]]
) -> list[T]:
    async def run(item: Any) -> T:
        async with limit:
            return await task(item)

    outcomes = await asyncio.gather(*(run(item) for item in items), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return outcomes  # type: ignore[return-value]


class JsonlSessionPersistence(SessionPersistence):
    """The JSONL backend of the ``sessionPersistence`` seam."""

    backend_name = NAME
    """Upstream's diagnostic service name."""

    def __init__(self, ctx: Context, config: JsonlConfig, tracker: JsonlBackendTracker) -> None:
        super().__init__(ctx, config)
        if session_format_catalog.current_version != SESSION_FORMAT_VERSION:
            raise RuntimeError(
                f"{NAME}: format catalog v{session_format_catalog.current_version} does not "
                f"match Session v{SESSION_FORMAT_VERSION}"
            )
        self._root = os.path.abspath(config.root)
        self._compression: JsonlCompression = config.compression
        self._limit = asyncio.Semaphore(config.list_concurrency)
        self._tracker = tracker
        self._cold_log_memo: OrderedDict[SessionId, _StoredLog] = OrderedDict()
        self._root_encoding_check: asyncio.Future[None] | None = None
        self._assert_usable_root()

    @property
    def root(self) -> str:
        return self._root

    def _locate(self, meta: SessionHeader) -> SessionLocation:
        return SessionLocation(
            "jsonl", log_path(self._root, meta.get("cwd"), meta["id"], self._compression)
        )

    # --- SessionPersistence -----------------------------------------------------------

    async def create(
        self, header: SessionHeader, *, inherited_event_count: int | None = None
    ) -> SessionHandle:
        snapshot = materialize_create_header(header)
        to_header_line(snapshot, inherited_event_count)
        cut = session_log_offset(inherited_event_count or 0)
        await self._ensure_root_encoding()
        sid = SessionId(snapshot["id"])
        if self._tracker.has_pending(sid) or await self._find_log(sid) is not None:
            raise SessionAlreadyExistsError(sid)
        self._tracker.register_created(snapshot, cut)
        return self._tracker.adopt(
            JsonlSessionHandle(
                self,
                sid,
                snapshot,
                "write",
                StorageHandleState(cursor=0, materialized=False, inherited_event_count=cut),
            )
        )

    async def open(self, session_id: SessionId | str, access: SessionAccess) -> SessionHandle:
        sid = SessionId(session_id)
        await self._ensure_root_encoding()
        pending = self._tracker.pending_of(sid)
        if access == "read":
            if pending is not None:
                return self._tracker.adopt(
                    JsonlSessionHandle(
                        self,
                        sid,
                        pending.header,
                        "read",
                        StorageHandleState(
                            cursor=0,
                            materialized=False,
                            inherited_event_count=pending.inherited_event_count,
                        ),
                    )
                )
            stored = await self._require_stored_log(sid)
            return self._tracker.adopt(
                JsonlSessionHandle(
                    self,
                    sid,
                    stored.meta,
                    "read",
                    StorageHandleState(
                        cursor=0,
                        materialized=True,
                        inherited_event_count=stored.inherited_event_count,
                    ),
                )
            )
        self._tracker.claim_write(sid)
        lease: SessionWriteLease | None = None
        try:
            resolved = await self._find_log(sid)
            if resolved is None:
                raise SessionPersistenceNotFoundError(sid)
            lease = await SessionWriteLease.acquire(os.path.dirname(resolved.current_path), sid)
            stored = await self._require_stored_log(sid)
            return self._tracker.adopt(
                JsonlSessionHandle(
                    self,
                    sid,
                    stored.meta,
                    "write",
                    StorageHandleState(
                        cursor=len(stored.events),
                        materialized=True,
                        torn_truncate_to=stored.torn_truncate_to,
                        recovered_tail=builtins.list(stored.recovered_tail),
                        inherited_event_count=stored.inherited_event_count,
                        primed=stored,
                    ),
                    lease,
                )
            )
        except BaseException as failure:
            release_failure: Exception | None = None
            if lease is not None:
                try:
                    await lease.release()
                except Exception as error:
                    release_failure = error
            self._tracker.release_claim(sid)
            if release_failure is not None and isinstance(failure, Exception):
                raise ExceptionGroup(
                    f'session "{sid}": write open failed and its lock release failed',
                    [failure, release_failure],
                ) from None
            raise

    async def flush(self) -> None:
        await self._tracker.flush_all()

    async def stat(self, session_id: SessionId | str) -> SessionPersistenceSnapshot | None:
        sid = SessionId(session_id)
        await self._ensure_root_encoding()
        pending = self._tracker.pending_of(sid)
        if pending is not None:
            return SessionPersistenceSnapshot(header=pending.header, revision=pending.revision)
        selected = await self._find_log(sid)
        if selected is None:
            return None
        metadata = await self._read_generation_header(selected, sid)
        if metadata is None:
            return None
        try:
            identity = await asyncio.to_thread(_stat_identity, selected.source_path)
        except FileNotFoundError:
            return None
        return SessionPersistenceSnapshot(
            header=metadata.header,
            format_status=metadata.format_status,
            revision=_revision(identity),
            size_bytes=identity.size,
            location=SessionLocation("jsonl", selected.source_path),
        )

    async def list(self) -> Sequence[SessionPersistenceSnapshot]:
        snapshots: builtins.list[SessionPersistenceSnapshot] = []
        listed: set[SessionId] = set()
        pending = builtins.list(self._tracker.pending_entries())
        artifacts = await self._list_artifacts()

        async def stat_artifact(
            artifact: tuple[_GenerationHeader, str],
        ) -> tuple[_GenerationHeader, str, JsonlPhysicalIdentity] | None:
            try:
                identity = await asyncio.to_thread(_stat_identity, artifact[1])
            except FileNotFoundError:
                return None
            return artifact[0], artifact[1], identity

        for entry in await _gather_limited(self._limit, artifacts, stat_artifact):
            if entry is None:
                continue
            metadata, path, identity = entry
            listed.add(SessionId(metadata.header["id"]))
            snapshots.append(
                SessionPersistenceSnapshot(
                    header=metadata.header,
                    format_status=metadata.format_status,
                    revision=_revision(identity),
                    size_bytes=identity.size,
                    location=SessionLocation("jsonl", path),
                )
            )
        for sid, waiting in pending:
            if sid not in listed:
                snapshots.append(
                    SessionPersistenceSnapshot(header=waiting.header, revision=waiting.revision)
                )
        return snapshots

    # --- JsonlHandleStorage -------------------------------------------------------------

    async def read_stored_log(self, path: str, expected_id: SessionId) -> SessionHandleReadResult:
        probe = _revision(await asyncio.to_thread(_stat_identity, path))
        memoized = self._cold_log_memo.get(expected_id)
        if memoized is not None and memoized.revision == probe:
            self._cold_log_memo.move_to_end(expected_id)
            return memoized
        current = await read_stable_jsonl_file(path)
        return await self._decode_stored_log(
            path, expected_id, current.bytes, _revision(current.identity)
        )

    async def resolve_current_log(self, session_id: SessionId) -> str | None:
        await self._ensure_root_encoding()
        selected = await self._find_log(session_id)
        if selected is None:
            return None
        if selected.source_version == SESSION_FORMAT_VERSION:
            return selected.source_path
        raise self._version_refusal(session_id, selected)

    async def persist_batch(
        self,
        header: SessionHeader,
        events: Sequence[SessionEvent],
        is_materialized: bool,
        inherited_event_count: SessionLogOffset,
    ) -> None:
        self._cold_log_memo.pop(SessionId(header["id"]), None)
        await self._ensure_root_encoding()
        if is_materialized:
            path = log_path(self._root, header.get("cwd"), header["id"], self._compression)
            plaintext = (event_lines(events) + "\n").encode("utf-8", "surrogatepass")
            content = compress_zstd_frame(plaintext) if self._compression == "zstd" else plaintext
            await asyncio.to_thread(_append_synced, path, content)
        else:
            await self._materialize(header, inherited_event_count, events)
            self._tracker.materialized(SessionId(header["id"]))

    async def persist_header(
        self, header: SessionHeader, inherited_event_count: SessionLogOffset
    ) -> None:
        self._cold_log_memo.pop(SessionId(header["id"]), None)
        await self._ensure_root_encoding()
        await self._materialize(header, inherited_event_count, [])
        self._tracker.materialized(SessionId(header["id"]))

    async def truncate_torn_tail(self, header: SessionHeader, truncate_to: int) -> None:
        self._cold_log_memo.pop(SessionId(header["id"]), None)
        path = log_path(self._root, header.get("cwd"), header["id"], self._compression)
        await asyncio.to_thread(_truncate_synced, path, truncate_to)
        logger.warning(
            '%s: session "%s" recovered from a torn tail; incomplete tail bytes were discarded',
            NAME,
            header["id"],
        )

    def has_pending_session(self, session_id: SessionId) -> bool:
        return self._tracker.has_pending(session_id)

    def release_handle(self, handle: JsonlSessionHandle, materialized: bool) -> None:
        self._tracker.release(handle, materialized)

    async def acquire_write_lease(self, header: SessionHeader) -> SessionWriteLease:
        await self._reject_opposite_artifact(header.get("cwd"), header["id"])
        return await SessionWriteLease.acquire(
            session_dir(self._root, header.get("cwd"), header["id"]), SessionId(header["id"])
        )

    # --- reading -----------------------------------------------------------------------

    def _version_refusal(
        self, session_id: SessionId, selected: _ResolvedGeneration
    ) -> SessionFormatUnsupportedError:
        reason = session_format_version_refusal(session_id, selected.source_version)
        return SessionFormatUnsupportedError(
            f"{reason} (raw log: {selected.source_path})",
            SessionLocation("jsonl", selected.source_path),
        )

    async def _require_stored_log(self, session_id: SessionId) -> _StoredLog:
        selected = await self._find_log(session_id)
        if selected is None:
            raise SessionPersistenceNotFoundError(session_id)
        if selected.source_version != SESSION_FORMAT_VERSION:
            if selected.source_version > SESSION_FORMAT_VERSION:
                header = await self._read_generation_header(selected, session_id)
                if header is None:
                    error = SessionPersistenceCorruptionError(
                        f'session "{session_id}": stored log has a malformed header '
                        f"(raw log: {selected.source_path})"
                    )
                    raise error from ValueError("malformed Session header")
            raise self._version_refusal(session_id, selected)
        result = await self.read_stored_log(selected.source_path, session_id)
        assert isinstance(result, _StoredLog)
        return result

    async def _decode_stored_log(
        self, path: str, expected_id: SessionId, buffer: bytes, revision: SessionPersistenceRevision
    ) -> _StoredLog:
        try:
            if self._compression == "zstd":
                (
                    meta,
                    inherited_event_count,
                    decoded_events,
                    torn_truncate_to,
                    recovered_tail,
                ) = self._decode_zstd_prefix(buffer)
            else:
                scan = scan_log(buffer)
                meta = scan.meta
                inherited_event_count = scan.inherited_event_count
                decoded_events = scan.events
                torn_truncate_to = (
                    scan.committed_bytes if scan.committed_bytes < len(buffer) else None
                )
                recovered_tail = ()
        except SessionFormatUnsupportedError as error:
            raise SessionFormatUnsupportedError(
                f"{error} (raw log: {path})", SessionLocation("jsonl", path)
            ) from error
        except Exception as error:
            raise SessionPersistenceCorruptionError(
                f'session "{expected_id}": stored log is corrupt: {error} (raw log: {path})'
            ) from error
        await self._assert_stored_identity(path, SESSION_FORMAT_VERSION, meta, expected_id)
        assert_stored_id(expected_id, meta)
        events = validate_stored_events(meta, builtins.list(decoded_events), self._locate(meta))
        stored = _StoredLog(
            event_state="shared-frozen",
            events=tuple(events),
            meta=meta,
            torn_truncate_to=torn_truncate_to,
            recovered_tail=tuple(recovered_tail),
            inherited_event_count=inherited_event_count,
            revision=revision,
        )
        self._memoize(expected_id, stored)
        return stored

    @staticmethod
    def _decode_zstd_prefix(
        buffer: bytes,
    ) -> tuple[
        SessionHeader,
        SessionLogOffset,
        builtins.list[SessionEvent],
        int | None,
        tuple[SessionEvent, ...],
    ]:
        scan = scan_zstd_frames(buffer)
        if not scan.frames:
            raise ValueError("empty or header-less Zstandard session log")
        decoded = iter(decode_zstd_frames(buffer, scan.frames))
        header_frame = next(decoded)
        _assert_zstd_header_frame(header_frame)
        scanner = SessionLogScanner(header_frame)
        for plaintext in decoded:
            scanner.write(plaintext)
        complete = scanner.checkpoint()
        if complete.committed_bytes != complete.input_bytes:
            raise ValueError(
                "corrupt Zstandard session log: complete frame contains a torn JSONL record"
            )
        if scan.torn_start is None:
            prefix = scanner.finish()
            return (
                prefix.meta,
                prefix.inherited_event_count,
                prefix.events,
                None,
                (),
            )
        scanner.write(decompress_zstd_prefix(buffer[scan.torn_start :]))
        prefix = scanner.finish()
        return (
            prefix.meta,
            prefix.inherited_event_count,
            prefix.events,
            scan.torn_start,
            tuple(prefix.events[complete.event_count :]),
        )

    def _memoize(self, session_id: SessionId, stored: _StoredLog) -> None:
        self._cold_log_memo.pop(session_id, None)
        self._cold_log_memo[session_id] = stored
        while len(self._cold_log_memo) > _COLD_LOG_MEMO_MAX_ENTRIES:
            self._cold_log_memo.popitem(last=False)

    async def _read_generation_header(
        self, selected: _ResolvedGeneration, expected_id: SessionId | None = None
    ) -> _GenerationHeader | None:
        try:
            reader = _read_first_zstd_line if self._compression == "zstd" else _read_first_line
            first = await asyncio.to_thread(reader, selected.source_path)
        except FileNotFoundError:
            return None
        if first is None:
            return None
        try:
            value = parse_json(first)
        except ValueError:
            return None
        assert_no_retired_header_fields(value)
        result = session_format_catalog.read_header(value)
        if result.stored_version is not None and result.stored_version != selected.source_version:
            raise ValueError(
                f"session generation filename identifies v{selected.source_version}, but its "
                f"header identifies v{result.stored_version}"
            )
        if result.status == "unsupported":
            physical_id = str(value.get("id")) if isinstance(value, Mapping) else "None"
            reason = result.reason or ""
            if result.stored_version is not None and result.stored_version > SESSION_FORMAT_VERSION:
                reason = session_format_version_refusal(physical_id, result.stored_version)
            raise SessionFormatUnsupportedError(
                f"{reason} (raw log: {selected.source_path})",
                SessionLocation("jsonl", selected.source_path),
            )
        if result.status == "malformed" or result.header is None:
            return None
        header = self._current_header(result.header)
        await self._assert_stored_identity(
            selected.source_path, selected.source_version, header, expected_id
        )
        status: Literal["current", "migration-required"] = (
            "current" if result.status == "current" else "migration-required"
        )
        return _GenerationHeader(header, status)

    def _current_header(self, header: Mapping[str, Any]) -> SessionHeader:
        if header["version"] != SESSION_FORMAT_VERSION:
            raise ValueError(
                f"format catalog returned non-current logical header v{header['version']}"
            )
        current: dict[str, object] = {
            "version": SESSION_FORMAT_VERSION,
            "id": header["id"],
            "createdAt": header["createdAt"],
        }
        for key in ("cwd", "parentSession"):
            if key in header:
                current[key] = header[key]
        current["isSeeded"] = header["isSeeded"]
        if "origin" in header:
            current["origin"] = header["origin"]
        current["delegationDepth"] = header["delegationDepth"]
        if "agentPreset" in header:
            current["agentPreset"] = header["agentPreset"]
        return current

    async def _assert_stored_identity(
        self,
        path: str,
        stored_version: int,
        meta: SessionHeader,
        expected_id: SessionId | None = None,
    ) -> None:
        if expected_id is not None and meta["id"] != expected_id:
            raise ValueError(
                f'corrupt session log "{path}": requested id "{expected_id}" does not match '
                f'header id "{meta["id"]}"'
            )
        try:
            expected_path = generation_log_path(
                self._root, meta.get("cwd"), meta["id"], stored_version, self._compression
            )
        except Exception as error:
            raise ValueError(
                f'corrupt session log "{path}": header id cannot name a storage path'
            ) from error
        if path != expected_path and not await asyncio.to_thread(
            self._same_file, path, expected_path
        ):
            raise ValueError(
                f'corrupt session log "{path}": header id "{meta["id"]}" and cwd identify '
                f'"{expected_path}"'
            )

    @staticmethod
    def _same_file(path: str, expected_path: str) -> bool:
        if not os.path.exists(expected_path):
            return False
        return os.path.realpath(path) == os.path.realpath(expected_path)

    # --- directory scanning ------------------------------------------------------------

    def _assert_usable_root(self) -> None:
        try:
            os.listdir(self._root)
        except FileNotFoundError:
            return

    def _resolve_generation_in_directory(self, directory: str) -> _ResolvedGeneration | None:
        try:
            names = os.listdir(directory)
        except FileNotFoundError:
            return None
        generations: builtins.list[tuple[int, str]] = []
        opposite: builtins.list[str] = []
        for name in names:
            version = parse_generation_log_filename(name, self._compression)
            if version is not None:
                generations.append((version, os.path.join(directory, name)))
                continue
            if parse_generation_log_filename(name, self._opposite_compression()) is not None:
                opposite.append(os.path.join(directory, name))
        if opposite:
            raise self._encoding_mismatch(opposite[0])
        if not generations:
            return None
        version, path = max(generations)
        return _ResolvedGeneration(
            source_path=path,
            source_version=version,
            current_path=os.path.join(
                directory,
                generation_log_filename(session_format_catalog.current_version, self._compression),
            ),
        )

    async def _find_log(self, session_id: SessionId) -> _ResolvedGeneration | None:
        def find() -> builtins.list[_ResolvedGeneration]:
            matches: builtins.list[_ResolvedGeneration] = []
            for project in self._list_project_dirs():
                self._reject_legacy_flat_artifact(project, session_id)
                found = self._resolve_generation_in_directory(
                    os.path.join(project, encode_segment(session_id))
                )
                if found is not None:
                    matches.append(found)
            return matches

        matches = await asyncio.to_thread(find)
        if len(matches) > 1:
            raise ValueError(
                f'duplicate JSONL session id "{session_id}" appears in multiple project directories'
            )
        return matches[0] if matches else None

    def _list_project_dirs(self) -> builtins.list[str]:
        try:
            entries = builtins.list(os.scandir(self._root))
        except FileNotFoundError:
            return []
        # Like Node's Dirent.isDirectory(): a symlink is not a directory entry here.
        return [
            os.path.join(self._root, e.name) for e in entries if e.is_dir(follow_symlinks=False)
        ]

    def _list_session_dirs(self, project: str) -> builtins.list[str]:
        entries = builtins.list(os.scandir(project))
        for entry in entries:
            if entry.is_file(follow_symlinks=False) and (
                entry.name.endswith(".jsonl") or entry.name.endswith(".jsonl.zstd")
            ):
                raise self._legacy_layout(os.path.join(project, entry.name))
        return [os.path.join(project, e.name) for e in entries if e.is_dir(follow_symlinks=False)]

    async def _map_session_directories(
        self, task: Callable[[str], Awaitable[T]]
    ) -> builtins.list[T]:
        async with self._limit:
            projects = await asyncio.to_thread(self._list_project_dirs)

        async def session_dirs(project: str) -> builtins.list[str]:
            async with self._limit:
                return await asyncio.to_thread(self._list_session_dirs, project)

        outcomes = await asyncio.gather(
            *(session_dirs(project) for project in projects), return_exceptions=True
        )
        # Like upstream: directories of projects before the first failing one still run,
        # and their failures win over that project's, as a sequential traversal reports.
        dirs: builtins.list[str] = []
        project_failure: BaseException | None = None
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                project_failure = outcome
                break
            dirs.extend(outcome)
        results = await _gather_limited(self._limit, dirs, task)
        if project_failure is not None:
            raise project_failure
        return results

    async def _list_artifacts(self) -> builtins.list[tuple[_GenerationHeader, str]]:
        await self._ensure_root_encoding()

        async def resolve(directory: str) -> _ResolvedGeneration | None:
            return await asyncio.to_thread(self._resolve_generation_in_directory, directory)

        generations = [g for g in await self._map_session_directories(resolve) if g is not None]

        async def header_of(
            selected: _ResolvedGeneration,
        ) -> tuple[_ResolvedGeneration, _GenerationHeader | None]:
            try:
                return selected, await self._read_generation_header(selected)
            except (SessionFormatUnsupportedError, SessionPersistenceCorruptionError):
                return selected, None

        artifacts: builtins.list[tuple[_GenerationHeader, str]] = []
        ids: set[str] = set()
        for selected, metadata in await _gather_limited(self._limit, generations, header_of):
            if metadata is None:
                continue
            sid = metadata.header["id"]
            if sid in ids:
                raise ValueError(
                    f'duplicate JSONL session id "{sid}" appears in multiple project directories'
                )
            ids.add(sid)
            artifacts.append((metadata, selected.source_path))
        return artifacts

    def _ensure_root_encoding(self) -> asyncio.Future[None]:
        if self._root_encoding_check is None:
            self._root_encoding_check = asyncio.ensure_future(self._check_root_encoding())
        return asyncio.shield(self._root_encoding_check)

    async def _check_root_encoding(self) -> None:
        async def check(directory: str) -> None:
            incompatible = await asyncio.to_thread(self._find_opposite_generation, directory)
            if incompatible is not None:
                raise self._encoding_mismatch(incompatible)

        await self._map_session_directories(check)

    def _reject_legacy_flat_artifact(self, project: str, session_id: SessionId) -> None:
        encoded = encode_segment(session_id)
        compressions: tuple[JsonlCompression, ...] = ("zstd", "none")
        for compression in compressions:
            path = os.path.join(project, encoded + log_suffix(compression))
            if self._exists(path):
                raise self._legacy_layout(path)

    @staticmethod
    def _exists(path: str) -> bool:
        """Upstream ``exists()``: only absence (ENOENT) means false; other errors raise."""
        try:
            with open(path, "rb"):
                return True
        except FileNotFoundError:
            if sys.platform == "win32":
                parent = os.path.dirname(path)
                if os.path.exists(parent) and not os.path.isdir(parent):
                    raise NotADirectoryError(
                        f"ENOTDIR: parent path exists but is not a directory: {parent}"
                    ) from None
            return False

    async def _reject_opposite_artifact(self, cwd: str | None, session_id: str) -> None:
        directory = session_dir(self._root, cwd, session_id)
        path = await asyncio.to_thread(self._find_opposite_generation, directory)
        if path is not None:
            raise self._encoding_mismatch(path)

    def _find_opposite_generation(self, directory: str) -> str | None:
        try:
            names = os.listdir(directory)
        except FileNotFoundError:
            return None
        found = [
            (version, name)
            for name in names
            if (version := parse_generation_log_filename(name, self._opposite_compression()))
            is not None
        ]
        return os.path.join(directory, max(found)[1]) if found else None

    def _opposite_compression(self) -> JsonlCompression:
        return "none" if self._compression == "zstd" else "zstd"

    def _encoding_mismatch(self, path: str) -> ValueError:
        return ValueError(
            f"session artifact {stringify_json(path)} uses "
            f"{log_suffix(self._opposite_compression())}, but this backend is configured for "
            f"compression {stringify_json(self._compression)}; use a separate root or select "
            "the matching compression mode"
        )

    def _legacy_layout(self, path: str) -> ValueError:
        return ValueError(
            f"session artifact {stringify_json(path)} uses the unsupported flat-file layout; "
            "use a separate root or move it into a project/session directory before loading"
        )

    # --- writing -----------------------------------------------------------------------

    async def _materialize(
        self,
        meta: SessionHeader,
        inherited_event_count: SessionLogOffset,
        events: Sequence[SessionEvent],
    ) -> None:
        cwd = meta.get("cwd")
        await self._reject_opposite_artifact(cwd, meta["id"])
        header = (
            stringify_json(
                to_header_line(meta, inherited_event_count if meta["isSeeded"] else None)
            )
            + "\n"
        )
        body = event_lines(events) + "\n" if events else ""
        if self._compression == "zstd":
            content = compress_zstd_frame(header)
            if body:
                content += compress_zstd_frame(body)
        else:
            content = (header + body).encode("utf-8", "surrogatepass")
        root = self._root
        project = project_dir(root, cwd)
        directory = session_dir(root, cwd, meta["id"])
        final_path = log_path(root, cwd, meta["id"], self._compression)
        sid = meta["id"]

        def materialize() -> None:
            os.makedirs(root, mode=0o700, exist_ok=True)
            _fsync_dir(os.path.dirname(root))
            os.makedirs(project, mode=0o700, exist_ok=True)
            _fsync_dir(root)
            os.makedirs(directory, mode=0o700, exist_ok=True)
            _fsync_dir(project)
            if self._resolve_generation_in_directory(directory) is not None:
                raise ValueError(
                    f'refusing to materialize "{sid}": a log already exists on disk (open it '
                    "instead)"
                )
            tmp = _write_synced_temp(final_path, content)
            _publish_new(tmp, final_path)

        await asyncio.to_thread(materialize)


class _JsonlConfigValidator:
    """deepseek-cordis ``ConfigValidator`` for this plugin (upstream: the ``Config`` schema)."""

    def validate(self, value: object) -> JsonlConfig:
        return value if isinstance(value, JsonlConfig) else JsonlConfig.model_validate(value)


name = NAME
"""Plugin name read by the deepseek-cordis Loader (upstream ``export const name``)."""

Config = _JsonlConfigValidator()
"""Plugin config validator read by the Loader (upstream ``export const Config``)."""


async def apply(ctx: Context, config: JsonlConfig | Mapping[str, Any]) -> None:
    """Plugin entry: mount the backend and route live session events to its writers."""
    settings = Config.validate(config)
    fibers: list[Fiber] = []

    def setup() -> Iterator[Callable[[], object]]:
        tracker = JsonlBackendTracker(NAME)
        yield tracker.install(ctx)

        def service_plugin(service_ctx: Context, _config: object) -> None:
            JsonlSessionPersistence(service_ctx, settings, tracker)

        fiber = ctx.plugin(service_plugin)
        fibers.append(fiber)
        yield fiber.dispose

    ctx.effect(setup, f"{NAME} resources")
    await fibers[0].wait()
