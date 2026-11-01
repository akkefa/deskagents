"""On-disk format of the JSONL backend: layout, header and event lines, log scanning.

Upstream: packages/session/session-persistence-jsonl/src/format.ts

Python differences: path segments are escaped per UTF-16 code unit, exactly as upstream
iterates a JavaScript string, so ids outside the Basic Multilingual Plane map to the same
directory names. Record text is decoded as UTF-8 with replacement characters, as Node's
``Buffer.toString('utf8')`` does.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from deskagents.core.session import (
    KNOWN_SESSION_EVENT_TYPES,
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
    session_log_offset,
)
from deskagents.session.format import (
    SessionFormatRecovery,
    SessionFormatRestore,
    SessionFormatRestoreOptions,
    SessionFormatUnsupportedMigrationError,
    parse_session_format_log_filename,
    session_format_log_filename,
)
from deskagents.session.format_catalog import session_format_catalog
from deskagents.session.format_v3_to_v4 import (
    assert_released_v4_relationships,
    assert_v4_row_admission,
)
from deskagents.session.persistence import (
    SessionFormatUnsupportedError,
    SessionStorageMetadata,
    session_format_version_refusal,
)
from deskagents.util.values import is_safe_integer, parse_json, stringify_json

JsonlCompression = Literal["zstd", "none"]
"""Physical encoding selected for JSONL session artifacts."""

_SAFE = re.compile(r"[A-Za-z0-9._-]")
_HEADER_REQUIRED = ("type", "version", "id", "createdAt", "isSeeded", "delegationDepth")
_HEADER_OPTIONAL = ("cwd", "parentSession", "origin", "agentPreset")
_HEADER_KEYS = frozenset(_HEADER_REQUIRED + _HEADER_OPTIONAL)


def log_suffix(compression: JsonlCompression) -> str:
    """``.jsonl.zstd`` for Zstandard or ``.jsonl`` for plaintext."""
    return f".jsonl{_compression_suffix(compression)}"


def _compression_suffix(compression: JsonlCompression) -> str:
    return ".zstd" if compression == "zstd" else ""


def generation_log_filename(version: int, compression: JsonlCompression) -> str:
    """The canonical filename of one immutable Session format generation."""
    return f"{session_format_log_filename(version)}{_compression_suffix(compression)}"


def parse_generation_log_filename(filename: str, compression: JsonlCompression) -> int | None:
    """Parse one canonical generation filename for the selected encoding, or ``None``."""
    suffix = _compression_suffix(compression)
    if not filename.endswith(suffix):
        return None
    return parse_session_format_log_filename(filename[: len(filename) - len(suffix)])


def assert_no_retired_header_fields(value: object) -> None:
    """Refuse policy fields that never belong to a released Session header."""
    if isinstance(value, Mapping) and ("sandboxMode" in value or "approvalPolicy" in value):
        raise ValueError("session header uses retired policy baseline fields")


def to_header_line(
    header: SessionHeader, inherited_event_count: int | None = None
) -> Mapping[str, Any]:
    """The ``type: "session"`` physical header record for a header."""
    if header["isSeeded"] and inherited_event_count is None:
        raise ValueError("seeded session header requires an inherited event count")
    cut = session_log_offset(inherited_event_count or 0)
    if not header["isSeeded"] and cut != 0:
        raise ValueError("unseeded session header inherited event count must be 0")
    # Like upstream's `{...header, delegationDepth: header.delegationDepth ?? 0}`: an
    # existing key keeps its position, an absent one is appended.
    depth = header.get("delegationDepth")
    logical = {**header, "delegationDepth": 0 if depth is None else depth}
    encoded: Mapping[str, Any] = session_format_catalog.encode_current_header(logical, cut)
    return encoded


def _from_header_line(line: Mapping[str, Any]) -> SessionStorageMetadata:
    meta: dict[str, object] = {
        "version": SESSION_FORMAT_VERSION,
        "id": line["id"],
        "createdAt": line["createdAt"],
    }
    if "cwd" in line:
        meta["cwd"] = line["cwd"]
    if "parentSession" in line:
        meta["parentSession"] = line["parentSession"]
    meta["isSeeded"] = line["isSeeded"]
    if "origin" in line:
        meta["origin"] = line["origin"]
    meta["delegationDepth"] = line["delegationDepth"]
    if "agentPreset" in line:
        meta["agentPreset"] = line["agentPreset"]
    return SessionStorageMetadata(meta, SessionLogOffset(0))


def _count(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and is_safe_integer(value)
        and value >= 0
        and not (isinstance(value, float) and str(value).startswith("-"))
    )


def _is_header_line(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not all(key in value for key in _HEADER_REQUIRED):
        return False
    if not all(key in _HEADER_KEYS for key in value):
        return False
    cwd = value.get("cwd")
    return (
        value["type"] == "session"
        and isinstance(value["version"], int | float)
        and not isinstance(value["version"], bool)
        and isinstance(value["id"], str)
        and _count(value["createdAt"])
        and _count(value["delegationDepth"])
        and ("cwd" not in value or (isinstance(cwd, str) and os.path.isabs(cwd)))
        and ("parentSession" not in value or isinstance(value["parentSession"], str))
        and isinstance(value["isSeeded"], bool)
        and ("origin" not in value or value["origin"] == "subagent")
        and ("agentPreset" not in value or isinstance(value["agentPreset"], str))
    )


def _code_units(text: str) -> list[int]:
    data = text.encode("utf-16-le", "surrogatepass")
    return [data[i] | (data[i + 1] << 8) for i in range(0, len(data), 2)]


def encode_segment(raw: str) -> str:
    """Encode any string as one safe path segment, injectively (``~XXXX`` escapes)."""
    if not raw:
        raise ValueError("cannot encode an empty path segment")
    if raw == ".":
        return "~002E"
    if raw == "..":
        return "~002E~002E"
    out = []
    for code in _code_units(raw):
        ch = chr(code)
        if ch != "~" and code < 0x80 and _SAFE.fullmatch(ch):
            out.append(ch)
        else:
            out.append(f"~{code:04X}")
    return "".join(out)


def project_key(cwd: str) -> str:
    """The readable, bounded directory key for a project path."""
    if not cwd:
        raise ValueError("cannot encode an empty project path")
    readable = []
    separator_run = False
    for code in _code_units(cwd):
        ch = chr(code)
        if ch in "/\\:":
            if not separator_run:
                readable.append("-")
            separator_run = True
        elif ch != "~" and code < 0x80 and _SAFE.fullmatch(ch):
            readable.append(ch)
            separator_run = False
        else:
            readable.append(f"~{code:04X}")
            separator_run = False
    slug = "".join(readable).lstrip("-") or "root"
    return f"--{slug[:251]}--"


def project_dir(root: str, cwd: str | None) -> str:
    """The project directory under ``root``; ``None`` selects ``_no-cwd``."""
    return os.path.join(root, "_no-cwd" if cwd is None else project_key(cwd))


def session_dir(root: str, cwd: str | None, session_id: SessionId | str) -> str:
    """The directory owned by one session."""
    return os.path.join(project_dir(root, cwd), encode_segment(session_id))


def generation_log_path(
    root: str,
    cwd: str | None,
    session_id: SessionId | str,
    version: int,
    compression: JsonlCompression,
) -> str:
    """The path of one immutable Session format generation."""
    return os.path.join(
        session_dir(root, cwd, session_id), generation_log_filename(version, compression)
    )


def log_path(
    root: str, cwd: str | None, session_id: SessionId | str, compression: JsonlCompression
) -> str:
    """The current generation's append target for a Session."""
    return generation_log_path(root, cwd, session_id, SESSION_FORMAT_VERSION, compression)


def event_line(event: SessionEvent) -> str:
    """One current event as one JSONL record, without its newline."""
    return stringify_json(session_format_catalog.encode_current_event(event))


def event_lines(events: Sequence[SessionEvent]) -> str:
    """A batch as JSONL lines with no trailing newline (the writer adds it)."""
    return "\n".join(event_line(event) for event in events)


@dataclass(frozen=True)
class SessionLogScan:
    meta: SessionHeader
    inherited_event_count: SessionLogOffset
    events: list[SessionEvent]
    committed_bytes: int


@dataclass(frozen=True)
class ScanCheckpoint:
    input_bytes: int
    committed_bytes: int
    event_count: SessionLogOffset


def _refuse_foreign_format_version(parsed: Mapping[str, Any]) -> None:
    version = parsed.get("version")
    if not isinstance(version, int | float) or isinstance(version, bool):
        return
    if version == SESSION_FORMAT_VERSION:
        return
    session_id = parsed.get("id")
    raise SessionFormatUnsupportedError(
        session_format_version_refusal(
            session_id if isinstance(session_id, str) else str(session_id), int(version)
        )
    )


def _decode(record: bytes) -> str:
    return record.decode("utf-8", errors="replace")


def _parse_header_record(record: bytes) -> tuple[SessionHeader, SessionFormatRestore]:
    if not record or record[-1] != 0x0A or record.index(0x0A) != len(record) - 1:
        raise ValueError("empty or header-less session log")
    try:
        parsed = parse_json(_decode(record[:-1]))
    except ValueError:
        raise ValueError("corrupt session log: header line is not valid JSON") from None
    if not isinstance(parsed, Mapping):
        raise ValueError("corrupt session log: first line is not a JSON object")
    _refuse_foreign_format_version(parsed)
    assert_no_retired_header_fields(parsed)
    if not _is_header_line(parsed):
        raise ValueError("corrupt session log: first line is not a session header")
    try:
        restore = session_format_catalog.create_restore(
            parsed, SessionFormatRestoreOptions(recovery="strict", validation="transformed")
        )
    except Exception:
        raise ValueError("corrupt session log: first line is not a session header") from None
    return _from_header_line(parsed).meta, restore


class SessionLogScanner:
    """Incrementally scan complete JSONL event records after a separate header record.

    Newline search and byte offsets stay on raw bytes; only complete records are
    decoded. An incomplete final record is kept until more bytes arrive.
    """

    def __init__(self, header_record: bytes, recovery: SessionFormatRecovery = "recoverable"):
        self._meta, self._restore = _parse_header_record(header_record)
        self._recovery = recovery
        self._event_count = 0
        self._fragment = bytearray()
        self._input_bytes = len(header_record)
        self._committed_bytes = len(header_record)
        self._event_line = 0
        self._issue: Exception | None = None
        self._finished = False

    def write(self, chunk: bytes) -> None:
        """Consume the next plaintext chunk, retaining only an incomplete final record."""
        if self._finished:
            raise ValueError("cannot write to a finished session log scanner")
        chunk_start = self._input_bytes
        self._input_bytes += len(chunk)
        line_start = 0
        newline = chunk.find(0x0A)
        while newline != -1:
            if self._fragment:
                self._fragment += chunk[line_start:newline]
                line = bytes(self._fragment)
                self._fragment.clear()
            else:
                line = chunk[line_start:newline]
            self._consume_event_line(line, chunk_start + newline + 1)
            line_start = newline + 1
            newline = chunk.find(0x0A, line_start)
        if line_start < len(chunk):
            self._fragment += chunk[line_start:]

    def checkpoint(self) -> ScanCheckpoint:
        """Progress before appending a recoverable torn-frame prefix."""
        return ScanCheckpoint(
            self._input_bytes, self._committed_bytes, SessionLogOffset(self._event_count)
        )

    def finish(self) -> SessionLogScan:
        """Finish scanning; a final record without a newline is a torn tail."""
        self._finished = True
        artifact = self._restore.finish()
        assert_released_v4_relationships(artifact, KNOWN_SESSION_EVENT_TYPES)
        return SessionLogScan(
            self._meta,
            session_log_offset(artifact.inherited_event_count),
            list(artifact.events),
            self._committed_bytes,
        )

    def _consume_event_line(self, line: bytes, end_byte: int) -> None:
        self._event_line += 1
        try:
            decoded = parse_json(_decode(line))
        except ValueError:
            issue = ValueError(
                f"corrupt session log: unparsable committed event at line {self._event_line}"
            )
            if self._recovery == "strict":
                raise issue from None
            if self._issue is None:
                self._issue = issue
            return
        try:
            assert_v4_row_admission(decoded, KNOWN_SESSION_EVENT_TYPES)
        except SessionFormatUnsupportedMigrationError as error:
            raise SessionFormatUnsupportedError(str(error)) from error
        is_turn_end = isinstance(decoded, Mapping) and decoded.get("type") == "turn/end"
        if self._issue is not None:
            if is_turn_end:
                raise self._issue
            return
        try:
            self._restore.decode_row(decoded)
        except Exception as error:
            issue = ValueError(
                f"corrupt session log: invalid committed event at line {self._event_line}: {error}"
            )
            issue.__cause__ = error
            if self._recovery == "strict":
                raise issue from error
            self._issue = issue
            if is_turn_end:
                raise issue from error
            return
        self._event_count += 1
        self._committed_bytes = end_byte


def scan_log(buffer: bytes) -> SessionLogScan:
    """Parse a complete or torn JSONL buffer into its preserved event prefix."""
    header_end = buffer.find(0x0A)
    if header_end == -1:
        raise ValueError("empty or header-less session log")
    scanner = SessionLogScanner(buffer[: header_end + 1])
    scanner.write(buffer[header_end + 1 :])
    return scanner.finish()
