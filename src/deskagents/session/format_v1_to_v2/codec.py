"""Upstream: packages/session/session-format-v1-to-v2/src/codec.ts

The frozen released-v2 physical row framing: one header record, then one event record
per row with compacted ``sourceEventSeqs`` ranges. Later formats (v3, v4) reuse it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from deskagents.session.format import (
    SessionFormatArtifactDecoder,
    SessionFormatCodec,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatHeader,
    SessionFormatJsonObject,
    SessionFormatJsonValue,
    SessionFormatMigrationContext,
    SessionFormatRecovery,
    session_format_count,
    session_format_safe_integer,
    snapshot_session_format_json,
)
from deskagents.util.values import JsonValue, deep_freeze, format_json_number

from .validation import assert_released_v2_header, is_json_number

__all__ = ["ReleasedSessionFormatCodec", "released_v2_session_format_codec"]

_HEADER_REQUIRED = ("type", "version", "id", "createdAt", "isSeeded", "delegationDepth")
_HEADER_OPTIONAL = ("cwd", "parentSession", "origin", "agentPreset")
_EVENT_REQUIRED = ("type", "seq", "time", "data")
_EVENT_OPTIONAL = ("ignorable", "sourceEventSeqs", "surfaceOp")
_EVENT_KEYS = frozenset((*_EVENT_REQUIRED, *_EVENT_OPTIONAL))


@dataclass(frozen=True)
class ReleasedSessionFormatCodec(SessionFormatCodec):
    """A released physical codec that is also the stateless encoder for its records.

    Upstream declares one frozen object that satisfies both ``SessionFormatCodec`` and
    ``SessionFormatCurrentEncoder``; here the encoder members are extra fields.
    """

    encode_header: Callable[[SessionFormatHeader, int], SessionFormatJsonObject]
    """Encode one physical header record."""
    encode_event: Callable[[SessionFormatEvent], SessionFormatJsonObject]
    """Encode one physical event record."""


def _json_record(value: object, label: str) -> SessionFormatJsonObject:
    if not isinstance(value, Mapping):
        raise SessionFormatError(f"{label} must be an object")
    return value


def _exact_keys(
    record: Mapping[str, object],
    required: Sequence[str],
    optional: Sequence[str],
    label: str,
) -> None:
    allowed = {*required, *optional}
    missing = next((key for key in required if key not in record), None)
    if missing is not None:
        raise SessionFormatError(f"{label} lacks {missing}")
    unexpected = next((key for key in record if key not in allowed), None)
    if unexpected is not None:
        raise SessionFormatError(f"{label} has unexpected field {unexpected}")


def _decode_physical_header(value: object) -> SessionFormatHeader:
    snapshot = snapshot_session_format_json(value, "released v2 physical header")
    record = _json_record(snapshot, "released v2 physical header")
    _exact_keys(record, _HEADER_REQUIRED, _HEADER_OPTIONAL, "released v2 physical header")
    if record["type"] != "session" or not is_json_number(record["version"], 2):
        raise SessionFormatError("expected released v2 physical Session header")
    if not isinstance(record["id"], str):
        raise SessionFormatError("released v2 header id must be a string")
    created_at = session_format_count(record["createdAt"], "released v2 header createdAt")
    delegation_depth = session_format_count(
        record["delegationDepth"], "released v2 header delegationDepth"
    )
    if not isinstance(record["isSeeded"], bool):
        raise SessionFormatError("released v2 header isSeeded must be boolean")
    for key in ("cwd", "parentSession", "agentPreset"):
        if key in record and not isinstance(record[key], str):
            raise SessionFormatError(f"released v2 header {key} must be a string")
    if "origin" in record and record["origin"] != "subagent":
        raise SessionFormatError('released v2 header origin must be "subagent"')
    logical: dict[str, JsonValue] = {"version": 2, "id": record["id"], "createdAt": created_at}
    for key in ("cwd", "parentSession"):
        if key in record:
            logical[key] = record[key]
    logical["isSeeded"] = record["isSeeded"]
    if "origin" in record:
        logical["origin"] = record["origin"]
    logical["delegationDepth"] = delegation_depth
    if "agentPreset" in record:
        logical["agentPreset"] = record["agentPreset"]
    header = snapshot_session_format_json(logical, "released v2 logical header")
    assert isinstance(header, Mapping)
    assert_released_v2_header(header)
    return header


class _ReleasedV2Decoder:
    """Row-at-a-time released-v2 decoder with an explicit failure policy."""

    def __init__(self, header_value: object, recovery: SessionFormatRecovery) -> None:
        self.header = _decode_physical_header(header_value)
        self._recovery = recovery
        self._row_index = 0
        self._event_count = 0
        self._inherited_event_count: int | None = None
        self._issue: SessionFormatError | None = None

    def decode_row(self, row_value: object, context: SessionFormatMigrationContext) -> None:
        current_row = self._row_index
        self._row_index += 1
        try:
            event = _decode_event(row_value, current_row)
        except Exception as error:
            if isinstance(error, SessionFormatError):
                current = error
            else:
                current = SessionFormatError(f"released v2 row {current_row} is malformed")
                current.__cause__ = error
            if self._recovery == "strict":
                raise current  # noqa: B904 - the cause is set explicitly above
            if self._issue is None:
                self._issue = current
            return
        if self._issue is not None:
            if event["type"] == "turn/end":
                raise self._issue
            return
        if not is_json_number(event["seq"], self._event_count):
            gap = SessionFormatError(
                f"released v2 row {current_row} has seq gap "
                f"(expected {self._event_count}, got {_js_text(event['seq'])})"
            )
            if self._recovery == "strict":
                raise gap
            self._issue = gap
            if event["type"] == "turn/end":
                raise gap
            return
        seq = self._event_count
        self._event_count += 1
        if event["type"] == "session/end-seed":
            data = _json_record(event["data"], f"session/end-seed {seq} data")
            if data.get("inherited") is True:
                self._inherited_event_count = seq
        # Upstream's current-format restore freezes restored events in place; Python
        # freezes a detached copy here so every decoded event is immutable.
        context.emit_event(deep_freeze(event))

    def finish(self, context: SessionFormatMigrationContext) -> int:
        seeded = self.header["isSeeded"]
        if seeded and self._inherited_event_count is None:
            raise SessionFormatError(
                "released v2 seeded Session lacks an inherited end-seed marker"
            )
        if not seeded and self._inherited_event_count is not None:
            raise SessionFormatError(
                "released v2 unseeded Session contains an inherited end-seed marker"
            )
        return self._inherited_event_count or 0


def _js_text(value: object) -> str:
    """Template-literal text for a scalar diagnostic value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        try:
            return format_json_number(value)
        except ValueError:
            return str(value)
    if isinstance(value, list | tuple):
        return ",".join("" if item is None else _js_text(item) for item in value)
    if isinstance(value, Mapping):
        return "[object Object]"
    return str(value)


def _create_decoder(
    header_value: object, recovery: SessionFormatRecovery
) -> SessionFormatArtifactDecoder:
    return _ReleasedV2Decoder(header_value, recovery)


def _decode_event(value: object, row_index: int) -> SessionFormatEvent:
    record = _json_record(value, f"released v2 row {row_index}")
    missing = next((key for key in _EVENT_REQUIRED if key not in record), None)
    if missing is not None:
        raise SessionFormatError(f"released v2 row {row_index} lacks required field {missing}")
    unexpected = next((key for key in record if key not in _EVENT_KEYS), None)
    if unexpected is not None:
        raise SessionFormatError(f"released v2 row {row_index} has unexpected field {unexpected}")
    if not isinstance(record["type"], str):
        raise SessionFormatError(f"released v2 row {row_index} type must be a string")
    session_format_safe_integer(record["time"], f"released v2 row {row_index} time")
    if "ignorable" in record and record["ignorable"] is not True:
        raise SessionFormatError(f"released v2 row {row_index} ignorable must be true when present")
    if "sourceEventSeqs" not in record:
        return record
    seq = session_format_count(record["seq"], f"released v2 row {row_index} seq")
    return {**record, "sourceEventSeqs": _decode_seq_ranges(record["sourceEventSeqs"], seq)}


def _encode_header(
    header: SessionFormatHeader, inherited_event_count: int
) -> SessionFormatJsonObject:
    assert_released_v2_header(header)
    cut = session_format_count(inherited_event_count, "format v2 inherited event count")
    if not header["isSeeded"] and cut != 0:
        raise SessionFormatError("unseeded format v2 Session has inherited events")
    record: dict[str, JsonValue] = {
        "type": "session",
        "version": 2,
        "id": header["id"],
        "createdAt": header["createdAt"],
    }
    for key in ("cwd", "parentSession"):
        if key in header:
            record[key] = header[key]
    record["isSeeded"] = header["isSeeded"]
    if "origin" in header:
        record["origin"] = header["origin"]
    record["delegationDepth"] = header["delegationDepth"]
    if "agentPreset" in header:
        record["agentPreset"] = header["agentPreset"]
    return record


def _encode_source_event_ranges(event: SessionFormatEvent) -> SessionFormatJsonObject:
    if "sourceEventSeqs" not in event:
        return event
    values = event["sourceEventSeqs"]
    assert isinstance(values, list | tuple)
    return {**event, "sourceEventSeqs": _encode_seq_ranges(values)}


def _decode_seq_ranges(value: object, max_entries: int) -> list[JsonValue]:
    if not isinstance(value, list | tuple):
        raise SessionFormatError("sourceEventSeqs must be an array")
    output: list[int] = []
    has_range = False
    for entry in value:
        if not isinstance(entry, list | tuple):
            output.append(session_format_count(entry, "sourceEventSeqs member"))
            continue
        if len(entry) != 2:
            raise SessionFormatError("sourceEventSeqs range must be a [start, end] pair")
        start = session_format_count(entry[0], "sourceEventSeqs range start")
        end = session_format_count(entry[1], "sourceEventSeqs range end")
        if start > end or end >= max_entries or end - start + 1 > max_entries - len(output):
            raise SessionFormatError("sourceEventSeqs range exceeds its event seq")
        output.extend(range(start, end + 1))
        has_range = True
    seen: set[int] = set()
    for source in output:
        if source >= max_entries or source in seen:
            raise SessionFormatError("sourceEventSeqs ranges must contain unique earlier seqs")
        seen.add(source)
    if has_range and any(
        index > 0 and source <= output[index - 1] for index, source in enumerate(output)
    ):
        raise SessionFormatError("sourceEventSeqs ranges must be strictly increasing")
    return list(output)


def _encode_seq_ranges(values: Sequence[SessionFormatJsonValue]) -> list[JsonValue]:
    numbers: Sequence[int] = values  # type: ignore[assignment]
    if any(index > 0 and value <= numbers[index - 1] for index, value in enumerate(numbers)):
        return list(values)
    output: list[JsonValue] = []
    index = 0
    while index < len(numbers):
        start = numbers[index]
        end = start
        while index + 1 < len(numbers) and numbers[index + 1] == end + 1:
            index += 1
            end += 1
        output.append([start, end] if end - start >= 2 else start)
        if end - start == 1:
            output.append(end)
        index += 1
    return output


released_v2_session_format_codec = ReleasedSessionFormatCodec(
    version=2,
    decode_header=_decode_physical_header,
    create_decoder=_create_decoder,
    encode_header=_encode_header,
    encode_event=_encode_source_event_ranges,
)
"""Frozen physical JSON codec for released v2."""
