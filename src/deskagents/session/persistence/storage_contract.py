"""Backend-shared storage validation: version gate, fail-closed vocabulary, batch snapshots.

Upstream: packages/session/session-persistence/src/storage-contract.ts
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any

from deskagents.core.session import (
    KNOWN_SESSION_EVENT_TYPES,
    SESSION_FORMAT_VERSION,
    SessionEvent,
    SessionHeader,
    SessionId,
    adopt_session_event,
)
from deskagents.util.values import is_safe_integer, snapshot_json_value

from .errors import (
    SessionFormatUnsupportedError,
    SessionLocation,
    SessionPersistenceCorruptionError,
    session_format_version_refusal,
)


def _unsupported(reason: str, location: SessionLocation | None) -> SessionFormatUnsupportedError:
    message = reason if location is None else f"{reason} (raw log: {location.path})"
    return SessionFormatUnsupportedError(message, location)


def assert_stored_id(session_id: SessionId | str, meta: SessionHeader) -> None:
    """Refuse stored metadata that is not bound to the requested session id."""
    if meta["id"] != session_id:
        raise ValueError(
            f'stored session identity mismatch: requested "{session_id}", header contains '
            f'"{meta["id"]}"'
        )


def assert_version(meta: Mapping[str, Any], location: SessionLocation | None = None) -> None:
    """Refuse a header that has not been restored to the current logical format."""
    if meta["version"] != SESSION_FORMAT_VERSION:
        raise _unsupported(session_format_version_refusal(meta["id"], meta["version"]), location)


def validate_stored_events(
    meta: SessionHeader,
    events: MutableSequence[SessionEvent],
    location: SessionLocation | None = None,
) -> MutableSequence[SessionEvent]:
    """Validate a decoded event list in place: adopt (validate and freeze) each record.

    Unknown event types refuse unless marked ``ignorable: true``; retired pre-release
    shapes refuse too. Performs no migration.
    """
    for event in events:
        kind = event["type"]
        if kind not in KNOWN_SESSION_EVENT_TYPES and event.get("ignorable") is not True:
            raise _unsupported(
                f'session "{meta["id"]}" contains event type "{kind}" (seq {event["seq"]}) '
                "unknown to this harness and not marked ignorable; refusing to interpret the "
                "log — it was likely written by a newer harness",
                location,
            )
        if kind == "request/header":
            data = event["data"]
            if isinstance(data, Mapping) and data.get("reason") == "fallback":
                raise _unsupported(
                    f'session "{meta["id"]}" contains a request/header event '
                    f'(seq {event["seq"]}) with the unsupported legacy reason "fallback"; '
                    "refusing to interpret the log — it was written by a retired "
                    "pre-release harness",
                    location,
                )
    try:
        for index, event in enumerate(events):
            events[index] = adopt_session_event(event)
    except SessionFormatUnsupportedError:
        raise
    except Exception as error:
        raise SessionPersistenceCorruptionError(
            f'stored session "{meta["id"]}" failed validation: {error}'
        ) from error
    return events


def materialize_create_header(header: SessionHeader) -> SessionHeader:
    """Validate and deep-snapshot a header passed to ``create`` in one traversal."""
    snapshot = snapshot_json_value(header, freeze=True)
    if snapshot is None:
        raise TypeError("session metadata must be losslessly JSON-serializable")
    created = snapshot.get("createdAt")
    if not is_safe_integer(created) or isinstance(created, bool) or created < 0:  # type: ignore[operator]
        raise TypeError("session metadata createdAt must be a non-negative safe integer")
    return snapshot


def materialize_append_batch(events: Sequence[SessionEvent]) -> Sequence[SessionEvent]:
    """Validate and deep-snapshot one append batch in a single traversal."""
    batch = snapshot_json_value(list(events), freeze=True)
    if batch is None:
        raise TypeError(
            "session event batch is not losslessly JSON-serializable because it contains "
            "non-JSON-serializable data"
        )
    return batch


def assert_contiguous(
    session_id: SessionId | str, events: Sequence[SessionEvent], cursor: int
) -> None:
    """Refuse a batch that does not contiguously continue the stored log."""
    for index, event in enumerate(events):
        if event["seq"] != cursor + index:
            raise ValueError(
                f'append seq mismatch for "{session_id}": expected {cursor + index} at index '
                f"{index}, got {event['seq']}"
            )
