"""Upstream: packages/session/session-format/src/json.ts"""

from __future__ import annotations

from collections.abc import Mapping

from deskagents.util.values import JsonValue, is_safe_integer, snapshot_json_value

from .errors import SessionFormatError
from .types import SessionFormatHeader


def is_session_format_json_object(value: object) -> bool:
    """Test whether a value is a JSON object (a mapping, not an array)."""
    return isinstance(value, Mapping)


def _negative_zero(value: object) -> bool:
    return isinstance(value, float) and value == 0.0 and str(value).startswith("-")


def session_format_count(value: object, label: str) -> int:
    """Require a non-negative safe integer without the JSON-unstable negative zero."""
    if not is_safe_integer(value) or _negative_zero(value) or value < 0:  # type: ignore[operator]
        raise SessionFormatError(f"{label} must be a non-negative safe integer")
    return int(value)  # type: ignore[call-overload,no-any-return]


def session_format_safe_integer(value: object, label: str) -> int:
    """Require a safe integer without the JSON-unstable negative zero."""
    if not is_safe_integer(value) or _negative_zero(value):
        raise SessionFormatError(f"{label} must be a safe integer")
    return int(value)  # type: ignore[call-overload,no-any-return]


def session_format_version(value: object, label: str = "Session format version") -> int:
    """Require a non-negative integral format version."""
    return session_format_count(value, label)


def inspect_session_format_version(header_value: object) -> int:
    """Read only the version required for directional dispatch."""
    if not isinstance(header_value, Mapping):
        raise SessionFormatError("Session header must be a JSON object")
    return session_format_version(header_value.get("version"))


def snapshot_session_format_json(value: object, label: str = "Session value") -> JsonValue:
    """Detach and deeply freeze a caller-supplied lossless JSON value."""
    if value is None:
        return None
    snapshot = snapshot_json_value(value, freeze=True)
    if snapshot is None:
        raise SessionFormatError(f"{label} is not lossless JSON")
    return snapshot  # type: ignore[return-value]


def snapshot_session_format_header(
    header: object, label: str = "Session header"
) -> SessionFormatHeader:
    """Snapshot one logical header without inspecting an event body."""
    snapshot = snapshot_session_format_json(header, label)
    if not isinstance(snapshot, Mapping):
        raise SessionFormatError(f"{label} must be a JSON object")
    inspect_session_format_version(snapshot)
    if not isinstance(snapshot.get("id"), str):
        raise SessionFormatError(f"{label} id must be a string")
    session_format_count(snapshot.get("createdAt"), f"{label} createdAt")
    if not isinstance(snapshot.get("isSeeded"), bool):
        raise SessionFormatError(f"{label} isSeeded must be a boolean")
    session_format_count(snapshot.get("delegationDepth"), f"{label} delegationDepth")
    return snapshot
