"""Upstream: packages/session/session-format-v1-to-v2/src/validation.ts

Only the header validation and the record helpers the released-v2 codec needs are
ported. ``restoreReleasedV2Artifact`` and the physical-artifact check depend on the
v0-to-v1 relationship rules and follow with the v1-to-v2 migration.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from deskagents.session.format import (
    SessionFormatError,
    SessionFormatJsonObject,
    session_format_count,
)

HEADER_REQUIRED = ("version", "id", "createdAt", "isSeeded", "delegationDepth")
HEADER_OPTIONAL = ("cwd", "parentSession", "origin", "agentPreset")


def is_json_number(value: object, expected: int) -> bool:
    """JavaScript ``value === expected`` for a number: never true for a boolean."""
    return isinstance(value, int | float) and not isinstance(value, bool) and value == expected


def released_v2_record(value: object, label: str) -> SessionFormatJsonObject:
    """Require one released-v2 value to be a JSON object."""
    if not isinstance(value, Mapping):
        raise SessionFormatError(f"{label} must be an object")
    return value


def assert_released_v2_keys(
    value: Mapping[str, object],
    required: Iterable[str],
    optional: Iterable[str],
    label: str,
) -> None:
    """Require one released-v2 object to contain exactly the admitted keys."""
    required = tuple(required)
    allowed = {*required, *optional}
    missing = next((key for key in required if key not in value), None)
    if missing is not None:
        raise SessionFormatError(f"{label} lacks required field {missing}")
    unexpected = next((key for key in value if key not in allowed), None)
    if unexpected is not None:
        raise SessionFormatError(f"{label} has unexpected field {unexpected}")


def assert_released_v2_header(header: object) -> None:
    """Validate the exact logical header written by released v2.

    Raises :class:`SessionFormatError` when the header is not an exact released-v2 value.
    """
    record = released_v2_record(header, "format v2 header")
    assert_released_v2_keys(record, HEADER_REQUIRED, HEADER_OPTIONAL, "format v2 header")
    if not is_json_number(record["version"], 2):
        raise SessionFormatError("expected format v2 header")
    if not isinstance(record["id"], str):
        raise SessionFormatError("format v2 header id must be a string")
    session_format_count(record["createdAt"], "format v2 header createdAt")
    session_format_count(record["delegationDepth"], "format v2 header delegationDepth")
    if not isinstance(record["isSeeded"], bool):
        raise SessionFormatError("format v2 header isSeeded must be boolean")
    if "cwd" in record and (not isinstance(record["cwd"], str) or not os.path.isabs(record["cwd"])):
        raise SessionFormatError("format v2 header cwd must be absolute")
    for key in ("parentSession", "agentPreset"):
        if key in record and not isinstance(record[key], str):
            raise SessionFormatError(f"format v2 header {key} must be a string")
    if "origin" in record and record["origin"] != "subagent":
        raise SessionFormatError('format v2 header origin must be "subagent"')
