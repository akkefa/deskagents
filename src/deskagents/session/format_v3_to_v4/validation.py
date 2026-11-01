"""Upstream: packages/session/session-format-v3-to-v4/src/validation.ts

Native V4 metadata and generation-owned relationship validation.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatUnsupportedMigrationError,
    session_format_count,
)

from ._js import UNDEFINED, greater_or_equal, has, json_text, prop, strict_equal
from .developer import assert_v4_developer_data
from .facts import catalog_fact
from .fork_result import assert_v4_fork_result
from .message_sources import assert_v4_message_sources
from .relationships import assert_v4_lifecycle_relationships
from .retired_syntax import assert_v4_retired_syntax
from .system_message import assert_v4_system_message_fields
from .tool_role import assert_v4_tool_result_message

_HEADER_REQUIRED = ("version", "id", "createdAt", "isSeeded", "delegationDepth")
_HEADER_ALLOWED = frozenset((*_HEADER_REQUIRED, "cwd", "parentSession", "origin", "agentPreset"))


def assert_released_v4_header(header: object) -> None:
    """Validate the exact native V4 logical header.

    ``header`` is a decoded or otherwise untrusted V4 Session header candidate.
    """
    if not isinstance(header, Mapping) or not strict_equal(prop(header, "version"), 4):
        raise SessionFormatError("expected format v4 header")
    missing = next((key for key in _HEADER_REQUIRED if key not in header), None)
    unexpected = next((key for key in header if key not in _HEADER_ALLOWED), None)
    if missing is not None:
        raise SessionFormatError(f"format v4 header lacks required field {missing}")
    if unexpected is not None:
        raise SessionFormatError(f"format v4 header has unexpected field {unexpected}")
    if not isinstance(header["id"], str):
        raise SessionFormatError("format v4 header id must be a string")
    session_format_count(header["createdAt"], "format v4 header createdAt")
    session_format_count(header["delegationDepth"], "format v4 header delegationDepth")
    if not isinstance(header["isSeeded"], bool):
        raise SessionFormatError("format v4 header isSeeded must be boolean")
    if "cwd" in header and (not isinstance(header["cwd"], str) or not os.path.isabs(header["cwd"])):
        raise SessionFormatError("format v4 header cwd must be absolute")
    for key in ("parentSession", "agentPreset"):
        if key in header and not isinstance(header[key], str):
            raise SessionFormatError(f"format v4 header {key} must be a string")
    if "origin" in header and header["origin"] != "subagent":
        raise SessionFormatError('format v4 header origin must be "subagent"')


def restore_released_v4_artifact(
    artifact: SessionFormatArtifact, known_event_types: Collection[str]
) -> SessionFormatArtifact:
    """Validate V4 inheritance, vocabulary, native message admission, and lifecycle.

    Also checks compaction, tool, retry, title, command, catalog, and delivery
    ownership. Installed Session restoration owns common event envelopes and message
    acceptance. Returns the same validated artifact and event objects.
    """
    assert_released_v4_header(artifact.header)
    cut = session_format_count(artifact.inherited_event_count, "format v4 inherited event count")
    if cut > len(artifact.events):
        raise SessionFormatError("format v4 inherited event count exceeds its events")
    if not artifact.header["isSeeded"] and cut != 0:
        raise SessionFormatError("unseeded format v4 Session has inherited events")
    last_inherited_marker: int | None = None
    for index, event in enumerate(artifact.events):
        event_type = prop(event, "type")
        if not has(known_event_types, event_type) and prop(event, "ignorable") is not True:
            raise SessionFormatUnsupportedMigrationError(
                f"format v4 contains unknown event type {json_text(event_type)} at seq {index}"
            )
        if not strict_equal(prop(event, "seq"), index):
            raise SessionFormatError(f"format v4 event {index} is not dense")
        if not has(known_event_types, event_type):
            continue
        assert_v4_retired_syntax(event)
        assert_v4_system_message_fields(event)
        assert_v4_tool_result_message(event)
        assert_v4_fork_result(event)
        data = prop(event, "data")
        if (
            event_type == "session/end-seed"
            and isinstance(data, Mapping)
            and prop(data, "inherited") is True
        ):
            last_inherited_marker = index
    if artifact.header["isSeeded"] and last_inherited_marker != cut:
        raise SessionFormatError(
            "format v4 seeded header disagrees with its last inherited end-seed marker"
        )
    if not artifact.header["isSeeded"] and last_inherited_marker is not None:
        raise SessionFormatError("format v4 unseeded Session contains an inherited end-seed marker")
    assert_released_v4_relationships(artifact, known_event_types)
    return artifact


def validate_delivery_accepted(event: SessionFormatEvent, current_version: int) -> str | None:
    """Validate delivery generation and active-generation coordinates.

    ``current_version`` (3 or 4) is the generation whose watermark coordinates are
    active. Returns the active delivery's nonempty Session id, or ``None`` for other
    events and generations.
    """
    if prop(event, "type") != "session-log-deepseek/delivery-accepted":
        return None
    data = prop(event, "data")
    if not isinstance(data, Mapping):
        raise SessionFormatError("delivery-accepted data must be an object")
    stored = prop(data, "sessionFormatVersion")
    version = session_format_count(
        0 if stored is UNDEFINED else stored, "delivery sessionFormatVersion"
    )
    if version != current_version:
        return None
    through_seq = session_format_count(prop(data, "throughSeq"), "delivery throughSeq")
    if greater_or_equal(through_seq, prop(event, "seq")):
        raise SessionFormatError("delivery throughSeq must precede its marker")
    session_id = prop(data, "sessionId")
    if not isinstance(session_id, str) or session_id == "":
        raise SessionFormatError("delivery requires a nonempty Session id")
    return session_id


def assert_released_v4_relationships(
    artifact: SessionFormatArtifact, known_event_types: Collection[str]
) -> None:
    """Validate native developer fields, message sources, lifecycle, catalog, and delivery.

    Event vocabulary and tail recovery are unchanged: the owning admission stage rejects
    unknown required events, and unknown ignorable records keep their uninterpreted
    payloads. ``artifact`` carries its final inherited cut.
    """
    ids: set[str] = set()
    cut = artifact.inherited_event_count
    for event in artifact.events:
        if not has(known_event_types, prop(event, "type")):
            continue
        assert_v4_developer_data(event)
        assert_v4_message_sources(event)
        delivery_id = validate_delivery_accepted(event, 4)
        seq = prop(event, "seq")
        if (
            delivery_id is not None
            and greater_or_equal(seq, cut)
            and delivery_id != artifact.header["id"]
        ):
            raise SessionFormatError("current-generation delivery marker names the wrong Session")
        if prop(event, "type") == "subagent/catalog" and greater_or_equal(seq, cut):
            fact = catalog_fact(prop(event, "data"))
            child_id = fact["childId"]
            assert isinstance(child_id, str)
            if child_id in ids:
                raise SessionFormatError(f"duplicate catalog child {child_id}")
            ids.add(child_id)
    assert_v4_lifecycle_relationships(artifact, known_event_types)
