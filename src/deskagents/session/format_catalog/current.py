"""Upstream: packages/session/session-format-catalog/src/current.ts

Current installed Session validation used after vocabulary-aware format restoration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    Session,
    SessionEvent,
    SessionHeader,
)
from deskagents.session.format import SessionFormatArtifact, SessionFormatHeader
from deskagents.util.values import format_json_number

from .message_projections import current_session_message_projections


def _version_text(value: object) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return format_json_number(value)
    return str(value)


def _check_version(header: SessionFormatHeader) -> None:
    version = header.get("version")
    if not (
        isinstance(version, int | float)
        and not isinstance(version, bool)
        and version == SESSION_FORMAT_VERSION
    ):
        raise ValueError(
            f"installed Session format is v{SESSION_FORMAT_VERSION}, got v{_version_text(version)}"
        )


def validate_installed_current_session_header(header: SessionFormatHeader) -> None:
    """Validate current logical metadata through the installed Session package."""
    _check_version(header)
    Session.from_restore(cast(str, header["id"]), [], cast(SessionHeader, header), 0, "detached")


def validate_installed_current_session_artifact(artifact: SessionFormatArtifact) -> None:
    """Validate current header, event envelopes, messages, surface operations and seed cut."""
    _check_version(artifact.header)
    Session.from_restore(
        cast(str, artifact.header["id"]),
        cast(Sequence[SessionEvent], artifact.events),
        cast(SessionHeader, artifact.header),
        artifact.inherited_event_count,
        "detached",
        current_session_message_projections,
    )
