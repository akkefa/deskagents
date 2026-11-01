"""Upstream: packages/session/session-format-catalog/src/generated.ts

Upstream generates this assembly with ``scripts/gen-session-format-catalog.ts``. The
direct imports make historical readability independent of mounted plugins. Formats
v0-v3 are not ported yet, so their codecs and migrations refuse every artifact as
unsupported (see :mod:`.unported`).
"""

from __future__ import annotations

from deskagents.core.session import KNOWN_SESSION_EVENT_TYPES
from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatCatalog,
    SessionFormatCatalogOptions,
    SessionFormatHeader,
    create_session_format_catalog,
)
from deskagents.session.format_v3_to_v4 import (
    assert_released_v4_header,
    released_v4_session_format_codec,
    restore_released_v4_artifact,
)

from .current import (
    validate_installed_current_session_artifact,
    validate_installed_current_session_header,
)
from .unported import unported_codec, unported_migration


def _restore_current(artifact: SessionFormatArtifact) -> SessionFormatArtifact:
    restored = restore_released_v4_artifact(artifact, KNOWN_SESSION_EVENT_TYPES)
    validate_installed_current_session_artifact(restored)
    return restored


def _restore_transformed_current(artifact: SessionFormatArtifact) -> SessionFormatArtifact:
    return restore_released_v4_artifact(artifact, KNOWN_SESSION_EVENT_TYPES)


def _restore_current_header(header: SessionFormatHeader) -> SessionFormatHeader:
    assert_released_v4_header(header)
    validate_installed_current_session_header(header)
    return header


session_format_catalog_options = SessionFormatCatalogOptions(
    current_version=4,
    codecs=(
        unported_codec(0),
        unported_codec(1),
        unported_codec(2),
        unported_codec(3),
        released_v4_session_format_codec,
    ),
    current_encoder=released_v4_session_format_codec,
    migrations=(
        unported_migration("@deepseek-ai/dsh-session-format-v0-to-v1", 0),
        unported_migration("@deepseek-ai/dsh-session-format-v1-to-v2", 1),
        unported_migration("@deepseek-ai/dsh-session-format-v2-to-v3", 2),
        unported_migration("@deepseek-ai/dsh-session-format-v3-to-v4", 3),
    ),
    restore_current=_restore_current,
    restore_transformed_current=_restore_transformed_current,
    restore_current_header=_restore_current_header,
)
"""Static assembly shared by current reads and (later) parent-specific historical restoration."""

session_format_catalog: SessionFormatCatalog = create_session_format_catalog(
    session_format_catalog_options
)
"""Physical codec dispatch and complete adjacent chain, independent of mounted plugins."""
