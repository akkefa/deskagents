"""Refusing stand-ins for the released v0-v3 codecs and migrations (Python only).

Upstream registers ``releasedV0SessionFormatCodec`` .. ``releasedV3SessionFormatCodec``
and the four adjacent migrations. Until they are ported, every stored v0-v3 artifact
is refused with :class:`SessionFormatUnsupportedMigrationError`, so
``read_header`` classifies it as ``unsupported`` (not ``malformed``) and
``create_restore`` raises.
"""

from __future__ import annotations

from typing import NoReturn

from deskagents.session.format import (
    SessionFormatCodec,
    SessionFormatMigration,
    SessionFormatUnsupportedMigrationError,
)


def _refuse(version: int) -> NoReturn:
    raise SessionFormatUnsupportedMigrationError(
        f"Session format v{version} is not supported by this Python build yet; "
        "only v4 sessions can be read"
    )


def unported_codec(version: int) -> SessionFormatCodec:
    """A codec whose header decode and decoder creation refuse format ``version``."""

    def decode_header(value: object) -> NoReturn:
        _refuse(version)

    def create_decoder(value: object, recovery: object) -> NoReturn:
        _refuse(version)

    return SessionFormatCodec(
        version=version, decode_header=decode_header, create_decoder=create_decoder
    )


def unported_migration(name: str, from_version: int) -> SessionFormatMigration:
    """An adjacent migration that refuses every header and body from ``from_version``."""

    def refuse(value: object) -> NoReturn:
        _refuse(from_version)

    return SessionFormatMigration(
        name=name,
        from_version=from_version,
        to_version=from_version + 1,
        migrate_header=refuse,
        create_stage=refuse,
        validate_target_header=refuse,
    )
