"""Upstream: packages/session/session-format-v3-to-v4/src/codec.ts

V4 framing with native tool-role admission and released physical rows.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from deskagents.session.format import (
    SessionFormatArtifactDecoder,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatHeader,
    SessionFormatJsonObject,
    SessionFormatMigrationContext,
    SessionFormatRecovery,
    is_session_format_json_object,
)
from deskagents.session.format_v1_to_v2 import (
    ReleasedSessionFormatCodec,
    released_v2_session_format_codec,
)
from deskagents.util.values import deep_freeze

from ._js import prop, strict_equal
from .developer import assert_v4_developer_data
from .fork_result import assert_v4_fork_result
from .message_sources import assert_v4_source_row_admission
from .retired_syntax import assert_v4_retired_syntax
from .system_message import assert_v4_system_message_fields
from .tool_role import assert_v4_tool_result_message
from .validation import assert_released_v4_header

__all__ = ["assert_v4_row_admission", "released_v4_session_format_codec"]

_V2 = released_v2_session_format_codec


def _physical_v2(value: object) -> SessionFormatHeader:
    if not isinstance(value, Mapping) or not strict_equal(prop(value, "version"), 4):
        raise SessionFormatError("expected format v4 physical header")
    return {**value, "version": 2}


def _decode_header(value: object) -> SessionFormatHeader:
    return deep_freeze({**_V2.decode_header(_physical_v2(value)), "version": 4})


class _V4Decoder:
    """The released v2 row decoder behind native V4 row admission."""

    def __init__(self, value: object, recovery: SessionFormatRecovery) -> None:
        self._decoder = _V2.create_decoder(_physical_v2(value), recovery)
        self.header: SessionFormatHeader = deep_freeze({**self._decoder.header, "version": 4})

    def decode_row(self, row_value: object, context: SessionFormatMigrationContext) -> None:
        assert_v4_row_admission(row_value)
        self._decoder.decode_row(row_value, context)

    def finish(self, context: SessionFormatMigrationContext) -> int:
        return self._decoder.finish(context)


def _create_decoder(value: object, recovery: SessionFormatRecovery) -> SessionFormatArtifactDecoder:
    return _V4Decoder(value, recovery)


def _encode_header(
    header: SessionFormatHeader, inherited_event_count: int
) -> SessionFormatJsonObject:
    assert_released_v4_header(header)
    return {**_V2.encode_header({**header, "version": 2}, inherited_event_count), "version": 4}


def _encode_event(event: SessionFormatEvent) -> SessionFormatJsonObject:
    if prop(event, "type") == "developer/message" and prop(event, "ignorable") is True:
        assert_v4_developer_data(event)
        assert_v4_retired_syntax(event)
    assert_v4_row_admission(event)
    return _V2.encode_event(event)


def assert_v4_row_admission(row: object, known_event_types: Collection[str] | None = None) -> None:
    """Apply native V4 admission before a scanner discards a recoverable suffix.

    Ignorable developer payloads require reader vocabulary; physical decoding defers
    them. ``row`` is a parsed physical row before framing and source-event range
    decoding; ``known_event_types`` is supplied by native readers before tail recovery.
    """
    if is_session_format_json_object(row):
        assert isinstance(row, Mapping)
        if (
            prop(row, "type") == "developer/message"
            and prop(row, "ignorable") is True
            and (known_event_types is None or "developer/message" not in known_event_types)
        ):
            return
        assert_v4_developer_data(row)
    assert_v4_source_row_admission(row)
    assert_v4_retired_syntax(row)
    assert_v4_system_message_fields(row)
    if not isinstance(row, Mapping) or prop(row, "type") != "tool/result":
        return
    assert_v4_tool_result_message(row)
    assert_v4_fork_result(row)


released_v4_session_format_codec = ReleasedSessionFormatCodec(
    version=4,
    decode_header=_decode_header,
    create_decoder=_create_decoder,
    encode_header=_encode_header,
    encode_event=_encode_event,
)
"""V4 physical encoder and decoder.

They keep the released row framing while validating the native tool-role message
directly.
"""
