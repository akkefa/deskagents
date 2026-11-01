"""Upstream: packages/session/session-format-v3-to-v4/src/system-message.ts

Native system-message fields checked before physical recovery can discard a row.
"""

from __future__ import annotations

from collections.abc import Mapping

from deskagents.session.format import SessionFormatError, session_format_count

from ._js import UNDEFINED, is_array, prop

_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")


def _record(value: object, subject: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SessionFormatError(f"format v4 {subject} requires an object")
    return value


def _string(value: object, subject: str, nonempty: bool = False) -> None:
    if not isinstance(value, str) or (nonempty and value == ""):
        suffix = " with content" if nonempty else ""
        raise SessionFormatError(f"format v4 {subject} requires a string{suffix}")


def _positive(value: object, subject: str) -> None:
    if session_format_count(value, subject) == 0:
        raise SessionFormatError(f"format v4 {subject} must be positive")


def _image(value: object) -> None:
    attachment = _record(value, "system image attachment")
    _string(prop(attachment, "attachmentId"), "system image attachmentId", True)
    media_type = prop(attachment, "mediaType")
    if not (isinstance(media_type, str) and media_type in _MEDIA_TYPES):
        raise SessionFormatError("format v4 system image requires a supported mediaType")
    session_format_count(prop(attachment, "bytes"), "system image bytes")
    _positive(prop(attachment, "width"), "system image width")
    _positive(prop(attachment, "height"), "system image height")
    if prop(attachment, "name") is not UNDEFINED:
        _string(attachment["name"], "system image name")
    if prop(attachment, "originalDimensions") is not UNDEFINED:
        original = _record(attachment["originalDimensions"], "system image originalDimensions")
        _positive(prop(original, "width"), "system image original width")
        _positive(prop(original, "height"), "system image original height")


def assert_v4_system_message_fields(row: object) -> None:
    """Validate system fields independently of historical message representations.

    Additional JSON fields are preserved. ``row`` is a complete logical event or a
    parsed physical row before corruption recovery.
    """
    if not isinstance(row, Mapping) or prop(row, "type") != "system/message":
        return
    data = _record(prop(row, "data"), "system/message data")
    _positive(prop(data, "turn"), "system/message turn")
    _positive(prop(data, "step"), "system/message step")
    message = _record(prop(data, "message"), "system message")
    _string(prop(message, "id"), "system message id", True)
    if prop(message, "role") != "system":
        raise SessionFormatError("format v4 system message requires system role")
    content = prop(message, "content")
    if not is_array(content):
        raise SessionFormatError("format v4 system message requires content array")
    assert isinstance(content, list | tuple)
    for value in content:
        block = _record(value, "system content block")
        _string(prop(block, "type"), "system content type", True)
        block_type = block["type"]
        if block_type in ("text", "reasoning"):
            _string(prop(block, "text"), "system content text")
        elif block_type == "tool-call":
            _string(prop(block, "id"), "system tool-call id", True)
            _string(prop(block, "name"), "system tool-call name", True)
            _string(prop(block, "arguments"), "system tool-call arguments")
        elif block_type == "image":
            _image(prop(block, "attachment"))
        elif block_type == "tool-result":
            raise SessionFormatError(
                "format v4 system content rejects retired tool-result wrappers"
            )
        # ContentBlockMap is merge-extensible; only known block fields are interpreted here.
