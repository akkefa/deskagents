"""Upstream: packages/session/session-format-v3-to-v4/src/facts.ts

Only ``catalogFact`` is ported; the historical child evidence helpers belong to the
v3->v4 migration and the children catalog.
"""

from __future__ import annotations

from collections.abc import Mapping

from deskagents.session.format import (
    SessionFormatError,
    SessionFormatJsonObject,
    session_format_count,
)

from ._js import UNDEFINED, prop, strict_equal

_MODES = ("continuable", "one-shot", "external", "unknown")


def catalog_fact(value: object, subject: str = "subagent/catalog") -> SessionFormatJsonObject:
    """Validate the historical fields used for catalog membership.

    Extensions are not interpreted. Returns the validated catalog payload.
    """
    if not isinstance(value, Mapping) or not _supported(value):
        raise SessionFormatError(f"{subject} requires a supported versioned catalog fact")
    session_format_count(prop(value, "childCreatedAt"), "catalog child creation time")
    return value


def _supported(value: Mapping[str, object]) -> bool:
    version = prop(value, "version")
    mode = prop(value, "mode")
    label = prop(value, "label")
    return not (
        not any(strict_equal(version, known) for known in (0, 1, 2))
        or not isinstance(prop(value, "childId"), str)
        or not (isinstance(mode, str) and mode in _MODES)
        or (strict_equal(version, 2) != (mode == "external"))
        or (strict_equal(version, 0) and mode == "unknown")
        or (mode == "continuable" and not isinstance(label, str))
        or (label is not UNDEFINED and not isinstance(label, str))
    )
