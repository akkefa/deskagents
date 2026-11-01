"""Canonical raw log basename shared by every generation-addressed Session artifact.

Upstream: packages/session/session-format/src/filename.ts
"""

from __future__ import annotations

import re

from deskagents.util.values import is_safe_integer

from .json import session_format_version

_CANONICAL_LOG_FILENAME = re.compile(r"session(?:\.v([1-9][0-9]*))?\.jsonl")


def session_format_log_filename(version: int) -> str:
    """Name the raw JSONL log of one immutable Session format generation.

    Version zero keeps ``session.jsonl``; every later generation carries a lowercase
    numeric ``.vN`` component before ``.jsonl``. No compression suffix is included.
    """
    generation = session_format_version(version, "Session log generation version")
    return "session.jsonl" if generation == 0 else f"session.v{generation}.jsonl"


def parse_session_format_log_filename(filename: str) -> int | None:
    """Read the generation a canonical raw JSONL basename names, or ``None``.

    Temporary, uppercase, leading-zero, ``.v0`` and compression-suffixed names are not
    canonical.
    """
    match = _CANONICAL_LOG_FILENAME.fullmatch(filename)
    if match is None:
        return None
    if match[1] is None:
        return 0
    version = int(match[1])
    return version if is_safe_integer(version) else None
