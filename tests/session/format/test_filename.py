"""Ported from packages/session/session-format/tests/filename.spec.ts."""

from __future__ import annotations

import pytest

from deskagents.session.format import (
    SessionFormatError,
    parse_session_format_log_filename,
    session_format_log_filename,
)

SPEC = "packages/session/session-format/tests/filename.spec.ts"
GROUP = "canonical Session log filenames"


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > names version zero without a generation component and later "
    "generations with .vN"
)
def test_names_generations() -> None:
    assert session_format_log_filename(0) == "session.jsonl"
    assert session_format_log_filename(1) == "session.v1.jsonl"
    assert session_format_log_filename(27) == "session.v27.jsonl"
    for invalid in [-1, -0.0, 1.5, 2**53]:
        with pytest.raises(SessionFormatError, match="non-negative safe integer"):
            session_format_log_filename(invalid)  # type: ignore[arg-type]


@pytest.mark.upstream(f"{SPEC}::{GROUP} > parses only canonical raw generation names")
def test_parses_only_canonical_names() -> None:
    assert parse_session_format_log_filename("session.jsonl") == 0
    assert parse_session_format_log_filename("session.v1.jsonl") == 1
    assert parse_session_format_log_filename("session.v27.jsonl") == 27
    assert parse_session_format_log_filename("session.v9007199254740992.jsonl") is None
    for name in [
        "session.v0.jsonl",
        "session.v01.jsonl",
        "session.V1.jsonl",
        "session.v1.backup.jsonl",
        "session.migration.deadbeef.tmp.jsonl",
        "session.v1.jsonl.zstd",
        "session.1.jsonl",
    ]:
        assert parse_session_format_log_filename(name) is None
