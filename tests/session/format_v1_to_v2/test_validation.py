"""Ported from packages/session/session-format-v1-to-v2/tests/validation.spec.ts.

Only "released v2 header validation" is ported. The artifact, stream, seed and
surface cases exercise ``restoreReleasedV2Artifact`` and the released-v2 payload
dispositions, which are not ported until Python reads v2 sessions.
"""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.session.format import SessionFormatError
from deskagents.session.format_v1_to_v2 import assert_released_v2_header

SPEC = "packages/session/session-format-v1-to-v2/tests/validation.spec.ts"
GROUP = "released v2 header validation"
MINIMAL: dict[str, Any] = {
    "version": 2,
    "id": "validation",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}


@pytest.mark.upstream(f"{SPEC}::{GROUP} > accepts minimal and complete logical headers")
def test_accepts_minimal_and_complete_headers() -> None:
    assert_released_v2_header(MINIMAL)
    assert_released_v2_header(
        {
            **MINIMAL,
            "cwd": "/work",
            "parentSession": "parent",
            "origin": "subagent",
            "agentPreset": "default",
        }
    )


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects exact logical header drift: %s")
@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(None, "must be an object", id="null"),
        pytest.param([], "must be an object", id="array"),
        pytest.param("header", "must be an object", id="scalar"),
        pytest.param(
            {k: v for k, v in MINIMAL.items() if k != "id"},
            "lacks required field id",
            id="missing field",
        ),
        pytest.param({**MINIMAL, "extra": True}, "unexpected field extra", id="unexpected field"),
        pytest.param({**MINIMAL, "version": 1}, "expected format v2", id="wrong version"),
        pytest.param({**MINIMAL, "id": 1}, "id must be a string", id="non-string id"),
        pytest.param({**MINIMAL, "createdAt": -1}, "createdAt", id="invalid createdAt"),
        pytest.param({**MINIMAL, "delegationDepth": -1}, "delegationDepth", id="invalid depth"),
        pytest.param({**MINIMAL, "isSeeded": 1}, "isSeeded must be boolean", id="invalid isSeeded"),
        pytest.param({**MINIMAL, "cwd": 1}, "cwd must be absolute", id="non-string cwd"),
        pytest.param({**MINIMAL, "cwd": "relative"}, "cwd must be absolute", id="relative cwd"),
        pytest.param(
            {**MINIMAL, "parentSession": 1}, "parentSession must be a string", id="invalid parent"
        ),
        pytest.param(
            {**MINIMAL, "agentPreset": 1}, "agentPreset must be a string", id="invalid preset"
        ),
        pytest.param(
            {**MINIMAL, "origin": "parent"}, 'origin must be "subagent"', id="invalid origin"
        ),
    ],
)
def test_rejects_header_drift(value: object, message: str) -> None:
    with pytest.raises(SessionFormatError, match=message):
        assert_released_v2_header(value)


def test_boolean_version_is_not_two() -> None:
    """Python-specific: ``True`` and ``2.0`` are judged by JavaScript ``===``."""
    assert_released_v2_header({**MINIMAL, "version": 2.0})
    with pytest.raises(SessionFormatError, match="expected format v2"):
        assert_released_v2_header({**MINIMAL, "version": True})
