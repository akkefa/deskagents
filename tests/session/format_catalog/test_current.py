"""Ported from packages/session/session-format-catalog/tests/current.spec.ts."""

from __future__ import annotations

from typing import Any

import pytest

from deskagents.core.session import SESSION_FORMAT_VERSION
from deskagents.session.format import SessionFormatArtifact
from deskagents.session.format_catalog import (
    validate_installed_current_session_artifact,
    validate_installed_current_session_header,
)

SPEC = "packages/session/session-format-catalog/tests/current.spec.ts"
GROUP = "installed current Session restoration"
HEADER: dict[str, Any] = {
    "version": SESSION_FORMAT_VERSION,
    "id": "installed-current",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects version skew before entering current Session validation"
)
def test_rejects_version_skew() -> None:
    message = f"installed Session format is v{SESSION_FORMAT_VERSION}, got v0"
    with pytest.raises(ValueError, match=message):
        validate_installed_current_session_header({**HEADER, "version": 0})
    artifact = SessionFormatArtifact({**HEADER, "version": 0}, 0, [])
    with pytest.raises(ValueError, match=message):
        validate_installed_current_session_artifact(artifact)


def artifact(reason: str, starts_series: bool | None = None) -> SessionFormatArtifact:
    data: dict[str, Any] = {
        "header": {"config": {"provider": "mock", "model": "mock"}},
        "reason": reason,
    }
    if starts_series is not None:
        data["startsSeries"] = starts_series
    return SessionFormatArtifact(
        {**HEADER},
        0,
        [
            {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
            {"type": "request/header", "seq": 1, "time": 2, "data": data},
        ],
    )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > accepts only current request-header reasons and the true "
    "starts-series marker"
)
def test_accepts_only_current_request_header_reasons() -> None:
    with pytest.raises(ValueError, match=r"request/header.*reason"):
        validate_installed_current_session_artifact(artifact("fallback"))
    with pytest.raises(ValueError, match="startsSeries"):
        validate_installed_current_session_artifact(artifact("initial", False))
    validate_installed_current_session_artifact(artifact("series", True))
