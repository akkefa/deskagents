"""Ported from packages/core/session/tests/preparation.spec.ts."""

from __future__ import annotations

import pytest

from deskagents.core.session import Session, SessionPreparation, session_id

SPEC = "packages/core/session/tests/preparation.spec.ts"
GROUP = "SessionPreparation"


class Release:
    """Call recorder standing in for ``vi.fn()``."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > retains the exact Session and releases provider state only once"
)
def test_retains_session_and_releases_once() -> None:
    session = Session.create(session_id("prepared"))
    release = Release()
    with SessionPreparation.create(session, release=release) as preparation:
        assert preparation.session is session
        preparation.dispose()
        assert release.calls == 1
    assert release.calls == 1


@pytest.mark.upstream(f"{SPEC}::{GROUP} > disposes a preparation without provider state")
def test_disposes_without_provider_state() -> None:
    with SessionPreparation.create(Session.create(session_id("unretained"))) as preparation:
        preparation.dispose()
