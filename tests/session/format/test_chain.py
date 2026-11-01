"""Ported from packages/session/session-format/tests/chain.spec.ts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from support.session_format import Discard, Recorder, Run, Stage, identity_stage, migration, raise_

from deskagents.session.format import (
    SessionFormatChainOptions,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatEventCollector,
    SessionFormatHeader,
    SessionFormatMigration,
    SessionFormatMigrationStageInput,
    SessionFormatUnsupportedMigrationError,
    create_session_format_chain,
    define_session_format_migration,
)

SPEC = "packages/session/session-format/tests/chain.spec.ts"
GROUP = "Session format chain"

CURRENT_HEADER: SessionFormatHeader = {
    "version": 1,
    "id": "session-1",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}
EVENT: SessionFormatEvent = {"type": "turn/start", "seq": 0, "time": 2, "data": {"turn": 1}}
SOURCE = {**CURRENT_HEADER, "version": 0}


def chain(edge: SessionFormatMigration | None = None) -> Any:
    return create_session_format_chain(
        SessionFormatChainOptions(
            current_version=1,
            migrations=[edge or migration()],
            restore_current_header=lambda header: header,
        )
    )


def options(current_version: int, migrations: list[SessionFormatMigration]) -> Any:
    return SessionFormatChainOptions(
        current_version=current_version,
        migrations=migrations,
        restore_current_header=lambda header: header,
    )


@pytest.mark.upstream(f"{SPEC}::{GROUP} > bypasses adjacent stages for current input")
def test_bypasses_stages_for_current_input() -> None:
    create_stage = Recorder(lambda given: identity_stage(given.source_inherited_event_count))
    current = chain(migration(create_stage=create_stage))
    events = SessionFormatEventCollector()
    stream = current.create_stream(CURRENT_HEADER, 0, events)

    stream.emit_event(EVENT)

    assert stream.header == CURRENT_HEADER
    assert stream.finish() == 0
    assert events.values == [EVENT]
    assert create_stage.calls == []


@pytest.mark.upstream(f"{SPEC}::{GROUP} > composes one adjacent stage and its header conversion")
def test_composes_one_stage() -> None:
    current = chain()
    events = SessionFormatEventCollector()
    stream = current.create_stream(SOURCE, 0, events)

    stream.emit_event(EVENT)

    assert stream.header["version"] == 1
    assert stream.finish() == 0
    assert events.values == [EVENT]
    assert current.migrate_header(SOURCE)["version"] == 1


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects invalid declarations and incomplete chain construction"
)
def test_rejects_invalid_declarations() -> None:
    base = migration()
    with pytest.raises(SessionFormatError, match="name"):
        define_session_format_migration(replace(base, name=""))
    with pytest.raises(SessionFormatError, match="adjacent"):
        define_session_format_migration(replace(base, to_version=2))
    with pytest.raises(SessionFormatError, match="missing"):
        create_session_format_chain(options(1, []))
    with pytest.raises(SessionFormatError, match="duplicated"):
        create_session_format_chain(options(1, [base, base]))
    with pytest.raises(SessionFormatError, match="name .* duplicated"):
        create_session_format_chain(options(2, [base, replace(base, from_version=1, to_version=2)]))
    with pytest.raises(SessionFormatError, match="does not lead"):
        create_session_format_chain(
            options(1, [base, replace(base, name="@test/v1-to-v2", from_version=1, to_version=2)])
        )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > composes stages whose inherited cut is derived from the body"
)
def test_composes_body_derived_cut() -> None:
    first = migration(create_stage=lambda given: Stage(None, finish_fn=lambda context: 0))
    second = replace(
        migration(),
        name="@test/v1-to-v2",
        from_version=1,
        to_version=2,
        migrate_header=lambda header: {**header, "version": 2},
    )
    current = create_session_format_chain(options(2, [first, second]))
    assert current.create_stream(SOURCE, 0, Discard()).finish() == 0

    def second_stage(given: SessionFormatMigrationStageInput) -> Stage:
        assert given.source_kind == "transformed"
        return identity_stage(0)

    create_second_stage = Recorder(second_stage)
    complete = create_session_format_chain(
        options(2, [migration(), replace(second, create_stage=create_second_stage)])
    )
    stream = complete.create_stream(SOURCE, 0, Discard())
    assert stream.finish() == 0
    assert len(create_second_stage.calls) == 1


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects newer inputs and wrong header versions")
def test_rejects_newer_inputs_and_wrong_versions() -> None:
    current = chain(migration(migrate_header=lambda header: header))

    with pytest.raises(SessionFormatError, match="newer"):
        current.create_stream({**CURRENT_HEADER, "version": 2}, 0, Discard())
    with pytest.raises(SessionFormatError, match="non-negative"):
        current.create_stream({**CURRENT_HEADER, "version": -1}, 0, Discard())
    with pytest.raises(SessionFormatError, match="header returned v0"):
        current.create_stream(SOURCE, 0, Discard())
    with pytest.raises(SessionFormatError, match="header returned v0"):
        current.migrate_header(SOURCE)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > classifies stage failures as unsupported and preserves explicit refusals"
)
def test_classifies_stage_failures() -> None:
    policy_failure = RuntimeError("target relationship is invalid")
    failed = chain(migration(create_stage=lambda given: raise_(policy_failure)))
    with pytest.raises(SessionFormatUnsupportedMigrationError) as refusal:
        failed.create_stream(SOURCE, 0, Discard())
    assert "target relationship is invalid" in str(refusal.value)
    assert refusal.value.__cause__ is policy_failure

    already = SessionFormatUnsupportedMigrationError("explicit edge refusal")
    preserved = chain(migration(create_stage=lambda given: raise_(already)))
    with pytest.raises(SessionFormatUnsupportedMigrationError) as kept:
        preserved.create_stream(SOURCE, 0, Discard())
    assert kept.value is already

    finish_failure = RuntimeError("finish relationship is invalid")
    finishing = chain(
        migration(create_stage=lambda given: Stage(0, finish_fn=lambda c: raise_(finish_failure)))
    ).create_stream(SOURCE, 0, Discard())
    with pytest.raises(SessionFormatUnsupportedMigrationError) as finish_refusal:
        finishing.finish()
    assert finish_refusal.value.__cause__ is finish_failure

    event_failure = RuntimeError("event relationship is invalid")
    event_failing = chain(
        migration(
            create_stage=lambda given: Stage(
                0, transform_event_fn=lambda e, c: raise_(event_failure)
            )
        )
    ).create_stream(SOURCE, 0, Discard())
    with pytest.raises(SessionFormatError, match="event relationship is invalid"):
        event_failing.emit_event(EVENT)

    run_failure = RuntimeError("run relationship is invalid")
    run_failing = chain(
        migration(
            create_stage=lambda given: Stage(0, transform_run_fn=lambda r, c: raise_(run_failure))
        )
    ).create_stream(SOURCE, 0, Discard())
    with pytest.raises(SessionFormatUnsupportedMigrationError) as run_refusal:
        run_failing.emit_run(Run("test-run", 0, 1, [EVENT]))
    assert run_refusal.value.__cause__ is run_failure

    non_error = chain(
        migration(create_stage=lambda given: raise_(ValueError("non-Error stage refusal")))
    )
    with pytest.raises(SessionFormatError, match="non-Error stage refusal"):
        non_error.create_stream(SOURCE, 0, Discard())


@pytest.mark.upstream(f"{SPEC}::{GROUP} > collects compact runs as expanded events")
def test_collects_compact_runs() -> None:
    collector = SessionFormatEventCollector()
    collector.emit_run(Run("test-run", 0, 1, [EVENT]))
    assert collector.values == [EVENT]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > validates every adjacent target header and the final current header"
)
def test_validates_target_and_current_headers() -> None:
    def check_marker(header: SessionFormatHeader) -> None:
        if header.get("targetMarker") is not True:
            raise RuntimeError("target header lacks marker")

    def check_id(header: SessionFormatHeader) -> SessionFormatHeader:
        if not isinstance(header.get("id"), str):
            raise RuntimeError("current header lacks id")
        return header

    validate_target_header = Recorder(check_marker)
    current = create_session_format_chain(
        SessionFormatChainOptions(
            current_version=1,
            migrations=[migration(validate_target_header=validate_target_header)],
            restore_current_header=check_id,
        )
    )
    with pytest.raises(SessionFormatUnsupportedMigrationError, match="target header lacks marker"):
        current.migrate_header(SOURCE)
    assert len(validate_target_header.calls) == 1

    rejecting = create_session_format_chain(
        SessionFormatChainOptions(
            current_version=1,
            migrations=[
                migration(migrate_header=lambda h: raise_(RuntimeError("historical header policy")))
            ],
            restore_current_header=check_id,
        )
    )
    with pytest.raises(SessionFormatError, match="historical header policy"):
        rejecting.migrate_header(SOURCE)

    bad_current = create_session_format_chain(
        SessionFormatChainOptions(
            current_version=1,
            migrations=[migration()],
            restore_current_header=lambda header: {"version": 1},
        )
    )
    with pytest.raises(SessionFormatError, match="current Session header restoration id"):
        bad_current.migrate_header(CURRENT_HEADER)

    wrong_version = create_session_format_chain(
        SessionFormatChainOptions(
            current_version=1,
            migrations=[migration()],
            restore_current_header=lambda header: {**header, "version": 0},
        )
    )
    with pytest.raises(SessionFormatError, match="header restorer returned v0"):
        wrong_version.migrate_header(CURRENT_HEADER)
