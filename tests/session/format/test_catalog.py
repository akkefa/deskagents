"""Ported from packages/session/session-format/tests/catalog.spec.ts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import pytest
from support.session_format import Recorder, Run, identity_stage, migration, raise_

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatCatalog,
    SessionFormatCatalogOptions,
    SessionFormatCodec,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatHeader,
    SessionFormatHeaderReadResult,
    SessionFormatJsonObject,
    SessionFormatMigration,
    SessionFormatMigrationContext,
    SessionFormatRecovery,
    SessionFormatRestoreOptions,
    SessionFormatUnsupportedMigrationError,
    create_session_format_catalog,
)

SPEC = "packages/session/session-format/tests/catalog.spec.ts"
GROUP = "Session format catalog"

OLD_HEADER: SessionFormatHeader = {
    "version": 0,
    "id": "old",
    "createdAt": 1,
    "isSeeded": False,
    "delegationDepth": 0,
}
EVENT: SessionFormatEvent = {"type": "turn/start", "seq": 0, "time": 2, "data": {"turn": 1}}
STRICT_CURRENT = SessionFormatRestoreOptions(recovery="strict", validation="current")
STRICT_TRANSFORMED = SessionFormatRestoreOptions(recovery="strict", validation="transformed")


@dataclass
class Decoder:
    header: Any
    inherited: int | None = 0
    finish_with: int | None = None
    decode: Callable[[object, SessionFormatMigrationContext], None] | None = None

    def __post_init__(self) -> None:
        if self.inherited is not None:
            self.header_inherited_event_count = self.inherited

    def decode_row(self, row_value: object, context: SessionFormatMigrationContext) -> None:
        if self.decode is not None:
            self.decode(row_value, context)
        else:
            context.emit_event(row_value)  # type: ignore[arg-type]

    def finish(self, context: SessionFormatMigrationContext) -> int:
        if self.finish_with is not None:
            return self.finish_with
        return self.inherited or 0


class Encoder:
    def encode_header(
        self, header: SessionFormatHeader, inherited_event_count: int
    ) -> SessionFormatJsonObject:
        return header

    def encode_event(self, event: SessionFormatEvent) -> SessionFormatJsonObject:
        return event


def codec(version: int, inherited_event_count: int = 0) -> SessionFormatCodec:
    def create_decoder(header_value: object, recovery: SessionFormatRecovery) -> Decoder:
        return Decoder(header_value, inherited_event_count)

    def decode_header(value: object) -> SessionFormatHeader:
        return value  # type: ignore[return-value]

    return SessionFormatCodec(
        version=version, decode_header=decode_header, create_decoder=create_decoder
    )


def with_version(version: int) -> Callable[[SessionFormatArtifact], SessionFormatArtifact]:
    return lambda artifact: replace(artifact, header={**artifact.header, "version": version})


def catalog(
    edge: SessionFormatMigration | None = None,
    restore_version: int = 1,
    source_codec: SessionFormatCodec | None = None,
    current_encoder: Any = None,
    restore_current_header: Callable[[SessionFormatHeader], SessionFormatHeader] = lambda h: h,
) -> SessionFormatCatalog:
    return create_session_format_catalog(
        SessionFormatCatalogOptions(
            current_version=1,
            codecs=[source_codec or codec(0), codec(1)],
            current_encoder=current_encoder or Encoder(),
            migrations=[edge or migration()],
            restore_current=with_version(restore_version),
            restore_transformed_current=with_version(restore_version),
            restore_current_header=restore_current_header,
        )
    )


def plain_options(**overrides: Any) -> SessionFormatCatalogOptions:
    base: dict[str, Any] = {
        "current_version": 1,
        "codecs": [codec(0), codec(1)],
        "current_encoder": Encoder(),
        "migrations": [migration()],
        "restore_current": lambda artifact: artifact,
        "restore_transformed_current": lambda artifact: artifact,
        "restore_current_header": lambda header: header,
    }
    base.update(overrides)
    return SessionFormatCatalogOptions(**base)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > classifies headers and restores physical rows through the compiled chain"
)
def test_classifies_headers_and_restores_through_chain() -> None:
    create_stage = Recorder(lambda given: identity_stage(given.source_inherited_event_count))
    current = catalog(migration(create_stage=create_stage))

    assert current.read_header(OLD_HEADER) == SessionFormatHeaderReadResult(
        status="migration-required",
        stored_version=0,
        target_version=1,
        header={**OLD_HEADER, "version": 1},
    )
    newer = current.read_header({"version": 2})
    assert (newer.status, newer.stored_version, newer.target_version) == ("unsupported", 2, 1)
    same = current.read_header({**OLD_HEADER, "version": 1})
    assert (same.status, same.stored_version, same.target_version) == ("current", 1, 1)
    broken = current.read_header({"version": "broken"})
    assert (broken.status, broken.target_version) == ("malformed", 1)

    restore = current.create_restore(OLD_HEADER, STRICT_CURRENT)
    restore.decode_row(EVENT)
    restored = restore.finish()
    assert restored.header["version"] == 1 and list(restored.events) == [EVENT]
    assert len(create_stage.calls) == 1


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > restores current rows without a migration and encodes one record at a time"
)
def test_restores_current_rows_and_encodes_records() -> None:
    current = catalog()
    header = {**OLD_HEADER, "version": 1}
    restore = current.create_restore(header, STRICT_CURRENT)
    restore.decode_row(EVENT)

    assert restore.finish() == SessionFormatArtifact(header, 0, [EVENT])
    assert current.encode_current_header(header, 0) == header
    assert current.encode_current_event(EVENT) == EVENT
    with pytest.raises(SessionFormatError, match="requires Session format v1"):
        current.encode_current_header(OLD_HEADER, 0)

    physical_only = current.create_restore(header, STRICT_TRANSFORMED)
    physical_only.decode_row(EVENT)
    assert physical_only.finish() == SessionFormatArtifact(header, 0, [EVENT])

    transformed = current.create_restore(OLD_HEADER, STRICT_TRANSFORMED)
    transformed.decode_row(EVENT)
    result = transformed.finish()
    assert result.header["version"] == 1 and list(result.events) == [EVENT]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > classifies migrated target validation failures as unsupported"
)
@pytest.mark.parametrize(
    "target_failure",
    [
        RuntimeError("target artifact is incompatible"),
        ValueError("non-Error target artifact refusal"),
    ],
)
def test_classifies_target_validation_failures(target_failure: Exception) -> None:
    current = create_session_format_catalog(
        plain_options(restore_transformed_current=lambda artifact: raise_(target_failure))
    )
    restore = current.create_restore(OLD_HEADER, STRICT_TRANSFORMED)
    restore.decode_row(EVENT)

    with pytest.raises(SessionFormatUnsupportedMigrationError) as raised:
        restore.finish()
    assert raised.value.__cause__ is target_failure


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > preserves current-validation failures and explicit migration refusals"
)
def test_preserves_current_failures_and_explicit_refusals() -> None:
    current_failure = RuntimeError("installed current validation failed")
    current = create_session_format_catalog(
        plain_options(restore_current=lambda artifact: raise_(current_failure))
    ).create_restore(OLD_HEADER, STRICT_CURRENT)
    with pytest.raises(RuntimeError) as raised:
        current.finish()
    assert raised.value is current_failure

    explicit = SessionFormatUnsupportedMigrationError("explicit target refusal")
    transformed = create_session_format_catalog(
        plain_options(restore_transformed_current=lambda artifact: raise_(explicit))
    ).create_restore(OLD_HEADER, STRICT_TRANSFORMED)
    with pytest.raises(SessionFormatUnsupportedMigrationError) as kept:
        transformed.finish()
    assert kept.value is explicit


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects a current encoder that returns a non-current header"
)
def test_rejects_non_current_encoder_header() -> None:
    class BadEncoder(Encoder):
        def encode_header(
            self, header: SessionFormatHeader, inherited_event_count: int
        ) -> SessionFormatJsonObject:
            return {**header, "version": 0}

    current = catalog(current_encoder=BadEncoder())
    with pytest.raises(SessionFormatError, match="non-current header"):
        current.encode_current_header({**OLD_HEADER, "version": 1}, 0)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects duplicate, missing, and future codec declarations"
)
def test_rejects_bad_codec_declarations() -> None:
    with pytest.raises(SessionFormatError, match="codec v0 is duplicated"):
        create_session_format_catalog(plain_options(codecs=[codec(0), codec(0), codec(1)]))
    with pytest.raises(SessionFormatError, match="codec v1 is missing"):
        create_session_format_catalog(plain_options(codecs=[codec(0)]))
    with pytest.raises(SessionFormatError, match="codec v2 is newer"):
        create_session_format_catalog(plain_options(codecs=[codec(0), codec(1), codec(2)]))


@pytest.mark.upstream(f"{SPEC}::{GROUP} > preserves unsupported header and migration failures")
def test_preserves_unsupported_header_and_migration_failures() -> None:
    refusing = replace(
        codec(0),
        decode_header=lambda value: raise_(RuntimeError("bad header")),
        create_decoder=lambda value, recovery: raise_(RuntimeError("bad body")),
    )
    current = catalog(source_codec=refusing)
    result = current.read_header(OLD_HEADER)
    assert (result.status, result.reason) == ("malformed", "bad header")
    with pytest.raises(RuntimeError, match="bad body"):
        current.create_restore(OLD_HEADER, STRICT_CURRENT)
    with pytest.raises(SessionFormatError, match="newer"):
        current.create_restore({"version": 2}, STRICT_CURRENT)

    non_error = catalog(
        source_codec=replace(
            codec(0), decode_header=lambda value: raise_(ValueError("non-Error header failure"))
        )
    )
    result = non_error.read_header(OLD_HEADER)
    assert (result.status, result.reason) == ("malformed", "non-Error header failure")


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > classifies migrated header refusals as unsupported and current "
    "corruption as malformed"
)
def test_classifies_header_refusals_and_corruption() -> None:
    unsupported = catalog(
        migration(
            validate_target_header=lambda header: raise_(RuntimeError("target header lacks marker"))
        )
    )
    result = unsupported.read_header(OLD_HEADER)
    assert result.status == "unsupported" and "target header lacks marker" in (result.reason or "")

    malformed = catalog(
        restore_current_header=lambda header: raise_(RuntimeError("current header is corrupt"))
    )
    result = malformed.read_header({**OLD_HEADER, "version": 1})
    assert (result.status, result.reason) == ("malformed", "current header is corrupt")


@pytest.mark.upstream(f"{SPEC}::{GROUP} > enforces decoder, stage, and restorer lifecycle results")
def test_enforces_lifecycle_results() -> None:
    from support.session_format import Stage

    changed_cut = migration(create_stage=lambda given: Stage(1, finish_fn=lambda context: 0))
    changed = catalog(changed_cut, 1, codec(0, 1)).create_restore(
        {**OLD_HEADER, "isSeeded": True}, STRICT_CURRENT
    )
    with pytest.raises(SessionFormatError, match="changed its predeclared inherited cut"):
        changed.finish()

    def emit_trailing(context: SessionFormatMigrationContext) -> int:
        context.emit_event(EVENT)
        return 0

    trailing = catalog(
        migration(
            create_stage=lambda given: Stage(
                0,
                transform_event_fn=lambda e, c: None,
                transform_run_fn=lambda r, c: None,
                finish_fn=emit_trailing,
            )
        )
    ).create_restore(OLD_HEADER, STRICT_CURRENT)
    assert list(trailing.finish().events) == [EVENT]

    wrong_version = catalog(migration(), 0).create_restore(OLD_HEADER, STRICT_CURRENT)
    with pytest.raises(SessionFormatError, match="returned v0"):
        wrong_version.finish()

    changed_decoder_cut = replace(
        codec(0),
        create_decoder=lambda value, recovery: Decoder(
            value, 0, finish_with=1, decode=lambda r, c: None
        ),
    )
    changed_decoder = catalog(source_codec=changed_decoder_cut).create_restore(
        OLD_HEADER, STRICT_CURRENT
    )
    with pytest.raises(SessionFormatError, match="decoder changed its predeclared inherited cut"):
        changed_decoder.finish()

    deferred_cut = replace(
        codec(0),
        create_decoder=lambda value, recovery: Decoder(
            value, None, finish_with=0, decode=lambda r, c: None
        ),
    )
    restore = catalog(source_codec=deferred_cut).create_restore(OLD_HEADER, STRICT_CURRENT)
    assert restore.finish().inherited_event_count == 0


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > expands an unhandled compact run without an intermediate array"
)
def test_expands_unhandled_compact_run() -> None:
    run = Run("test-run", 0, 1, [EVENT])
    source = replace(
        codec(0),
        create_decoder=lambda value, recovery: Decoder(
            value, 0, decode=lambda row, context: context.emit_run(run)
        ),
    )
    restore = catalog(source_codec=source).create_restore(OLD_HEADER, STRICT_CURRENT)
    restore.decode_row({})

    assert list(restore.finish().events) == [EVENT]
