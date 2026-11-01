"""Upstream: packages/session/session-format/src/chain.ts"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

from .errors import SessionFormatError, SessionFormatUnsupportedMigrationError
from .json import session_format_count, session_format_version, snapshot_session_format_header
from .types import (
    SessionFormatChainOptions,
    SessionFormatEvent,
    SessionFormatEventRun,
    SessionFormatHeader,
    SessionFormatMigration,
    SessionFormatMigrationContext,
    SessionFormatMigrationStage,
    SessionFormatMigrationStageInput,
)


def define_session_format_migration(migration: SessionFormatMigration) -> SessionFormatMigration:
    """Validate one adjacent migration declaration and return a frozen copy."""
    if not isinstance(migration.name, str) or not migration.name:
        raise SessionFormatError("Session migration name must be a non-empty string")
    source = session_format_version(migration.from_version, f"{migration.name} fromVersion")
    target = session_format_version(migration.to_version, f"{migration.name} toVersion")
    if target != source + 1:
        raise SessionFormatError(f"{migration.name} must declare adjacent v{source}->v{source + 1}")
    return SessionFormatMigration(
        name=migration.name,
        from_version=source,
        to_version=target,
        migrate_header=migration.migrate_header,
        create_stage=migration.create_stage,
        validate_target_header=migration.validate_target_header,
    )


def create_session_format_chain(options: SessionFormatChainOptions) -> SessionFormatChain:
    """Compile a unique, complete adjacent migration chain."""
    return SessionFormatChain(options)


class SessionFormatChain:
    """Pure adjacent planner and streaming migration compiler."""

    def __init__(self, options: SessionFormatChainOptions) -> None:
        self.current_version = session_format_version(
            options.current_version, "current Session format version"
        )
        self._restore_current_header = options.restore_current_header
        by_from: dict[int, SessionFormatMigration] = {}
        names: set[str] = set()
        for candidate in options.migrations:
            migration = define_session_format_migration(candidate)
            if migration.from_version in by_from:
                raise SessionFormatError(
                    f"Session migration v{migration.from_version}->v{migration.to_version} "
                    "is duplicated"
                )
            if migration.name in names:
                raise SessionFormatError(f'Session migration name "{migration.name}" is duplicated')
            by_from[migration.from_version] = migration
            names.add(migration.name)
        ordered: list[SessionFormatMigration] = []
        for version in range(self.current_version):
            found = by_from.get(version)
            if found is None:
                raise SessionFormatUnsupportedMigrationError(
                    f"Session migration v{version}->v{version + 1} is missing"
                )
            ordered.append(found)
        if len(by_from) != len(ordered):
            invalid = next(v for v in by_from if v >= self.current_version)
            raise SessionFormatError(
                f"Session migration from v{invalid} does not lead to current "
                f"v{self.current_version}"
            )
        self._migrations: tuple[SessionFormatMigration, ...] = tuple(ordered)

    def _plan(self, from_version: object) -> tuple[SessionFormatMigration, ...]:
        source = session_format_version(from_version, "stored Session format version")
        if source > self.current_version:
            raise SessionFormatUnsupportedMigrationError(
                f"stored Session uses newer format v{source}; "
                f"this build writes v{self.current_version}"
            )
        return self._migrations[source:]

    def create_stream(
        self,
        source_header: SessionFormatHeader,
        source_cut: int | None,
        output: SessionFormatMigrationContext,
    ) -> SessionFormatMigrationStream:
        """Compile the complete migration stage chain for one decoded source artifact."""
        header = source_header
        validated_cut = (
            None
            if source_cut is None
            else session_format_count(source_cut, "Session inherited event count")
        )
        inherited = validated_cut
        stages: list[_CompiledStage] = []
        for index, migration in enumerate(self._plan(header.get("version"))):
            target_header = self._advance_header(migration, header)
            try:
                stage = migration.create_stage(
                    SessionFormatMigrationStageInput(
                        source_header=header,
                        target_header=target_header,
                        source_inherited_event_count=inherited,
                        source_kind="decoded" if index == 0 else "transformed",
                    )
                )
            except BaseException as error:
                _throw_unsupported_refusal(migration, error)
            header = target_header
            stages.append(_CompiledStage(migration, stage))
            inherited = getattr(stage, "header_inherited_event_count", None)
        return SessionFormatMigrationStream(header, validated_cut, stages, output)

    def migrate_header(self, source: SessionFormatHeader) -> SessionFormatHeader:
        """Convert only a supported header to the current logical representation."""
        current = snapshot_session_format_header(source, "stored Session header")
        for migration in self._plan(current.get("version")):
            current = self._advance_header(migration, current)
        current = snapshot_session_format_header(
            self._restore_current_header(current), "current Session header restoration"
        )
        if current["version"] != self.current_version:
            raise SessionFormatError(
                f"current Session header restorer returned v{current['version']}; "
                f"expected v{self.current_version}"
            )
        return current

    def _advance_header(
        self, migration: SessionFormatMigration, source: SessionFormatHeader
    ) -> SessionFormatHeader:
        try:
            target = migration.migrate_header(
                snapshot_session_format_header(source, f"{migration.name} header input")
            )
        except BaseException as error:
            _throw_unsupported_refusal(migration, error, "Session header")
        current = snapshot_session_format_header(target, f"{migration.name} header output")
        if current["version"] != migration.to_version:
            raise SessionFormatError(
                f"{migration.name} header returned v{current['version']}; "
                f"expected v{migration.to_version}"
            )
        try:
            migration.validate_target_header(current)
        except BaseException as error:
            _throw_unsupported_refusal(migration, error, "Session header")
        return current


@dataclass(frozen=True)
class _CompiledStage:
    migration: SessionFormatMigration
    stage: SessionFormatMigrationStage


class _ChainedMigrationContext:
    def __init__(self, entry: _CompiledStage, output: SessionFormatMigrationContext) -> None:
        self._entry = entry
        self._output = output

    def emit_event(self, event: SessionFormatEvent) -> None:
        try:
            self._entry.stage.transform_event(event, self._output)
        except BaseException as error:
            _throw_unsupported_refusal(self._entry.migration, error)

    def emit_run(self, run: SessionFormatEventRun) -> None:
        try:
            self._entry.stage.transform_run(run, self._output)
        except BaseException as error:
            _throw_unsupported_refusal(self._entry.migration, error)

    def finish(self) -> int:
        try:
            target_cut = self._entry.stage.finish(self._output)
        except BaseException as error:
            _throw_unsupported_refusal(self._entry.migration, error)
        declared = getattr(self._entry.stage, "header_inherited_event_count", None)
        if declared is not None and declared != target_cut:
            raise SessionFormatError(
                f"{self._entry.migration.name} changed its predeclared inherited cut"
            )
        return target_cut


class SessionFormatMigrationStream:
    """One composed migration chain that emits settled current events to its owner."""

    def __init__(
        self,
        header: SessionFormatHeader,
        source_inherited_event_count: int | None,
        entries: Sequence[_CompiledStage],
        output: SessionFormatMigrationContext,
    ) -> None:
        self.header = header
        self._source_cut = source_inherited_event_count
        stages: list[_ChainedMigrationContext] = []
        downstream: SessionFormatMigrationContext = output
        for entry in reversed(entries):
            context = _ChainedMigrationContext(entry, downstream)
            stages.insert(0, context)
            downstream = context
        self._input = downstream
        self._stages = tuple(stages)

    def emit_event(self, event: SessionFormatEvent) -> None:
        self._input.emit_event(event)

    def emit_run(self, run: SessionFormatEventRun) -> None:
        self._input.emit_run(run)

    def finish(self) -> int:
        """Settle all migration stages and return the exact current inherited cut."""
        inherited = self._source_cut
        for stage in self._stages:
            inherited = stage.finish()
        return session_format_count(inherited, "finished Session inherited event count")


def _throw_unsupported_refusal(
    migration: SessionFormatMigration, error: BaseException, subject: str = "Session"
) -> NoReturn:
    if isinstance(error, SessionFormatUnsupportedMigrationError):
        raise error
    if not isinstance(error, Exception):
        raise error  # KeyboardInterrupt, CancelledError and friends are never refusals.
    detail = (
        error.args[0] if len(error.args) == 1 and isinstance(error.args[0], str) else str(error)
    )
    raise SessionFormatUnsupportedMigrationError(
        f"{migration.name} refuses this format v{migration.from_version} {subject}: {detail}"
    ) from error
