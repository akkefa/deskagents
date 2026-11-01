"""Shared stand-ins for the chain and catalog tests (upstream builds them as literals)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from deskagents.session.format import (
    SessionFormatEvent,
    SessionFormatEventRun,
    SessionFormatHeader,
    SessionFormatMigration,
    SessionFormatMigrationContext,
    SessionFormatMigrationStageInput,
)


@dataclass
class Stage:
    """An identity migration stage; override any part with keyword arguments."""

    inherited: int | None = None
    transform_event_fn: (
        Callable[[SessionFormatEvent, SessionFormatMigrationContext], None] | None
    ) = None
    transform_run_fn: (
        Callable[[SessionFormatEventRun, SessionFormatMigrationContext], None] | None
    ) = None
    finish_fn: Callable[[SessionFormatMigrationContext], int] | None = None
    declared: bool = True

    def __post_init__(self) -> None:
        if self.declared and self.inherited is not None:
            self.header_inherited_event_count = self.inherited

    def transform_event(
        self, event: SessionFormatEvent, context: SessionFormatMigrationContext
    ) -> None:
        if self.transform_event_fn is not None:
            self.transform_event_fn(event, context)
        else:
            context.emit_event(event)

    def transform_run(
        self, run: SessionFormatEventRun, context: SessionFormatMigrationContext
    ) -> None:
        if self.transform_run_fn is not None:
            self.transform_run_fn(run, context)
        else:
            context.emit_run(run)

    def finish(self, context: SessionFormatMigrationContext) -> int:
        if self.finish_fn is not None:
            return self.finish_fn(context)
        return self.inherited or 0


def identity_stage(inherited: int | None) -> Stage:
    return Stage(inherited)


def raise_(error: BaseException) -> Any:
    raise error


def migration(**overrides: Any) -> SessionFormatMigration:
    base: dict[str, Any] = {
        "name": "@test/v0-to-v1",
        "from_version": 0,
        "to_version": 1,
        "migrate_header": lambda header: {**header, "version": 1},
        "create_stage": lambda given: identity_stage(given.source_inherited_event_count),
        "validate_target_header": lambda header: None,
    }
    base.update(overrides)
    return SessionFormatMigration(**base)


class Discard:
    def emit_event(self, event: SessionFormatEvent) -> None:
        pass

    def emit_run(self, run: SessionFormatEventRun) -> None:
        pass


@dataclass
class Run:
    run_type: str
    first_seq: int
    event_count: int
    events: list[SessionFormatEvent] = field(default_factory=list)

    def expand(self) -> Iterator[SessionFormatEvent]:
        yield from self.events


class Recorder:
    """Records calls like vitest's ``vi.fn``."""

    def __init__(self, implementation: Callable[..., Any]) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._implementation = implementation

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self._implementation(*args)


def headers(*values: SessionFormatHeader) -> Iterable[SessionFormatHeader]:
    return values


__all__ = [
    "Discard",
    "Recorder",
    "Run",
    "SessionFormatMigrationStageInput",
    "Stage",
    "identity_stage",
    "migration",
    "raise_",
]
