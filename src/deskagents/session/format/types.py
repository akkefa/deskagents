"""Upstream: packages/session/session-format/src/types.ts

Logical values are frozen JSON mappings (:class:`deskagents.util.values.FrozenJsonObject`).
Upstream's structural interfaces become :class:`typing.Protocol` classes, and the
declarations upstream writes as object literals become frozen dataclasses.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from deskagents.util.values import JsonValue

SessionFormatJsonValue = JsonValue
"""Lossless JSON value admitted at the durable Session boundary."""

SessionFormatJsonObject = Mapping[str, JsonValue]
"""Lossless JSON object admitted at the durable Session boundary."""

SessionFormatHeader = Mapping[str, JsonValue]
"""Logical Session metadata: ``version``, ``id``, ``createdAt``, ``isSeeded`` and
``delegationDepth`` always; ``cwd``, ``parentSession``, ``origin`` (``"subagent"``)
and ``agentPreset`` when present."""

SessionFormatEvent = Mapping[str, JsonValue]
"""One decoded logical Session event: ``type``, ``seq``, ``time`` and ``data``."""

SessionFormatRecovery = Literal["strict", "recoverable"]
"""Physical-row failure policy selected once for one restore."""

SessionFormatSourceKind = Literal["decoded", "transformed"]


@dataclass(frozen=True)
class SessionFormatArtifact:
    """One detached complete logical Session artifact."""

    header: SessionFormatHeader
    inherited_event_count: int
    """Exact inherited prefix length, available only after a body read."""
    events: Sequence[SessionFormatEvent]


class SessionFormatEventRun(Protocol):
    """A codec-owned compact run that adjacent migrations may consume without expanding."""

    @property
    def run_type(self) -> str: ...
    @property
    def first_seq(self) -> int: ...
    @property
    def event_count(self) -> int: ...
    def expand(self) -> Iterable[SessionFormatEvent]:
        """Expand the run for a migration that has no direct handler."""
        ...


class SessionFormatMigrationContext(Protocol):
    """Synchronous output channel owned by a compiled migration stream."""

    def emit_event(self, event: SessionFormatEvent) -> None:
        """Deliver one settled event to the next stage before returning."""
        ...

    def emit_run(self, run: SessionFormatEventRun) -> None:
        """Deliver one compact event run to the next stage before returning."""
        ...


class SessionFormatMigrationStage(Protocol):
    """Stateful adjacent migration stage used by streaming persistence restores.

    It may also define ``header_inherited_event_count``: the target inherited cut when
    it is unchanged and known before EOF.
    """

    def transform_event(
        self, event: SessionFormatEvent, context: SessionFormatMigrationContext
    ) -> None: ...
    def transform_run(
        self, run: SessionFormatEventRun, context: SessionFormatMigrationContext
    ) -> None: ...
    def finish(self, context: SessionFormatMigrationContext) -> int:
        """Emit trailing target items and return the exact target inherited cut."""
        ...


@dataclass(frozen=True)
class SessionFormatMigrationStageInput:
    """Headers and inherited cut supplied when one adjacent body stage is created."""

    source_header: SessionFormatHeader
    target_header: SessionFormatHeader
    source_inherited_event_count: int | None
    """Inherited prefix length in source coordinates, or None until body decoding completes."""
    source_kind: SessionFormatSourceKind


@dataclass(frozen=True)
class SessionFormatMigration:
    """One independently maintained adjacent streaming migration."""

    name: str
    from_version: int
    to_version: int
    migrate_header: Callable[[SessionFormatHeader], SessionFormatHeader]
    """Convert one header without reading event bodies."""
    create_stage: Callable[[SessionFormatMigrationStageInput], SessionFormatMigrationStage]
    """Create the stateful body stage for one source artifact."""
    validate_target_header: Callable[[SessionFormatHeader], None]
    """Refuse any header that the adjacent target writer cannot emit."""


class SessionFormatArtifactDecoder(Protocol):
    """Stateful physical-row decoder used by streaming persistence restores.

    It may also define ``header_inherited_event_count``: the inherited cut known before
    body decoding (current formats may derive it at EOF instead).
    """

    @property
    def header(self) -> SessionFormatHeader: ...
    def decode_row(self, row_value: object, context: SessionFormatMigrationContext) -> None:
        """Decode one physical row and synchronously emit its events or compact run."""
        ...

    def finish(self, context: SessionFormatMigrationContext) -> int:
        """Finish row validation and return the exact inherited cut."""
        ...


@dataclass(frozen=True)
class SessionFormatCodec:
    """Pure physical JSON codec for one Session version's supported representations."""

    version: int
    decode_header: Callable[[object], SessionFormatHeader]
    """Decode one physical header into body-independent logical metadata."""
    create_decoder: Callable[[object, SessionFormatRecovery], SessionFormatArtifactDecoder]
    """Create one row-at-a-time decoder with an explicit failure policy."""


class SessionFormatCurrentEncoder(Protocol):
    """Stateless physical record encoder for the installed current format."""

    def encode_header(
        self, header: SessionFormatHeader, inherited_event_count: int
    ) -> SessionFormatJsonObject: ...
    def encode_event(self, event: SessionFormatEvent) -> SessionFormatJsonObject: ...


@dataclass(frozen=True, kw_only=True)
class SessionFormatChainOptions:
    """Inputs that compile the unique complete migration chain."""

    current_version: int
    migrations: Sequence[SessionFormatMigration]
    restore_current_header: Callable[[SessionFormatHeader], SessionFormatHeader]
    """Restore and validate a detached current header without reading event bodies."""


@dataclass(frozen=True, kw_only=True)
class SessionFormatCatalogOptions(SessionFormatChainOptions):
    """Inputs for a build-static physical codec and migration catalog."""

    codecs: Sequence[SessionFormatCodec]
    restore_current: Callable[[SessionFormatArtifact], SessionFormatArtifact]
    """Restore and validate a complete current artifact."""
    current_encoder: SessionFormatCurrentEncoder
    """Encode current records without materializing an artifact-sized row array."""
    restore_transformed_current: Callable[[SessionFormatArtifact], SessionFormatArtifact]
    """Validate an exclusively owned transformed artifact without copying or freezing it."""


@dataclass(frozen=True)
class SessionFormatRestoreOptions:
    """Policies applied by one physical-row restore.

    ``validation="current"`` applies all installed current-format validation;
    ``"transformed"`` applies released current-format validation only after migration,
    and current input receives only codec validation.
    """

    recovery: SessionFormatRecovery
    validation: Literal["transformed", "current"]


SessionFormatHeaderStatus = Literal["current", "migration-required", "unsupported", "malformed"]


@dataclass(frozen=True)
class SessionFormatHeaderReadResult:
    """Header-only classification that never inspects event rows.

    ``current`` and ``migration-required`` carry the latest logical ``header`` (the exact
    inherited cut requires a body read); ``unsupported`` and ``malformed`` carry a
    ``reason``. ``stored_version`` is absent only for a malformed version field.
    """

    status: SessionFormatHeaderStatus
    target_version: int
    stored_version: int | None = None
    header: SessionFormatHeader | None = None
    reason: str | None = None
