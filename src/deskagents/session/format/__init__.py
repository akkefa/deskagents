"""Pure adjacent streaming Session format migration machinery.

Upstream: packages/session/session-format (``@deepseek-ai/dsh-session-format``) at ce7073d.

Python differences:

- Logical headers and events are frozen JSON mappings, so a snapshot is detached and
  immutable as upstream's ``deepFreeze`` makes it.
- Declarations upstream writes as object literals (migrations, codecs, options) are
  frozen dataclasses; stateful parts (decoders, stages, encoders) are protocols.
  Optional members such as ``header_inherited_event_count`` are read with ``getattr``.
- A refusal raised as a non-``Exception`` (``KeyboardInterrupt``, cancellation) is
  re-raised as is instead of being wrapped as an unsupported migration.
"""

from .catalog import SessionFormatCatalog, SessionFormatRestore, create_session_format_catalog
from .chain import (
    SessionFormatChain,
    SessionFormatMigrationStream,
    create_session_format_chain,
    define_session_format_migration,
)
from .context import SessionFormatEventCollector
from .errors import SessionFormatError, SessionFormatUnsupportedMigrationError
from .filename import parse_session_format_log_filename, session_format_log_filename
from .json import (
    inspect_session_format_version,
    is_session_format_json_object,
    session_format_count,
    session_format_safe_integer,
    session_format_version,
    snapshot_session_format_header,
    snapshot_session_format_json,
)
from .types import (
    SessionFormatArtifact,
    SessionFormatArtifactDecoder,
    SessionFormatCatalogOptions,
    SessionFormatChainOptions,
    SessionFormatCodec,
    SessionFormatCurrentEncoder,
    SessionFormatEvent,
    SessionFormatEventRun,
    SessionFormatHeader,
    SessionFormatHeaderReadResult,
    SessionFormatJsonObject,
    SessionFormatJsonValue,
    SessionFormatMigration,
    SessionFormatMigrationContext,
    SessionFormatMigrationStage,
    SessionFormatMigrationStageInput,
    SessionFormatRecovery,
    SessionFormatRestoreOptions,
)

__all__ = [
    "SessionFormatArtifact",
    "SessionFormatArtifactDecoder",
    "SessionFormatCatalog",
    "SessionFormatCatalogOptions",
    "SessionFormatChain",
    "SessionFormatChainOptions",
    "SessionFormatCodec",
    "SessionFormatCurrentEncoder",
    "SessionFormatError",
    "SessionFormatEvent",
    "SessionFormatEventCollector",
    "SessionFormatEventRun",
    "SessionFormatHeader",
    "SessionFormatHeaderReadResult",
    "SessionFormatJsonObject",
    "SessionFormatJsonValue",
    "SessionFormatMigration",
    "SessionFormatMigrationContext",
    "SessionFormatMigrationStage",
    "SessionFormatMigrationStageInput",
    "SessionFormatMigrationStream",
    "SessionFormatRecovery",
    "SessionFormatRestore",
    "SessionFormatRestoreOptions",
    "SessionFormatUnsupportedMigrationError",
    "create_session_format_catalog",
    "create_session_format_chain",
    "define_session_format_migration",
    "inspect_session_format_version",
    "is_session_format_json_object",
    "parse_session_format_log_filename",
    "session_format_count",
    "session_format_log_filename",
    "session_format_safe_integer",
    "session_format_version",
    "snapshot_session_format_header",
    "snapshot_session_format_json",
]
