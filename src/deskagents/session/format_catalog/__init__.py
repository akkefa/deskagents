"""Build-static first-party Session format codec and migration catalog.

Upstream: packages/session/session-format-catalog (``@deepseek-ai/dsh-session-format-catalog``)
at ce7073d.

Ported: ``generated.ts`` (:data:`session_format_catalog`,
:data:`session_format_catalog_options`), ``current.ts`` and
``message-projections.ts``. ``children.ts``
(``createSessionFormatCatalogWithChildren``) and ``historical.ts``
(``historicalSessionFormatCatalog``) exist only for historical v3->v4 migrations and
follow with them.

Python differences:

- Only v4 is readable. The v0-v3 codecs and migrations are refusing stand-ins whose
  header decode raises :class:`SessionFormatUnsupportedMigrationError` ("not supported
  by this Python build yet"), so ``read_header`` reports ``unsupported`` and
  ``create_restore`` raises that error.
- :data:`current_session_message_projections` is empty until the image-offload
  projection is ported.
- Version skew in ``current.ts`` raises ``ValueError`` (upstream ``Error``).
"""

from deskagents.session.format import SessionFormatUnsupportedMigrationError

from .current import (
    validate_installed_current_session_artifact,
    validate_installed_current_session_header,
)
from .generated import session_format_catalog, session_format_catalog_options
from .message_projections import current_session_message_projections

__all__ = [
    "SessionFormatUnsupportedMigrationError",
    "current_session_message_projections",
    "session_format_catalog",
    "session_format_catalog_options",
    "validate_installed_current_session_artifact",
    "validate_installed_current_session_header",
]
