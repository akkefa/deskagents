"""Native v4 Session framing, row admission and restore validation.

Upstream: packages/session/session-format-v3-to-v4 (``@deepseek-ai/dsh-session-format-v3-to-v4``)
at ce7073d.

Only the v4 half is ported: ``codec.ts``, ``validation.ts`` and the modules they use
(``message-sources.ts``, ``retired-syntax.ts``, ``system-message.ts``,
``developer.ts``, ``fork-result.ts``, ``tool-role.ts``, ``relationships.ts``,
``mapEventMessages`` from ``sources.ts`` and ``catalogFact`` from ``facts.ts``). The
v3->v4 migration (``migration.ts``, ``content.ts``, ``references.ts``,
``extension-identities.ts``, the v3 source and tool-result lifts and the historical
child facts) follows when Python reads v3 sessions.

Python differences:

- The codec is a frozen
  :class:`~deskagents.session.format_v1_to_v2.ReleasedSessionFormatCodec`. Its decoded
  headers and events are deep-frozen.
- Validators read absent members as JavaScript ``undefined`` (distinct from JSON
  ``null``) and compare with ``===`` semantics, so ``true`` never equals ``1``.
  Diagnostics format values like ``String(value)``.
- ``known_event_types`` is any collection of strings (upstream: ``ReadonlySet``).
"""

from .codec import assert_v4_row_admission, released_v4_session_format_codec
from .validation import (
    assert_released_v4_header,
    assert_released_v4_relationships,
    restore_released_v4_artifact,
    validate_delivery_accepted,
)

__all__ = [
    "assert_released_v4_header",
    "assert_released_v4_relationships",
    "assert_v4_row_admission",
    "released_v4_session_format_codec",
    "restore_released_v4_artifact",
    "validate_delivery_accepted",
]
