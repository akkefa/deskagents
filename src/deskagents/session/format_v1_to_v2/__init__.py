"""Frozen released-v2 physical row framing, the codec later formats reuse.

Upstream: packages/session/session-format-v1-to-v2 (``@deepseek-ai/dsh-session-format-v1-to-v2``)
at ce7073d.

Only the released-v2 codec (``codec.ts``) and the header validation it needs
(``assertReleasedV2Header`` from ``validation.ts``) are ported. The v1->v2 migration,
the released-v1 codec, the v2 event dispositions and ``restoreReleasedV2Artifact``
follow when Python reads v0-v3 sessions.

Python differences:

- The codec is a frozen :class:`ReleasedSessionFormatCodec` dataclass: a
  :class:`~deskagents.session.format.SessionFormatCodec` whose ``encode_header`` and
  ``encode_event`` fields make it a current encoder too.
- Decoded events are deep-frozen copies of their rows. Upstream returns the parsed row
  and the current-format restore freezes it in place later.
- Numeric comparisons follow JavaScript ``===``: ``true`` never equals ``1``.
- A row failure that is not an ``Exception`` (``KeyboardInterrupt``, cancellation)
  propagates as is instead of being contained in a recoverable tail.
"""

from .codec import ReleasedSessionFormatCodec, released_v2_session_format_codec
from .validation import assert_released_v2_header

__all__ = [
    "ReleasedSessionFormatCodec",
    "assert_released_v2_header",
    "released_v2_session_format_codec",
]
