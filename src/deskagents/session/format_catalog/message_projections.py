"""Upstream: packages/session/session-format-catalog/src/message-projections.ts

Pure first-party message interpreters for detached current-format replay.
"""

from __future__ import annotations

from collections.abc import Sequence

from deskagents.core.session import SessionMessageProjection

current_session_message_projections: Sequence[SessionMessageProjection] = ()
"""Installed interpretation definitions; recovery listeners are mounted by their plugins.

Empty for now: upstream's only entry, ``imageOffloadProjection`` from
``@deepseek-ai/dsh-compaction-image-offload``, is ported with that package.
"""
