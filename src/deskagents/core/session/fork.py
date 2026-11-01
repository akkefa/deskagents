"""Fork seed construction over an exact source-event prefix.

Upstream: packages/core/session/src/fork.ts
"""

from __future__ import annotations

from collections.abc import Sequence

from deskagents.util.values import deep_freeze

from .repair import open_turn_closers
from .types import SessionEvent, SessionSeq


def build_fork_seed(events: Sequence[SessionEvent], boundary: SessionSeq) -> list[SessionEvent]:
    """Copy an inclusive prefix, mark its inherited cut, and close its open tail.

    Closed steps and turns are preserved unchanged; the open tail receives forked
    tool results and step/turn endings after the inherited prefix. The caller validates
    that ``boundary`` is an existing contiguous event seq.
    """
    prefix = list(events[: boundary + 1])
    prefix.append(
        deep_freeze(
            {
                "type": "session/end-seed",
                "seq": boundary + 1,
                "time": events[boundary]["time"],
                "data": {"inherited": True},
            }
        )
    )
    return prefix + open_turn_closers(prefix, "forked")
