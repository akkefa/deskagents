"""Upstream: packages/session/session-format/src/context.ts"""

from __future__ import annotations

from .types import SessionFormatEvent, SessionFormatEventRun


class SessionFormatEventCollector:
    """Migration output context that expands compact runs into retained events."""

    def __init__(self) -> None:
        self.values: list[SessionFormatEvent] = []
        """Events retained by this collector in source order."""

    def emit_event(self, event: SessionFormatEvent) -> None:
        """Retain one settled event."""
        self.values.append(event)

    def emit_run(self, run: SessionFormatEventRun) -> None:
        """Expand one compact run directly into retained events."""
        self.values.extend(run.expand())
