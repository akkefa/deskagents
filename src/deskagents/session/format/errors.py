"""Upstream: packages/session/session-format/src/error.ts"""

from __future__ import annotations


class SessionFormatError(Exception):
    """A durable Session artifact cannot be restored or migrated losslessly."""


class SessionFormatUnsupportedMigrationError(SessionFormatError):
    """A readable artifact whose released source policy has no supported migration."""
