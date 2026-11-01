"""Shared transport-level normalization for supplied provider API keys."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal, TypeAlias, cast

from deskagents.util.values import deep_freeze

from .errors import INVALID_CREDENTIAL_CODE, LlmError

_LEGAL_API_KEY = re.compile(r"^[\x21-\x7e]+$")


ApiKeyCheck: TypeAlias = Mapping[str, bool | str]
ApiKeyRejection: TypeAlias = Literal["empty", "illegalCharacters"]


def normalize_api_key(raw: str) -> ApiKeyCheck:
    value = raw.strip()
    if not value:
        return cast(ApiKeyCheck, deep_freeze({"ok": False, "reason": "empty"}))
    if _LEGAL_API_KEY.fullmatch(value) is None:
        return cast(ApiKeyCheck, deep_freeze({"ok": False, "reason": "illegalCharacters"}))
    return cast(ApiKeyCheck, deep_freeze({"ok": True, "value": value}))


def assert_usable_api_key(raw: str, package: str, ref: str) -> str:
    """Return a trimmed usable key or raise without echoing any secret material."""
    checked = normalize_api_key(raw)
    if checked["ok"] is True:
        return str(checked["value"])
    if checked["reason"] == "empty":
        message = (
            f"{package}: the API key resolved from {ref} is blank; set {ref} to the raw key "
            "(the web Models page writes it) or export it in the launching environment"
        )
    else:
        message = (
            f"{package}: the API key resolved from {ref} contains characters no HTTP header "
            f"can carry; set {ref} to the raw key alone (the web Models page writes it)"
        )
    raise LlmError(message, INVALID_CREDENTIAL_CODE)
