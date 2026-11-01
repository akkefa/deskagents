"""Public application identity attached to provider requests.

Adapted from ``packages/llm/llm/src/attribution.ts`` at ce7073d.  The Python
distribution identifies itself as ``deskagents`` instead of claiming to be the
official DeepSeek product.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

__all__ = ["APP_IDENTITY", "AppIdentity", "attribution_headers", "user_agent"]


def _version() -> str:
    try:
        return version("deskagents")
    except PackageNotFoundError:
        return "0.1.0.dev0"


@dataclass(frozen=True, slots=True)
class AppIdentity:
    """Non-secret product facts safe to send on every provider request."""

    product: str
    version: str
    url: str


APP_IDENTITY = AppIdentity(
    product="deskagents",
    version=_version(),
    url="https://github.com/akkefa/deskagents",
)


def user_agent(identity: AppIdentity = APP_IDENTITY) -> str:
    """Render RFC-style ``product/version (+url)`` attribution."""
    return f"{identity.product}/{identity.version} (+{identity.url})"


def attribution_headers(identity: AppIdentity = APP_IDENTITY) -> dict[str, str]:
    """Headers every concrete model adapter must merge into provider requests."""
    return {"user-agent": user_agent(identity)}
