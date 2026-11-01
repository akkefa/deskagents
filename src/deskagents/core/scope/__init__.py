"""Scoped-context primitive: mint a Cordis context that tags registrations with an
opaque identity, and build routing-only event filters for that identity.

Upstream: packages/core/scope (``@deepseek-ai/dsh-scope``) at ce7073d.

Python differences:

- Upstream dispatches through a carrier object passed as the Cordis ``thisArg``.
  deepseek-cordis filters listeners with an explicit ``filter_`` callable instead, so
  :func:`scope_target` returns a :class:`ScopeCarrier` that *is* that filter:
  ``ctx.emit(name, *args, filter_=scope_target(base, key))``.
- A base subject's own Cordis filter is its ``matches_scope(ctx)`` method, which
  deepseek-cordis services provide; upstream reads ``base[Context.filter]``.
- Scope keys are compared by identity and held weakly, so they must support weak
  references (plain ``object()`` does not; any class instance does).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from ._identity import WeakIdentityMap, walk
from .store import AnonymousEntries, NamedEntries, ScopedLayers, ScopeLayer

if TYPE_CHECKING:
    from deepseek_cordis import Context, Fiber

__all__ = [
    "AnonymousEntries",
    "NamedEntries",
    "Scope",
    "ScopeCarrier",
    "ScopeKey",
    "ScopeLayer",
    "ScopeParentBinding",
    "ScopedLayers",
    "bind_scope_parent",
    "carrier_key_of",
    "create_scope",
    "is_scope_carrier",
    "scope_chain_of",
    "scope_of",
    "scope_parent_of",
    "scope_target",
]

ScopeKey = object
"""An opaque, identity-compared scope key."""

T = TypeVar("T")

# Context metadata entry written by create_scope (upstream: Symbol('dsh.scope')).
_SCOPE = "dsh.scope"

# The enclosing scope of each key. Registration views inherit down the chain and
# event admission extends up it.
_parents: WeakIdentityMap[ScopeKey, ScopeKey] = WeakIdentityMap()


class ScopeParentBinding:
    """The privileged handle that may move one scope key's parent link."""

    __slots__ = ("_key",)

    def __init__(self, key: ScopeKey) -> None:
        self._key = key

    def rebind(self, parent: ScopeKey) -> None:
        """Re-link the bound key to a different parent, with the same cycle check.

        Valid only while nothing produced under the old parent is retained.
        """
        _link_scope_parent(self._key, parent)


def _link_scope_parent(key: ScopeKey, parent: ScopeKey) -> None:
    for cursor in walk(parent, _parents.get):
        if cursor is key:
            raise ValueError("dsh-scope: scope parent link would form a cycle")
    _parents.set(key, parent)


def bind_scope_parent(key: ScopeKey, parent: ScopeKey) -> ScopeParentBinding:
    """Bind ``parent`` as ``key``'s enclosing scope, once.

    A key that already has a parent raises: only the original binder, who holds the
    returned binding, may re-link it. A link that would close a cycle is rejected.
    """
    if key in _parents:
        raise ValueError(
            "dsh-scope: scope key is already bound to a parent; re-linking requires "
            "the binding returned by the original bind"
        )
    _link_scope_parent(key, parent)
    return ScopeParentBinding(key)


def scope_parent_of(key: ScopeKey) -> ScopeKey | None:
    """Read one key's enclosing scope, or ``None`` for a root scope."""
    return _parents.get(key)


def scope_chain_of(key: ScopeKey | None) -> list[ScopeKey]:
    """The chain from a key to its root ancestor, nearest first."""
    return list(walk(key, _parents.get))


class Scope:
    """A minted registration scope and its quiescent disposal boundaries."""

    __slots__ = ("ctx", "_fiber", "_disposing")

    def __init__(self, ctx: Context, fiber: Fiber) -> None:
        self.ctx = ctx
        """Context through which scope-owned registrations are made."""
        self._fiber = fiber
        self._disposing: asyncio.Future[None] | None = None

    @property
    def raw_dispose(self) -> Callable[[], Awaitable[None]]:
        """Exact Cordis disposer, for nesting this scope in an ordered composite effect."""
        return self._fiber.dispose

    async def dispose(self) -> None:
        """Dispose every scope-owned registration; racing calls await the same completion."""
        if self._disposing is None:
            self._disposing = asyncio.ensure_future(_quiesce(self._fiber))
        await asyncio.shield(self._disposing)


async def _quiesce(fiber: Fiber) -> None:
    # deepseek-cordis disposal is joinable: a second dispose() call, including one that
    # follows an earlier raw dispose, waits for the same teardown to finish.
    await fiber.dispose()


def _scope_plugin(ctx: Context, config: object) -> None:
    """Shared no-op plugin used as the backing scope fiber."""


def create_scope(ctx: Context, key: ScopeKey, *, parent: ScopeKey | None = None) -> Scope:
    """Mint a scope under ``ctx``.

    The scoped context inherits the minting plugin's dependency API and owns every
    registration made through it. ``parent`` binds the enclosing scope first.
    """
    if parent is not None:
        bind_scope_parent(key, parent)
    fiber = ctx.plugin(_scope_plugin)
    return Scope(fiber.ctx.extend({_SCOPE: key}), fiber)


def scope_of(ctx: Context) -> ScopeKey | None:
    """Read the nearest scope tag inherited by a context, or ``None`` when unscoped."""
    return ctx.metadata.get(_SCOPE)


class ScopeCarrier(Generic[T]):
    """A routing-only event filter built by :func:`scope_target`.

    It admits untagged listeners globally and tagged listeners whose key is the
    dispatch key or one of its ancestors, after the base subject's own filter. The
    type parameter records the subject type; the carrier does not expose the
    subject, which travels in the event arguments.
    """

    __slots__ = ("_base_filter", "_key")

    def __init__(self, base_filter: Callable[[Context], bool] | None, key: ScopeKey | None):
        self._base_filter = base_filter
        self._key = key

    def __call__(self, ctx: Context) -> bool:
        if self._base_filter is not None and not self._base_filter(ctx):
            return False
        tag = scope_of(ctx)
        if tag is None:
            return True
        return any(cursor is tag for cursor in walk(self._key, _parents.get))


def scope_target(base: T, key: ScopeKey | None) -> ScopeCarrier[T]:
    """Build the event filter routing a dispatch to ``key`` and its ancestors' listeners.

    A listener owned by an enclosing scope receives every descendant scope's events;
    a tag below the dispatch key stays excluded. Events flow up the chain, never down.
    """
    base_filter = getattr(base, "matches_scope", None)
    return ScopeCarrier(base_filter if callable(base_filter) else None, key)


def is_scope_carrier(value: object) -> bool:
    """Test whether a value was built by :func:`scope_target`."""
    return isinstance(value, ScopeCarrier)


def carrier_key_of(value: object) -> ScopeKey | None:
    """Read a carrier's routing key, or ``None`` for an unkeyed carrier or non-carrier."""
    return value._key if isinstance(value, ScopeCarrier) else None
