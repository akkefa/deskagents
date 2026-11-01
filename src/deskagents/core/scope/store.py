"""Shared insertion-ordered storage and effect ownership for scope-aware registries.

Upstream: packages/core/scope/src/store.ts
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from ._identity import IdentityMap

if TYPE_CHECKING:
    from deepseek_cordis import Context, Effect

    from . import ScopeKey

T = TypeVar("T")
V = TypeVar("V")
L = TypeVar("L", bound="ScopeLayer")

# A generation with this many dead records, outnumbering live ones, is compacted
# once no iterator is reading it.
_COMPACT_AFTER = 32


class ScopeLayer(Protocol):
    """One scope's aggregate contribution to a registry."""

    def is_empty(self) -> bool:
        """Whether every table in this layer is empty."""
        ...


class _Record(Generic[T]):
    __slots__ = ("key", "value", "alive")

    def __init__(self, key: object, value: T) -> None:
        self.key = key
        self.value = value
        self.alive = True


class _Generation(Generic[T]):
    """One nonempty run of a table. Draining a table starts a new generation."""

    def __init__(self) -> None:
        self.records: list[_Record[T]] = []
        self.live = 0
        self.readers: weakref.WeakSet[_LiveIterator[T, object]] = weakref.WeakSet()

    def append(self, record: _Record[T]) -> None:
        self.records.append(record)
        self.live += 1

    def kill(self, record: _Record[T]) -> None:
        record.alive = False
        self.live -= 1
        dead = len(self.records) - self.live
        if dead >= _COMPACT_AFTER and dead > self.live and not self.readers:
            self.records = [r for r in self.records if r.alive]


class _LiveIterator(Generic[T, V]):
    """Iterates one generation in insertion order, seeing later appends.

    Matches JavaScript ``Map`` iterators: entries removed before they are reached
    are skipped, entries appended before the end are yielded, and once the table
    drains the iterator is finished even if new entries arrive.
    """

    def __init__(self, generation: _Generation[T], pick: Callable[[_Record[T]], V]) -> None:
        self._generation: _Generation[T] | None = generation
        self._index = 0
        self._pick = pick
        generation.readers.add(self)  # type: ignore[arg-type]

    def __iter__(self) -> _LiveIterator[T, V]:
        return self

    def __next__(self) -> V:
        generation = self._generation
        if generation is None:
            raise StopIteration
        records = generation.records
        while self._index < len(records):
            record = records[self._index]
            self._index += 1
            if record.alive:
                return self._pick(record)
        self._generation = None
        raise StopIteration


class _Table(Generic[T]):
    def __init__(self) -> None:
        self._generation: _Generation[T] = _Generation()

    def _add(self, key: object, value: T) -> Callable[[], None]:
        generation = self._generation
        record = _Record(key, value)
        generation.append(record)
        active = True

        def undo() -> None:
            nonlocal active
            if not active:
                return
            active = False
            self._removed(record)
            generation.kill(record)
            if generation.live == 0 and self._generation is generation:
                self._generation = _Generation()

        return undo

    def _removed(self, record: _Record[T]) -> None:
        """Hook for index maintenance before a record dies."""

    def _iterate(self, pick: Callable[[_Record[T]], V]) -> Iterator[V]:
        return _LiveIterator(self._generation, pick)

    def values(self) -> Iterator[T]:
        """Iterate live values in insertion order."""
        return self._iterate(lambda record: record.value)

    def is_empty(self) -> bool:
        """Test whether this table has no entries."""
        return self._generation.live == 0


class NamedEntries(_Table[V]):
    """Insertion-ordered named entries with caller-owned duplicate diagnostics.

    Values are borrowed. Iterators are live within one nonempty table generation;
    draining the table detaches them from later insertions. Each successful
    insertion returns an idempotent undo for that exact entry.
    """

    def __init__(self, duplicate_error: Callable[[str], BaseException]) -> None:
        super().__init__()
        self._duplicate_error = duplicate_error
        self._index: dict[str, _Record[V]] = {}

    def insert(self, name: str, value: V) -> Callable[[], None]:
        """Insert one unique name; return an undo that removes only this insertion."""
        if name in self._index:
            raise self._duplicate_error(name)
        undo = self._add(name, value)
        self._index[name] = self._generation.records[-1]
        return undo

    def _removed(self, record: _Record[V]) -> None:
        if self._index.get(str(record.key)) is record:
            del self._index[str(record.key)]

    def get(self, name: str) -> V | None:
        """Read one named value, or ``None`` when absent."""
        record = self._index.get(name)
        return None if record is None else record.value

    def has(self, name: str) -> bool:
        """Test one name for membership."""
        return name in self._index

    def keys(self) -> Iterator[str]:
        """Iterate live names in insertion order."""
        return self._iterate(lambda record: str(record.key))

    def entries(self) -> Iterator[tuple[str, V]]:
        """Iterate live ``(name, value)`` pairs in insertion order."""
        return self._iterate(lambda record: (str(record.key), record.value))


class AnonymousEntries(_Table[V]):
    """Insertion-ordered anonymous entries with independent registration identity.

    Equal values remain separate registrations. Values are borrowed, and iterators
    are live within one nonempty table generation; draining the table detaches them
    from later appends.
    """

    def append(self, value: V) -> Callable[[], None]:
        """Append one independently owned value; return an idempotent undo."""
        return self._add(object(), value)


class ScopedLayers(Generic[L]):
    """Own the global and exact-scope layers for one registry.

    Reads never create scoped layers. Registrations derive both visibility and effect
    ownership from the supplied Cordis context, collect undo before notification, and
    reclaim only a completely empty aggregate layer.
    """

    def __init__(
        self,
        create_layer: Callable[[ScopeKey | None], L],
        on_change: Callable[[], None],
    ) -> None:
        self._create_layer = create_layer
        self._on_change = on_change
        self._scoped: IdentityMap[ScopeKey, L] = IdentityMap()
        self.global_: L = create_layer(None)
        """The eagerly constructed context-global layer (upstream ``global``)."""

    def peek(self, scope: ScopeKey | None) -> L | None:
        """Read an existing exact-scope overlay without creating one.

        Deliberately chain-blind: callers addressing one scope's own contributions
        must not pick up an ancestor's. Use :meth:`chain_layers` for inheritance.
        """
        if scope is None:
            return None
        return self._scoped.get(scope)

    def chain_layers(self, scope: ScopeKey | None) -> list[L]:
        """Existing overlays along the scope chain, farthest ancestor first."""
        from . import scope_chain_of

        layers: list[L] = []
        for key in reversed(scope_chain_of(scope)):
            layer = self._scoped.get(key)
            if layer is not None:
                layers.append(layer)
        return layers

    def merge(self, scope: ScopeKey | None, pick: Callable[[L], NamedEntries[V]]) -> dict[str, V]:
        """Global named entries, then scope-chain shadows; the nearest scope wins a name."""
        merged = dict(pick(self.global_).entries())
        for layer in self.chain_layers(scope):
            for name, value in pick(layer).entries():
                merged[name] = value
        return merged

    def effect(
        self,
        ctx: Context,
        action: Callable[[L], Callable[[], None]],
        *,
        label: str,
        notify: bool = True,
    ) -> Effect:
        """Attach one synchronous layer mutation to its registration context.

        ``ctx`` decides both scope visibility and effect ownership. Returns the exact
        disposer ``ctx.effect()`` returned.
        """
        from . import scope_of

        scope = scope_of(ctx)

        def setup() -> Iterator[Callable[[], None]]:
            created = False
            if scope is None:
                layer = self.global_
            else:
                existing = self._scoped.get(scope)
                if existing is None:
                    layer = self._create_layer(scope)
                    self._scoped.set(scope, layer)
                    created = True
                else:
                    layer = existing
            try:
                undo = action(layer)
            except BaseException:
                if scope is not None and created and layer.is_empty():
                    self._scoped.delete(scope)
                raise

            def cleanup() -> None:
                undo()
                if scope is not None and layer.is_empty():
                    self._scoped.delete(scope)
                if notify:
                    self._on_change()

            yield cleanup
            if notify:
                self._on_change()

        return ctx.effect(setup, label)
