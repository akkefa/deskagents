"""Identity-keyed maps: scope keys compare by identity, never by ``__eq__``."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class IdentityMap(Generic[K, V]):
    """A strong map keyed by object identity, preserving insertion order."""

    def __init__(self) -> None:
        self._data: dict[int, tuple[K, V]] = {}

    def get(self, key: K) -> V | None:
        entry = self._data.get(id(key))
        return None if entry is None else entry[1]

    def set(self, key: K, value: V) -> None:
        self._data[id(key)] = (key, value)

    def delete(self, key: K) -> None:
        self._data.pop(id(key), None)

    def __contains__(self, key: object) -> bool:
        return id(key) in self._data

    def __len__(self) -> int:
        return len(self._data)


class WeakIdentityMap(Generic[K, V]):
    """A map keyed weakly by object identity, like a JavaScript ``WeakMap``.

    Keys must support weak references; an entry disappears with its key.
    """

    def __init__(self) -> None:
        self._data: dict[int, tuple[weakref.ref[K], V]] = {}

    def _ref(self, key: K) -> weakref.ref[K]:
        ident = id(key)
        data = self._data

        def drop(ref: weakref.ref[K], ident: int = ident) -> None:
            entry = data.get(ident)
            if entry is not None and entry[0] is ref:
                del data[ident]

        try:
            return weakref.ref(key, drop)
        except TypeError:
            raise TypeError(
                f"scope keys must support weak references; got {type(key).__name__}"
            ) from None

    def get(self, key: K) -> V | None:
        entry = self._data.get(id(key))
        if entry is None or entry[0]() is not key:
            return None
        return entry[1]

    def set(self, key: K, value: V) -> None:
        entry = self._data.get(id(key))
        ref = entry[0] if entry is not None and entry[0]() is key else self._ref(key)
        self._data[id(key)] = (ref, value)

    def __contains__(self, key: object) -> bool:
        entry = self._data.get(id(key))
        return entry is not None and entry[0]() is key


def walk(start: K | None, parent: Callable[[K], K | None]) -> Iterator[K]:
    """Yield ``start`` and each ancestor reached through ``parent``."""
    cursor = start
    while cursor is not None:
        yield cursor
        cursor = parent(cursor)
