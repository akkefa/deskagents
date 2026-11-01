"""Lossless JSON and immutable-value helpers.

Upstream: packages/util/values (``@deepseek-ai/dsh-util-values``) at ce7073d. Ported so
far: ``JsonValue``, ``assertNever``, ``snapshotJsonValue``, ``isJsonValue``,
``deepEqualJson`` and ``deepFreeze``. ``WeakMapWithValues`` and the partial-JSON
reader follow with the packages that need them.

Python differences:

- "Lossless JSON" means lossless for both runtimes. Upstream numbers are doubles, so an
  ``int`` outside ``±(2**53 - 1)`` is refused, like a non-finite number or ``-0.0``.
  Only ``dict`` with ``str`` keys and ``list``/``tuple`` containers are accepted;
  upstream likewise refuses class instances and custom prototypes.
- Upstream freezes objects in place. Python dicts and lists can't be frozen, so
  :func:`deep_freeze` returns :class:`FrozenJsonObject` / :class:`FrozenJsonArray`
  copies. They subclass ``dict`` and ``list``, compare equal to plain values and
  serialize with :mod:`json`, but refuse mutation.
- :func:`stringify_json` and :func:`parse_json` reproduce ``JSON.stringify`` and
  ``JSON.parse`` exactly (number formatting, lone-surrogate escapes, no ``NaN``), so
  files written here are byte-identical to the ones ``dsh`` writes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn, SupportsIndex, TypeAlias, TypeVar, cast, overload

__all__ = [
    "MAX_SAFE_INTEGER",
    "FrozenJsonArray",
    "FrozenJsonObject",
    "JsonValue",
    "assert_never",
    "deep_equal_json",
    "deep_freeze",
    "format_json_number",
    "is_json_value",
    "is_safe_integer",
    "parse_json",
    "snapshot_json_value",
    "stringify_json",
    "thaw_json",
]

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
"""A value that round-trips through JSON without loss (frozen forms included)."""

MAX_SAFE_INTEGER = 2**53 - 1
"""JavaScript ``Number.MAX_SAFE_INTEGER``."""

T = TypeVar("T")


def assert_never(value: NoReturn, context: str | None = None) -> NoReturn:
    """Mark an unreachable closed-union branch; always raises."""
    try:
        rendered = json.dumps(value)
    except (TypeError, ValueError):
        rendered = repr(value)
    where = f" in {context}" if context else ""
    raise AssertionError(f"unreachable variant{where}: {rendered}")


def is_safe_integer(value: object) -> bool:
    """``Number.isSafeInteger``: an integral number within ±(2**53 - 1), never a bool."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER
    if isinstance(value, float):
        return value.is_integer() and abs(value) <= MAX_SAFE_INTEGER
    return False


def _frozen(kind: str) -> NoReturn:
    raise TypeError(f"frozen JSON {kind} is immutable")


class FrozenJsonObject(dict[str, JsonValue]):
    """An immutable JSON object. Built by :func:`deep_freeze` and :func:`snapshot_json_value`."""

    __slots__ = ()

    def __setitem__(self, key: str, value: JsonValue) -> None:
        _frozen("object")

    def __delitem__(self, key: str) -> None:
        _frozen("object")

    def __ior__(self, other: object) -> FrozenJsonObject:  # type: ignore[override,misc]
        _frozen("object")

    def clear(self) -> None:
        _frozen("object")

    def pop(self, *args: object) -> JsonValue:
        _frozen("object")

    def popitem(self) -> tuple[str, JsonValue]:
        _frozen("object")

    def setdefault(self, *args: object) -> JsonValue:  # type: ignore[override]
        _frozen("object")

    def update(self, *args: object, **kwargs: object) -> None:
        _frozen("object")

    def __copy__(self) -> FrozenJsonObject:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenJsonObject:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore_object, (list(self.items()),))


class FrozenJsonArray(list[JsonValue]):
    """An immutable JSON array. Built by :func:`deep_freeze` and :func:`snapshot_json_value`."""

    __slots__ = ()

    def __setitem__(self, index: Any, value: Any) -> None:
        _frozen("array")

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        _frozen("array")

    def __iadd__(self, other: Iterable[JsonValue]) -> FrozenJsonArray:  # type: ignore[override,misc]
        _frozen("array")

    def __imul__(self, count: SupportsIndex) -> FrozenJsonArray:
        _frozen("array")

    def append(self, value: JsonValue) -> None:
        _frozen("array")

    def extend(self, values: Iterable[JsonValue]) -> None:
        _frozen("array")

    def insert(self, index: SupportsIndex, value: JsonValue) -> None:
        _frozen("array")

    def pop(self, index: SupportsIndex = -1) -> JsonValue:
        _frozen("array")

    def remove(self, value: JsonValue) -> None:
        _frozen("array")

    def clear(self) -> None:
        _frozen("array")

    def sort(self, *args: Any, **kwargs: Any) -> None:
        _frozen("array")

    def reverse(self) -> None:
        _frozen("array")

    def __copy__(self) -> FrozenJsonArray:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenJsonArray:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore_array, (list(self),))


def _restore_object(items: list[tuple[str, JsonValue]]) -> FrozenJsonObject:
    return FrozenJsonObject(items)


def _restore_array(items: list[JsonValue]) -> FrozenJsonArray:
    return FrozenJsonArray(items)


_OBJECTS = (dict, FrozenJsonObject)
_ARRAYS = (list, tuple, FrozenJsonArray)


class _Invalid(Exception):
    pass


def _walk(value: object, detach: bool, freeze: bool) -> JsonValue:
    """Validate lossless JSON iteratively, optionally building a detached copy.

    Iterative like upstream, so deep values don't hit the recursion limit.
    """
    ancestors: set[int] = set()
    result: list[JsonValue] = []
    # Each frame: (source id, items iterator, parts, is_object, parent parts).
    stack: list[tuple[int, Any, list[Any], bool]] = []

    def scalar(item: object) -> JsonValue:
        if item is None or isinstance(item, bool | str):
            return item
        if isinstance(item, int):
            if not -MAX_SAFE_INTEGER <= item <= MAX_SAFE_INTEGER:
                raise _Invalid
            return item
        if isinstance(item, float):
            if not math.isfinite(item) or (item == 0.0 and math.copysign(1.0, item) < 0):
                raise _Invalid
            return item
        raise _Invalid

    def push(item: object) -> bool:
        """Open a container frame; return False for a scalar."""
        if type(item) in _OBJECTS:
            assert isinstance(item, dict)
            if id(item) in ancestors or any(not isinstance(key, str) for key in item):
                raise _Invalid
            ancestors.add(id(item))
            stack.append((id(item), iter(list(item.items())), [], True))
            return True
        if type(item) in _ARRAYS:
            assert isinstance(item, list | tuple)
            if id(item) in ancestors:
                raise _Invalid
            ancestors.add(id(item))
            stack.append((id(item), iter(list(item)), [], False))
            return True
        return False

    def deliver(item: JsonValue, key: str | None) -> None:
        if stack:
            parts = stack[-1][2]
            parts.append((key, item) if stack[-1][3] else item)
        else:
            result.append(item)

    def open_or_scalar(item: object, key: str | None) -> None:
        if not push(item):
            deliver(scalar(item), key)
        else:
            pending_keys.append(key)

    pending_keys: list[str | None] = []
    open_or_scalar(value, None)
    while stack:
        source, items, parts, is_object = stack[-1]
        try:
            entry = next(items)
        except StopIteration:
            stack.pop()
            ancestors.discard(source)
            key = pending_keys.pop()
            built: JsonValue
            if not detach:
                built = None
            elif is_object:
                built = FrozenJsonObject(parts) if freeze else dict(parts)
            else:
                built = FrozenJsonArray(parts) if freeze else list(parts)
            deliver(built, key)
            continue
        if is_object:
            child_key, child = entry
            open_or_scalar(child, child_key)
        else:
            open_or_scalar(entry, None)
    return result[0]


def snapshot_json_value(value: T, *, freeze: bool = False) -> T | None:
    """Validate and detach lossless JSON; ``None`` when ``value`` is not lossless JSON.

    ``freeze=True`` builds the copy from :class:`FrozenJsonObject` and
    :class:`FrozenJsonArray` in the same pass. Note that the JSON value ``null`` also
    snapshots to ``None``; use :func:`is_json_value` to tell the cases apart.
    """
    try:
        return _walk(value, True, freeze)  # type: ignore[return-value]
    except _Invalid:
        return None


def is_json_value(value: object) -> bool:
    """Test the same lossless JSON rules as :func:`snapshot_json_value` without copying."""
    try:
        _walk(value, False, False)
    except _Invalid:
        return False
    return True


def deep_equal_json(a: object, b: object) -> bool:
    """Compare JSON values structurally; ``true`` never equals ``1``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(deep_equal_json(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(deep_equal_json(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping | list | tuple) or isinstance(b, Mapping | list | tuple):
        return False
    return a is b or a == b


@overload
def deep_freeze(value: dict[str, JsonValue]) -> FrozenJsonObject: ...
@overload
def deep_freeze(value: list[JsonValue]) -> FrozenJsonArray: ...
@overload
def deep_freeze(value: T) -> T: ...
def deep_freeze(value: object) -> object:
    """Return a frozen copy of a JSON-shaped graph; scalars and frozen values return as is.

    Unlike :func:`snapshot_json_value` this does not validate: non-JSON leaves are kept.
    """
    memo: dict[int, object] = {}

    def freeze(node: object) -> object:
        if isinstance(node, FrozenJsonObject | FrozenJsonArray):
            return node
        if type(node) is dict:
            if id(node) not in memo:
                memo[id(node)] = FrozenJsonObject(
                    (k, cast(JsonValue, freeze(v))) for k, v in node.items()
                )
            return memo[id(node)]
        if type(node) in (list, tuple):
            assert isinstance(node, list | tuple)
            if id(node) not in memo:
                memo[id(node)] = FrozenJsonArray(cast(JsonValue, freeze(v)) for v in node)
            return memo[id(node)]
        return node

    return freeze(value)


def thaw_json(value: T) -> T:
    """Return a plain, mutable deep copy of a (possibly frozen) JSON value."""
    if isinstance(value, Mapping):
        return {k: thaw_json(v) for k, v in value.items()}  # type: ignore[return-value]
    if isinstance(value, list | tuple):
        return [thaw_json(v) for v in value]  # type: ignore[return-value]
    return value


def format_json_number(value: int | float) -> str:
    """Format a finite number exactly as JavaScript's ``Number.prototype.toString``."""
    if isinstance(value, bool):
        raise TypeError("booleans are not JSON numbers")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    # repr() gives the shortest round-tripping digits, the same digits JavaScript picks.
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    leading_zeros = len(whole + fraction) - len(digits)
    point = len(whole) - leading_zeros + (int(exponent) if exponent else 0)
    digits = digits.rstrip("0") or "0"
    k, n = len(digits), point
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        body = "0." + "0" * -n + digits
    else:
        exp = n - 1
        mantissa_text = digits[0] + (f".{digits[1:]}" if k > 1 else "")
        body = f"{mantissa_text}e{'+' if exp >= 0 else '-'}{abs(exp)}"
    return sign + body


_SURROGATES = re.compile(r"[\ud800-\udbff][\udc00-\udfff]|[\ud800-\udfff]")


def _surrogate(match: re.Match[str]) -> str:
    text = match.group()
    if len(text) == 2:
        high, low = ord(text[0]), ord(text[1])
        return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    return f"\\u{ord(text):04x}"


def _string(value: str) -> str:
    # json.dumps escapes the same characters JSON.stringify does (quote, backslash, and
    # C0 controls with the same short forms and lowercase \u00XX); it leaves lone
    # surrogates raw, which JSON.stringify escapes. A Python surrogate pair is one
    # JavaScript astral character, which JSON.stringify emits as is.
    encoded = json.dumps(value, ensure_ascii=False)
    return _SURROGATES.sub(_surrogate, encoded) if _SURROGATES.search(encoded) else encoded


def stringify_json(value: object) -> str:
    """``JSON.stringify(value)`` for lossless JSON; raises ``ValueError`` otherwise."""
    if not is_json_value(value):
        raise ValueError("value is not lossless JSON")
    out: list[str] = []
    # Explicit stack of (kind, iterator) frames, mirroring the iterative validation.
    stack: list[tuple[bool, Any, bool]] = []

    def emit(item: object) -> None:
        if item is None:
            out.append("null")
        elif item is True:
            out.append("true")
        elif item is False:
            out.append("false")
        elif isinstance(item, str):
            out.append(_string(item))
        elif isinstance(item, int | float):
            out.append(format_json_number(item))
        elif isinstance(item, dict):
            out.append("{")
            stack.append((True, iter(item.items()), True))
        else:
            assert isinstance(item, list | tuple)
            out.append("[")
            stack.append((False, iter(item), True))

    emit(value)
    while stack:
        is_object, items, first = stack[-1]
        try:
            entry = next(items)
        except StopIteration:
            stack.pop()
            out.append("}" if is_object else "]")
            continue
        if not first:
            out.append(",")
        else:
            stack[-1] = (is_object, items, False)
        if is_object:
            key, child = entry
            out.append(_string(key))
            out.append(":")
            emit(child)
        else:
            emit(entry)
    return "".join(out)


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"JSON does not allow {name}")


def parse_json(text: str) -> JsonValue:
    """``JSON.parse(text)``: rejects ``NaN`` and ``Infinity``, which :mod:`json` accepts."""
    return json.loads(text, parse_constant=_reject_constant)  # type: ignore[no-any-return]
