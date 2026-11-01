"""JavaScript value semantics the v4 validators rely on.

Upstream reads absent members as ``undefined``, compares with ``===`` and formats
diagnostics with ``String(value)``. Python ``dict.get`` folds an absent member into
JSON ``null`` and ``==`` lets ``True`` equal ``1``, so these helpers keep the two
runtimes' decisions identical.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Final

from deskagents.util.values import format_json_number, stringify_json


class _Undefined:
    """The JavaScript ``undefined`` value: an absent member, distinct from JSON ``null``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "undefined"

    def __bool__(self) -> bool:
        return False


UNDEFINED: Final = _Undefined()


def prop(value: object, key: str) -> object:
    """``value[key]``: ``UNDEFINED`` when absent; raises like JavaScript on ``null``."""
    if isinstance(value, Mapping):
        return value[key] if key in value else UNDEFINED
    if value is None or value is UNDEFINED:
        raise TypeError(f"Cannot read properties of {value!r} (reading '{key}')")
    return UNDEFINED


def opt_prop(value: object, key: str) -> object:
    """``value?.[key]``."""
    if value is None or value is UNDEFINED:
        return UNDEFINED
    return prop(value, key)


def at(values: Sequence[object], index: int) -> object:
    """``values[index]``: ``UNDEFINED`` outside the array."""
    return values[index] if 0 <= index < len(values) else UNDEFINED


def is_number(value: object) -> bool:
    """``typeof value === 'number'``."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def strict_equal(left: object, right: object) -> bool:
    """``left === right`` for JSON values; containers compare by identity."""
    if is_number(left) and is_number(right):
        return left == right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if left is None or right is None or left is UNDEFINED or right is UNDEFINED:
        return left is right
    if isinstance(left, Mapping | list | tuple) or isinstance(right, Mapping | list | tuple):
        return left is right
    return False


def includes(values: Sequence[object], value: object) -> bool:
    """``values.includes(value)`` (SameValueZero)."""
    if isinstance(value, float) and math.isnan(value):
        return any(isinstance(item, float) and math.isnan(item) for item in values)
    return any(strict_equal(item, value) for item in values)


def has(values: Collection[str], value: object) -> bool:
    """``set.has(value)`` for a set of strings."""
    return isinstance(value, str) and value in values


def is_array(value: object) -> bool:
    """``Array.isArray(value)``."""
    return isinstance(value, list | tuple)


def deep_strict_equal(left: object, right: object) -> bool:
    """``util.isDeepStrictEqual`` for JSON values (``-0`` differs from ``0``)."""
    if is_number(left) and is_number(right):
        if left == 0 and right == 0:
            return math.copysign(1.0, left) == math.copysign(1.0, right)
        return left == right or (left != left and right != right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            deep_strict_equal(left[key], right[key]) for key in left
        )
    if is_array(left) and is_array(right):
        assert isinstance(left, list | tuple) and isinstance(right, list | tuple)
        return len(left) == len(right) and all(
            deep_strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping | list | tuple) or isinstance(right, Mapping | list | tuple):
        return False
    return strict_equal(left, right)


def to_number(value: object) -> float:
    """``Number(value)`` for the primitive operands relational comparisons coerce."""
    if value is None:
        return 0.0
    if value is UNDEFINED:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if is_number(value):
        return float(value)  # type: ignore[arg-type]
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0.0
        try:
            return float(int(text, 0)) if text[:2].lower() in ("0x", "0o", "0b") else float(text)
        except ValueError:
            return math.nan
    return math.nan


def greater_or_equal(left: object, right: object) -> bool:
    """``left >= right`` with JavaScript numeric coercion; false whenever NaN appears."""
    if is_number(left) and is_number(right):
        return bool(left >= right)  # type: ignore[operator]
    return to_number(left) >= to_number(right)


def js_string(value: object) -> str:
    """``String(value)`` for JSON values."""
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if is_number(value):
        if isinstance(value, float) and not math.isfinite(value):
            return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return format_json_number(value)  # type: ignore[arg-type]
    if isinstance(value, list | tuple):
        return ",".join(
            "" if item is None or item is UNDEFINED else js_string(item) for item in value
        )
    if isinstance(value, Mapping):
        return "[object Object]"
    return str(value)


def json_text(value: object) -> str:
    """``JSON.stringify(value)`` as interpolated into a template literal."""
    if value is UNDEFINED:
        return "undefined"
    try:
        return stringify_json(value)
    except ValueError:
        return js_string(value)
