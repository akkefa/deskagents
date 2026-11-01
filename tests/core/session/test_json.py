"""Ported from packages/core/session/tests/json.spec.ts."""

from __future__ import annotations

import math
import types
from typing import Any

import pytest

from deskagents.util.values import MAX_SAFE_INTEGER, JsonValue, is_json_value, snapshot_json_value

SPEC = "packages/core/session/tests/json.spec.ts"
BOTH = "snapshotJsonValue and isJsonValue"
PREDICATE = "isJsonValue"


def unsupported_function() -> None:
    return None


class ExoticObject:
    """A class instance: upstream's exotic object."""

    def __init__(self) -> None:
        self.value = 1


class ExoticArray(list[int]):
    """A list subclass: upstream's ``class ExoticArray extends Array``."""


class CustomPrototype(dict[str, Any]):
    """A dict subclass: upstream's object with a custom prototype."""


def _cyclic() -> dict[str, Any]:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    return cyclic


INVALID_IDS = [
    "class-instance",
    "mapping-proxy",
    "set",
    "list-subclass",
    "non-string-key",
    "dict-subclass",
    "cycle",
    "array-non-json-member",
    "object-non-json-member",
]


def invalid_values() -> list[object]:
    """Python counterparts of upstream's exotic containers and invalid children."""
    return [
        ExoticObject(),
        types.MappingProxyType({"value": 1}),  # a non-plain mapping (upstream Map)
        {"value"},  # a set
        ExoticArray([1]),  # upstream ExoticArray / forged array prototype
        {1: True},  # a non-string key (upstream symbol key)
        CustomPrototype(value=1),  # upstream custom or forged prototype
        _cyclic(),
        [object()],  # a non-JSON member (upstream [undefined])
        {"value": object()},  # a non-JSON member (upstream { value: undefined })
    ]


@pytest.mark.upstream(
    f"{SPEC}::{BOTH} > copies the complete JSON scalar vocabulary and rejects unsupported scalars"
)
def test_copies_json_scalars_and_rejects_unsupported_scalars() -> None:
    # Python's snapshot returns None both for JSON null and for a rejection.
    assert snapshot_json_value(None) is None
    assert is_json_value(None)
    assert snapshot_json_value(True) is True
    assert snapshot_json_value("text") == "text"
    assert snapshot_json_value(1.25) == 1.25
    assert snapshot_json_value(-0.0) is None
    assert not is_json_value(-0.0)
    assert snapshot_json_value(math.nan) is None
    assert snapshot_json_value(math.inf) is None
    assert snapshot_json_value(MAX_SAFE_INTEGER + 1) is None
    assert snapshot_json_value(unsupported_function) is None
    assert snapshot_json_value(object()) is None


@pytest.mark.upstream(
    f"{SPEC}::{BOTH} > recursively detaches dense arrays and plain or null-prototype objects"
)
def test_recursively_detaches_arrays_and_objects() -> None:
    shared: dict[str, Any] = {"value": 1}
    record: dict[str, Any] = {"shared": shared}
    source: dict[str, Any] = {"list": [record, shared], "alias": shared}

    snapshot = snapshot_json_value(source)
    assert snapshot is not None
    shared["value"] = 2

    assert snapshot == {
        "list": [{"shared": {"value": 1}}, {"value": 1}],
        "alias": {"value": 1},
    }
    assert snapshot is not source
    assert snapshot["list"] is not source["list"]
    assert snapshot["alias"] is not shared
    assert snapshot["list"][0] is not record
    assert type(snapshot["list"][0]) is dict


@pytest.mark.upstream(
    f"{SPEC}::{BOTH} > detaches every occurrence of a shared child in a wide array"
)
def test_detaches_every_occurrence_of_a_shared_child() -> None:
    shared: dict[str, Any] = {"nested": [{"value": 1}]}
    source = [shared for _ in range(4_096)]

    assert is_json_value(source)
    snapshot = snapshot_json_value(source)
    assert snapshot is not None
    assert snapshot == source
    shared["nested"][0]["value"] = 2
    snapshot[0]["nested"][0]["value"] = 3

    assert source[-1]["nested"][0]["value"] == 2
    assert snapshot[0]["nested"][0]["value"] == 3
    assert snapshot[-1]["nested"][0]["value"] == 1


@pytest.mark.upstream(
    f"{SPEC}::{BOTH} > accepts deeply nested valid JSON without using the JavaScript call stack"
)
def test_accepts_deeply_nested_json_without_recursion() -> None:
    value: JsonValue = "leaf"
    for _ in range(5_000):
        value = [value]

    assert is_json_value(value)
    cursor: JsonValue | None = snapshot_json_value(value)
    for _ in range(5_000):
        assert isinstance(cursor, list)
        cursor = cursor[0]
    assert cursor == "leaf"


@pytest.mark.upstream(
    f"{SPEC}::{BOTH} > rejects exotic containers, sparse or decorated arrays, cycles, and "
    "invalid children"
)
@pytest.mark.parametrize("value", invalid_values(), ids=INVALID_IDS)
def test_snapshot_rejects_exotic_containers_cycles_and_invalid_children(value: object) -> None:
    assert snapshot_json_value(value) is None


@pytest.mark.upstream(
    f"{SPEC}::{BOTH} > preserves a literal __proto__ JSON key without changing the snapshot "
    "prototype"
)
def test_preserves_literal_proto_key() -> None:
    source: dict[str, Any] = {"__proto__": {"safe": True}}

    snapshot = snapshot_json_value(source)

    assert snapshot is not None
    assert type(snapshot) is dict
    assert "__proto__" in snapshot
    assert snapshot["__proto__"] == {"safe": True}


@pytest.mark.upstream(
    f"{SPEC}::{PREDICATE} > recognizes supported scalars and rejects every lossy scalar case"
)
def test_recognizes_supported_scalars_and_rejects_lossy_scalars() -> None:
    assert is_json_value(None)
    assert is_json_value(False)
    assert is_json_value("text")
    assert is_json_value(1.25)
    assert not is_json_value(-0.0)
    assert not is_json_value(math.nan)
    assert not is_json_value(MAX_SAFE_INTEGER + 1)
    assert not is_json_value(unsupported_function)
    assert not is_json_value(object())


@pytest.mark.upstream(
    f"{SPEC}::{PREDICATE} > accepts dense arrays and plain objects, including null-prototype "
    "records"
)
def test_accepts_dense_arrays_and_plain_objects() -> None:
    record = {"value": True}

    assert is_json_value([1, {"nested": None}, record])
    assert is_json_value({"value": [1, 2]})
    assert is_json_value(record)


@pytest.mark.upstream(
    f"{SPEC}::{PREDICATE} > rejects sparse or decorated arrays, invalid children, exotic "
    "objects, and cycles"
)
@pytest.mark.parametrize("value", invalid_values(), ids=INVALID_IDS)
def test_predicate_rejects_exotic_objects_cycles_and_invalid_children(value: object) -> None:
    assert not is_json_value(value)
