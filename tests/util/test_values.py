"""deskagents.util.values: lossless JSON rules and JSON.stringify/JSON.parse parity.

tests/fixtures/js_json_numbers.json was produced by Node's JSON.stringify, so number
formatting is checked against JavaScript without needing Node in CI.
"""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable
from pathlib import Path

import pytest

from deskagents.util.values import (
    MAX_SAFE_INTEGER,
    FrozenJsonArray,
    FrozenJsonObject,
    assert_never,
    deep_equal_json,
    deep_freeze,
    format_json_number,
    is_json_value,
    is_safe_integer,
    parse_json,
    snapshot_json_value,
    stringify_json,
    thaw_json,
)

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "js_json_numbers.json").read_text())


@pytest.mark.parametrize(
    "case", FIXTURE["numbers"], ids=[case["repr"] for case in FIXTURE["numbers"]]
)
def test_number_formatting_matches_javascript(case: dict[str, str]) -> None:
    assert format_json_number(float(case["repr"])) == case["js"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a b", '"a b"'),
        ("\ud800", '"\\ud800"'),
        ("x\udc00y", '"x\\udc00y"'),
        ("😀", '"\U0001f600"'),
        ("\x00\x1f\x7f", '"\\u0000\\u001f\x7f"'),
        ('tab\t"q"\\', '"tab\\t\\"q\\"\\\\"'),
        ("é", '"é"'),
    ],
)
def test_strings_match_javascript(value: str, expected: str) -> None:
    assert stringify_json(value) == expected


def test_containers_match_javascript() -> None:
    value = {"a": [1, 2.5, None, True, {"k": "v"}], "b": -0.5e-10, "c": 1.0}
    assert stringify_json(value) == '{"a":[1,2.5,null,true,{"k":"v"}],"b":-5e-11,"c":1}'


def test_stringify_refuses_lossy_values() -> None:
    for value in [float("nan"), -0.0, 2**53, {1: 2}, object(), {"x": {1, 2}}]:
        with pytest.raises(ValueError, match="lossless JSON"):
            stringify_json(value)


def test_parse_rejects_non_standard_constants() -> None:
    assert parse_json('{"a":[1,2.5,null]}') == {"a": [1, 2.5, None]}
    for text in ["NaN", "[Infinity]", '{"a":-Infinity}']:
        with pytest.raises(ValueError, match="does not allow"):
            parse_json(text)


def test_deep_structures_do_not_hit_the_recursion_limit() -> None:
    value: object = 1
    for _ in range(10_000):
        value = [value]
    assert is_json_value(value)
    text = stringify_json(value)
    assert text == "[" * 10_000 + "1" + "]" * 10_000
    snapshot = snapshot_json_value(value, freeze=True)
    assert isinstance(snapshot, FrozenJsonArray)


def test_safe_integers() -> None:
    assert is_safe_integer(MAX_SAFE_INTEGER) and is_safe_integer(-MAX_SAFE_INTEGER)
    assert not is_safe_integer(MAX_SAFE_INTEGER + 1)
    assert is_safe_integer(3.0) and not is_safe_integer(3.5)
    assert not is_safe_integer(True)


def test_snapshot_detaches_and_rejects_cycles() -> None:
    inner = {"x": 1}
    source = {"a": inner, "b": [inner]}
    snapshot = snapshot_json_value(source)
    assert snapshot == source and snapshot is not source
    assert snapshot is not None and snapshot["a"] is not inner
    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert snapshot_json_value(cyclic) is None
    assert not is_json_value(cyclic)
    pair: object = (1, 2)
    assert snapshot_json_value(pair) == [1, 2]


def test_frozen_values_refuse_every_mutation() -> None:
    frozen = deep_freeze({"a": [1, {"b": 2}]})
    array = frozen["a"]
    assert isinstance(frozen, FrozenJsonObject) and isinstance(array, FrozenJsonArray)
    mutations: list[Callable[[], object]] = [
        lambda: frozen.__setitem__("x", 1),
        lambda: frozen.__delitem__("a"),
        lambda: frozen.update(x=1),
        lambda: frozen.setdefault("x", 1),
        lambda: frozen.pop("a"),
        lambda: frozen.popitem(),
        lambda: frozen.clear(),
        lambda: array.append(1),
        lambda: array.extend([1]),
        lambda: array.insert(0, 1),
        lambda: array.pop(),
        lambda: array.remove(1),
        lambda: array.clear(),
        lambda: array.sort(),
        lambda: array.reverse(),
        lambda: array.__setitem__(0, 1),
        lambda: array.__delitem__(0),
    ]
    for mutate in mutations:
        with pytest.raises(TypeError, match="immutable"):
            mutate()
    with pytest.raises(TypeError, match="immutable"):
        frozen |= {"x": 1}
    assert frozen == {"a": [1, {"b": 2}]}


def test_frozen_values_serialize_copy_and_pickle() -> None:
    frozen = deep_freeze({"a": [1, {"b": 2}]})
    assert json.dumps(frozen) == '{"a": [1, {"b": 2}]}'
    assert copy.copy(frozen) is frozen and copy.deepcopy(frozen) is frozen
    restored = pickle.loads(pickle.dumps(frozen))
    assert restored == frozen and isinstance(restored, FrozenJsonObject)
    thawed = thaw_json(frozen)
    assert type(thawed) is dict and type(thawed["a"]) is list
    thawed["a"].append(3)
    assert frozen["a"] == [1, {"b": 2}]


def test_deep_equal_json_distinguishes_booleans_from_numbers() -> None:
    assert deep_equal_json({"a": [1, 2.0]}, {"a": [1.0, 2]})
    assert not deep_equal_json([True], [1])
    assert not deep_equal_json({"a": 1}, {"a": 1, "b": 2})
    assert not deep_equal_json([1], {"0": 1})


def test_assert_never_names_the_variant() -> None:
    with pytest.raises(AssertionError, match='unreachable variant in switch: "odd"'):
        assert_never("odd", "switch")  # type: ignore[arg-type]
