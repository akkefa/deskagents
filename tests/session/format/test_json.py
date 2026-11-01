"""Ported from packages/session/session-format/tests/json.spec.ts."""

from __future__ import annotations

import pytest

from deskagents.session.format import (
    SessionFormatError,
    inspect_session_format_version,
    session_format_count,
    session_format_safe_integer,
    snapshot_session_format_header,
    snapshot_session_format_json,
)
from deskagents.util.values import FrozenJsonArray, FrozenJsonObject

SPEC = "packages/session/session-format/tests/json.spec.ts"
GROUP = "lossless Session format JSON snapshots"


class RecordValue:
    value = 1


class ArrayValue(list[int]):
    pass


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses %s that JSON cannot preserve")
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-0.0, id="negative zero"),
        pytest.param(float("inf"), id="non-finite number"),
        pytest.param({"value": object()}, id="non-JSON member"),
        pytest.param({1: True}, id="non-string key"),
        pytest.param(2**53, id="unsafe integer"),
        pytest.param({"set": {1}}, id="set member"),
    ],
)
def test_refuses_values_json_cannot_preserve(value: object) -> None:
    with pytest.raises(SessionFormatError, match="payload is not lossless JSON"):
        snapshot_session_format_json(value, "payload")


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > detaches, freezes, and retains repeated non-cyclic values and "
    "__proto__ keys"
)
def test_detaches_freezes_and_retains_repeats() -> None:
    shared = {"value": 1}
    source: dict[str, object] = {"__proto__": {"safe": True}, "values": [shared, shared]}

    snapshot = snapshot_session_format_json(source)

    assert snapshot == source
    assert snapshot is not source
    assert isinstance(snapshot, FrozenJsonObject)
    assert isinstance(snapshot["values"], FrozenJsonArray)
    with pytest.raises(TypeError, match="immutable"):
        snapshot["values"] = []
    assert type(snapshot) is FrozenJsonObject and isinstance(snapshot, dict)


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > refuses invalid scalar coordinates, cycles, and custom prototypes"
)
def test_refuses_bad_coordinates_cycles_and_custom_types() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    with pytest.raises(SessionFormatError, match="non-negative"):
        session_format_count(-1, "count")
    with pytest.raises(SessionFormatError, match="safe integer"):
        session_format_safe_integer(1.5, "integer")
    with pytest.raises(SessionFormatError, match="header"):
        inspect_session_format_version([])
    with pytest.raises(SessionFormatError, match="not lossless JSON"):
        snapshot_session_format_json(cyclic)
    with pytest.raises(SessionFormatError, match="not lossless JSON"):
        snapshot_session_format_json(RecordValue())
    with pytest.raises(SessionFormatError, match="not lossless JSON"):
        snapshot_session_format_json(ArrayValue([1]))


@pytest.mark.upstream(f"{SPEC}::{GROUP} > refuses a non-object header snapshot")
def test_refuses_non_object_header() -> None:
    with pytest.raises(SessionFormatError, match="header|object"):
        snapshot_session_format_header(None)
    with pytest.raises(SessionFormatError, match="isSeeded"):
        snapshot_session_format_header(
            {"version": 1, "id": "missing-seeded", "createdAt": 1, "delegationDepth": 0}
        )


def test_counts_accept_integral_floats_and_reject_booleans() -> None:
    assert session_format_count(3.0, "count") == 3
    assert isinstance(session_format_count(3.0, "count"), int)
    with pytest.raises(SessionFormatError):
        session_format_count(True, "count")
    with pytest.raises(SessionFormatError):
        session_format_count(-0.0, "count")
