"""Ported from packages/session/session-persistence/tests/storage-contract.spec.ts.

Python differences: upstream's non-JSON values ``BigInt``, ``undefined``, a function,
a ``Symbol`` and a ``Map`` become an integer outside the safe range, ``math.nan``, a
lambda, ``object()`` and an ``OrderedDict``; ``Object.isFrozen`` becomes
``FrozenJsonObject``; ``error.cause`` becomes ``__cause__``.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from deskagents.core.session import SESSION_FORMAT_VERSION, SessionEvent, SessionId
from deskagents.session.persistence import (
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionFormatUnsupportedError,
    SessionHandleClosedError,
    SessionLocation,
    SessionOwnershipLostError,
    SessionPersistenceCorruptionError,
    SessionPersistenceNotFoundError,
    SessionPersistenceRevision,
    SessionReadOnlyError,
    assert_contiguous,
    assert_stored_id,
    assert_version,
    materialize_append_batch,
    session_format_version_refusal,
    validate_stored_events,
)
from deskagents.testing import meta
from deskagents.util.values import MAX_SAFE_INTEGER, FrozenJsonObject

SPEC = "packages/session/session-persistence/tests/storage-contract.spec.ts"
LOCATION = SessionLocation("jsonl", "/store/session.jsonl")


def user_message(seq: int) -> dict[str, Any]:
    """A stored user/message event with a well-formed identified message."""
    return {
        "type": "user/message",
        "seq": seq,
        "time": seq + 1,
        "data": {
            "id": "stored-user",
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
            "source": {"kind": "user"},
        },
        "surfaceOp": "append",
    }


@pytest.mark.upstream(
    f"{SPEC}::assertStoredId > accepts a bound header and names both ids on a mismatch"
)
def test_assert_stored_id() -> None:
    m = meta("bound")
    assert_stored_id(m["id"], m)
    with pytest.raises(
        Exception,
        match=re.escape(
            'stored session identity mismatch: requested "requested", header contains "bound"'
        ),
    ):
        assert_stored_id(SessionId("requested"), m)


@pytest.mark.upstream(f"{SPEC}::assertVersion > accepts the current format version")
def test_assert_version_accepts_current() -> None:
    assert_version(meta("current"))


@pytest.mark.upstream(
    f"{SPEC}::assertVersion > refuses a newer version with the upgrade direction and the "
    "artifact location"
)
def test_assert_version_refuses_newer() -> None:
    newer = {**meta("newer"), "version": SESSION_FORMAT_VERSION + 42}
    with pytest.raises(SessionFormatUnsupportedError) as caught:
        assert_version(newer, LOCATION)
    message = str(caught.value)
    assert "written by a newer harness — upgrade the harness to open it" in message
    assert f"(raw log: {LOCATION.path})" in message
    assert caught.value.location is LOCATION


@pytest.mark.upstream(
    f"{SPEC}::assertVersion > refuses an older version without a location suffix when the "
    "backend has no artifact"
)
def test_assert_version_refuses_older() -> None:
    older = {**meta("older"), "version": SESSION_FORMAT_VERSION - 1}
    with pytest.raises(SessionFormatUnsupportedError) as caught:
        assert_version(older)
    message = str(caught.value)
    assert message == session_format_version_refusal("older", SESSION_FORMAT_VERSION - 1)
    assert "this build ships no upgrade path for it" in message
    assert "raw log" not in message
    assert caught.value.location is None


@pytest.mark.upstream(
    f"{SPEC}::sessionFormatVersionRefusal > states the direction for newer and older stored "
    "versions"
)
def test_session_format_version_refusal() -> None:
    v = SESSION_FORMAT_VERSION
    assert session_format_version_refusal("s", v + 1) == (
        f'session "s" uses log format v{v + 1}, but this harness reads only v{v}: the log '
        "was written by a newer harness — upgrade the harness to open it"
    )
    assert session_format_version_refusal("s", v - 1) == (
        f'session "s" uses log format v{v - 1}, older than the supported v{v}, and this '
        "build ships no upgrade path for it"
    )


@pytest.mark.upstream(
    f"{SPEC}::validateStoredEvents > adopts and freezes the events in place, returning the "
    "same array"
)
def test_validate_adopts_and_freezes_in_place() -> None:
    m = meta("adopted")
    events: list[SessionEvent] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        user_message(1),
    ]
    validated = validate_stored_events(m, events)
    assert validated is events
    assert isinstance(validated[1]["data"], FrozenJsonObject)


@pytest.mark.upstream(
    f"{SPEC}::validateStoredEvents > adopts a merge-extended turn/end reason outside the "
    "closed built-in set"
)
def test_validate_adopts_extended_turn_end_reason() -> None:
    m = meta("extended-reason")
    events: list[SessionEvent] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "turn/end",
            "seq": 1,
            "time": 2,
            "data": {"turn": 1, "reason": {"kind": "plugin/custom-outcome"}},
        },
    ]
    assert validate_stored_events(m, events) is events


@pytest.mark.upstream(
    f"{SPEC}::validateStoredEvents > refuses an unknown event type before adopting anything"
)
def test_validate_refuses_unknown_type() -> None:
    m = meta("unknown-type")
    events: list[SessionEvent] = [{"type": "mystery/event", "seq": 0, "time": 1, "data": {}}]
    with pytest.raises(SessionFormatUnsupportedError):
        validate_stored_events(m, events)
    with pytest.raises(
        SessionFormatUnsupportedError,
        match=re.escape(
            'session "unknown-type" contains event type "mystery/event" (seq 0) unknown to '
            "this harness"
        ),
    ):
        validate_stored_events(m, events)
    # With an artifact, the refusal points at the raw log.
    with pytest.raises(SessionFormatUnsupportedError) as caught:
        validate_stored_events(m, events, LOCATION)
    assert f"(raw log: {LOCATION.path})" in str(caught.value)
    assert caught.value.location is LOCATION


@pytest.mark.upstream(
    f"{SPEC}::validateStoredEvents > retains an unknown event type its writer marked ignorable"
)
def test_validate_retains_ignorable_unknown_type() -> None:
    m = meta("ignorable-unknown")
    events: list[SessionEvent] = [
        {"type": "foreign/telemetry", "seq": 0, "time": 1, "data": {}, "ignorable": True}
    ]
    assert validate_stored_events(m, events) is events
    assert events[0]["type"] == "foreign/telemetry"
    assert events[0]["ignorable"] is True


@pytest.mark.upstream(
    f'{SPEC}::validateStoredEvents > refuses the retired request/header "fallback" reason '
    "while accepting current headers"
)
def test_validate_refuses_retired_fallback_reason() -> None:
    m = meta("retired-reason")
    retired: list[SessionEvent] = [
        {
            "type": "request/header",
            "seq": 0,
            "time": 1,
            "data": {
                "header": {"config": {"provider": "mock", "model": "legacy"}},
                "reason": "fallback",
            },
        }
    ]
    with pytest.raises(SessionFormatUnsupportedError):
        validate_stored_events(m, retired)
    with pytest.raises(
        SessionFormatUnsupportedError,
        match=re.escape(
            'session "retired-reason" contains a request/header event (seq 0) with the '
            'unsupported legacy reason "fallback"'
        ),
    ):
        validate_stored_events(m, retired)

    current: list[SessionEvent] = [
        {
            "type": "request/header",
            "seq": 0,
            "time": 1,
            "data": {
                "header": {"config": {"provider": "mock", "model": "current"}},
                "reason": "initial",
            },
        }
    ]
    assert validate_stored_events(m, current) is current


@pytest.mark.upstream(
    f"{SPEC}::validateStoredEvents > wraps a record validation failure as corruption with its cause"
)
def test_validate_wraps_failure_as_corruption() -> None:
    m = meta("damaged")
    events: list[SessionEvent] = [
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            # No identified message: adoption refuses the record.
            "data": {"role": "user", "content": [], "source": {"kind": "user"}},
            "surfaceOp": "append",
        }
    ]
    with pytest.raises(SessionPersistenceCorruptionError) as caught:
        validate_stored_events(m, events)
    message = str(caught.value)
    assert 'stored session "damaged" failed validation' in message
    assert "lacks an identified message" in message
    assert isinstance(caught.value.__cause__, Exception)


class _Trap(Mapping[str, Any]):
    """A stored record whose ``data`` access raises, like upstream's throwing getter."""

    def __init__(self, failure: Exception) -> None:
        self._fields: dict[str, Any] = {"type": "user/message", "seq": 0, "time": 1}
        self._failure = failure

    def __getitem__(self, key: str) -> Any:
        if key == "data":
            raise self._failure
        return self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter([*self._fields, "data"])

    def __len__(self) -> int:
        return len(self._fields) + 1


@pytest.mark.upstream(
    f"{SPEC}::validateStoredEvents > passes an adoption-side format refusal through unwrapped"
)
def test_validate_passes_adoption_refusal_through() -> None:
    # The adoption step may itself refuse a record as unsupported rather than damaged;
    # such a refusal must reach the caller as-is, never rebranded as corruption.
    m = meta("adoption-refusal")
    passthrough = SessionFormatUnsupportedError("refused during adoption")
    with pytest.raises(SessionFormatUnsupportedError) as caught:
        validate_stored_events(m, [_Trap(passthrough)])
    assert caught.value is passthrough


@pytest.mark.upstream(
    f"{SPEC}::materializeAppendBatch > returns a detached lossless-JSON snapshot of the batch"
)
def test_materialize_append_batch_detaches() -> None:
    original: list[dict[str, Any]] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
    ]
    batch = materialize_append_batch(original)
    assert batch is not original
    assert batch == original
    # The snapshot is detached: mutating the caller's batch afterwards cannot change
    # what a backend persists.
    original[0]["data"]["turn"] = 99
    assert batch[0]["data"]["turn"] == 1


@pytest.mark.upstream(
    f"{SPEC}::materializeAppendBatch > rejects every non-JSON value, not only BigInt"
)
def test_materialize_append_batch_rejects_non_json() -> None:
    # Every value `snapshot_json_value` refuses must be refused here — a backend passing
    # this helper cannot accept values that corrupt the durable round-trip.
    cyclic: dict[str, Any] = {"type": "text", "text": "x"}
    cyclic["self"] = cyclic
    bad_values: list[object] = [
        MAX_SAFE_INTEGER + 1,  # BigInt
        math.nan,  # undefined: no Python counterpart; NaN is likewise lost by JSON
        math.inf,  # → null
        lambda: 0,  # function
        object(),  # symbol
        OrderedDict(),  # exotic object
        cyclic,  # circular ref
    ]
    for bad in bad_values:
        events = [{"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1, "extra": bad}}]
        with pytest.raises(TypeError):
            materialize_append_batch(events)
        with pytest.raises(TypeError, match="losslessly JSON-serializable"):
            materialize_append_batch(events)


@pytest.mark.upstream(
    f"{SPEC}::assertContiguous > accepts a batch continuing exactly at the cursor"
)
def test_assert_contiguous_accepts() -> None:
    events: list[SessionEvent] = [
        {"type": "turn/start", "seq": 6, "time": 1, "data": {"turn": 2}},
        {
            "type": "turn/end",
            "seq": 7,
            "time": 2,
            "data": {"turn": 2, "reason": {"kind": "completed"}},
        },
    ]
    assert_contiguous(SessionId("ok"), events, 6)
    assert_contiguous(SessionId("ok"), [], 6)


@pytest.mark.upstream(
    f"{SPEC}::assertContiguous > names the expected seq for a stale batch and for a mid-batch gap"
)
def test_assert_contiguous_names_expected_seq() -> None:
    stale: list[SessionEvent] = [{"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}]
    with pytest.raises(
        Exception, match=re.escape('append seq mismatch for "stale": expected 6 at index 0, got 0')
    ):
        assert_contiguous(SessionId("stale"), stale, 6)

    gapped: list[SessionEvent] = [
        {"type": "turn/start", "seq": 6, "time": 1, "data": {"turn": 2}},
        {
            "type": "turn/end",
            "seq": 8,
            "time": 2,
            "data": {"turn": 2, "reason": {"kind": "completed"}},
        },
    ]
    with pytest.raises(
        Exception, match=re.escape('append seq mismatch for "gapped": expected 7 at index 1, got 8')
    ):
        assert_contiguous(SessionId("gapped"), gapped, 6)


@pytest.mark.upstream(
    f"{SPEC}::error vocabulary > carries stable names, messages, and the session id"
)
def test_error_vocabulary() -> None:
    sid = SessionId("errored")
    not_found = SessionPersistenceNotFoundError(sid)
    assert type(not_found).__name__ == "SessionPersistenceNotFoundError"
    assert not_found.session_id == sid
    assert str(not_found) == 'session "errored" not found'

    exists = SessionAlreadyExistsError(sid)
    assert type(exists).__name__ == "SessionAlreadyExistsError"
    assert str(exists) == 'session "errored" already exists'

    owned = SessionAlreadyOwnedError(sid)
    assert type(owned).__name__ == "SessionAlreadyOwnedError"
    assert str(owned) == 'session "errored" is already owned by an active write handle'

    read_only = SessionReadOnlyError(sid, "append")
    assert type(read_only).__name__ == "SessionReadOnlyError"
    assert str(read_only) == 'session "errored": append is not available on a read handle'

    lost = SessionOwnershipLostError(sid)
    assert type(lost).__name__ == "SessionOwnershipLostError"
    assert str(lost) == 'session "errored": write ownership was lost; close this handle and reopen'

    closed = SessionHandleClosedError(sid, "flush")
    assert type(closed).__name__ == "SessionHandleClosedError"
    assert str(closed) == 'session "errored": flush on a closed handle'

    cause = Exception("detail")
    with pytest.raises(SessionPersistenceCorruptionError) as caught:
        raise SessionPersistenceCorruptionError("bad log") from cause
    assert type(caught.value).__name__ == "SessionPersistenceCorruptionError"
    assert caught.value.__cause__ is cause


@pytest.mark.upstream(
    f"{SPEC}::SessionPersistenceRevision > brands the backend token without changing its "
    "runtime value"
)
def test_session_persistence_revision() -> None:
    assert SessionPersistenceRevision("rev:1") == "rev:1"
