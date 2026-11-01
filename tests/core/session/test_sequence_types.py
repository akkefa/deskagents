"""Ported from packages/core/session/tests/sequence-types.spec.ts.

The ``expectTypeOf`` assertions are TypeScript type-level checks with no runtime
counterpart; only their runtime halves are ported.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    Session,
    SessionEvent,
    SessionHeader,
    session_id,
    session_log_offset,
    session_seq,
)
from deskagents.util.values import MAX_SAFE_INTEGER

SPEC = "packages/core/session/tests/sequence-types.spec.ts"
GROUP = "Session log positions"


def seeded_header(sid: str) -> SessionHeader:
    return {"version": SESSION_FORMAT_VERSION, "id": sid, "createdAt": 1, "isSeeded": True}


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > admits non-negative safe integers into distinct sequence roles"
)
def test_admits_non_negative_safe_integers() -> None:
    seq = session_seq(3)
    offset = session_log_offset(4)

    assert seq == 3
    assert offset == 4


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects invalid Session sequence positions (%s)")
@pytest.mark.parametrize("value", [-1, -0.0, 0.5, MAX_SAFE_INTEGER + 1, float("nan")])
def test_rejects_invalid_sequence_positions(value: object) -> None:
    with pytest.raises(TypeError, match=r"SessionSeq must be a non-negative safe integer"):
        session_seq(value)
    with pytest.raises(TypeError, match=r"SessionLogOffset must be a non-negative safe integer"):
        session_log_offset(value)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > keeps event identities separate from log offsets")
def test_keeps_event_identities_separate() -> None:
    session = Session.create(session_id("typed-positions"))
    event = session.append("turn/start", {"turn": 1})

    assert event["seq"] == 0
    assert session.first_live_seq == 0
    assert session.seq == 1


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects a negative-zero seq at the restored event boundary"
)
def test_rejects_negative_zero_restored_seq() -> None:
    sid = session_id("negative-zero-event")
    with pytest.raises(ValueError, match=r"invalid event envelope"):
        Session.from_restore(
            sid,
            [{"type": "turn/start", "seq": -0.0, "time": 1, "data": {"turn": 1}}],
            {"version": SESSION_FORMAT_VERSION, "id": sid, "createdAt": 1, "isSeeded": False},
            session_log_offset(0),
            "detached",
        )


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > keeps fork lineage outside the logical header integer fields"
)
def test_keeps_fork_lineage_outside_header() -> None:
    source = Session.create(session_id("source"))
    source.append("turn/start", {"turn": 1})
    source.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    sid = session_id("child")

    child = Session.create(sid, source.snapshot_events(), seeded_header(sid), source.seq)

    assert child.header["isSeeded"] is True
    assert "seedLength" not in child.header
    assert child.inherited_event_count == 2
    assert [event["type"] for event in child.own_events()] == ["session/end-seed"]
    assert child.is_own_seq(session_seq(1)) is False
    assert child.is_own_seq(session_seq(2)) is True
    assert child.is_own_seq(session_seq(3)) is False

    fresh = Session.create(session_id("fresh"))
    assert fresh.header["isSeeded"] is False
    assert fresh.inherited_event_count == 0


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > appends child-owned setup after the constructor-owned inherited marker"
)
def test_appends_child_setup_after_inherited_marker() -> None:
    parent = Session.create(session_id("suffix-parent"))
    parent.append("turn/start", {"turn": 1})
    parent.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    sid = session_id("suffix-child")

    child = Session.create(sid, parent.snapshot_events(), seeded_header(sid), parent.seq)
    child.append("request/context", {"provider": "provider", "model": "model"})

    assert [event["type"] for event in child.own_events()] == [
        "session/end-seed",
        "request/context",
    ]
    assert child.is_own_seq(session_seq(1)) is False
    assert child.is_own_seq(session_seq(2)) is True
    assert child.is_own_seq(session_seq(3)) is True


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > accepts child closers after the inherited marker and rejects a later "
    "inherited marker"
)
def test_accepts_child_closers_rejects_later_marker() -> None:
    sid = session_id("marked-fork")
    header = seeded_header(sid)
    seed: list[SessionEvent] = [
        {"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}},
        {"type": "session/end-seed", "seq": session_seq(1), "time": 2, "data": {"inherited": True}},
        {
            "type": "turn/end",
            "seq": session_seq(2),
            "time": 3,
            "data": {"turn": 1, "reason": {"kind": "forked"}},
        },
    ]
    child = Session.create(sid, seed, header, session_log_offset(1))
    assert list(child.snapshot_events()) == seed
    assert [event["type"] for event in child.own_events()] == ["session/end-seed", "turn/end"]
    assert child.first_live_seq == 3
    later: SessionEvent = {
        "type": "session/end-seed",
        "seq": session_seq(3),
        "time": 4,
        "data": {"inherited": True},
    }
    with pytest.raises(ValueError, match=re.escape("final inherited marker")):
        Session.create(sid, [*seed, later], header, session_log_offset(1))


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > requires a separately supplied inherited cut for a seeded header"
)
def test_requires_inherited_cut_for_seeded_header() -> None:
    sid = session_id("missing-cut")
    with pytest.raises(ValueError, match=r"seeded session requires an inherited event count"):
        Session.create(sid, [], seeded_header(sid))


@pytest.mark.upstream(f"{SPEC}::{GROUP} > requires an explicit constructor seed for seeded lineage")
def test_requires_explicit_seed_for_seeded_lineage() -> None:
    sid = session_id("missing-seed")
    with pytest.raises(ValueError, match=r"seeded session requires an explicit constructor seed"):
        Session.create(sid, None, seeded_header(sid), session_log_offset(0))


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > requires the exact cut to agree with lineage and log length"
)
def test_requires_cut_to_agree_with_lineage() -> None:
    unseeded_id = session_id("unseeded-nonzero-cut")
    with pytest.raises(ValueError, match=r"unseeded session inherited event count must be 0"):
        Session.create(
            unseeded_id,
            [],
            {
                "version": SESSION_FORMAT_VERSION,
                "id": unseeded_id,
                "createdAt": 1,
                "isSeeded": False,
            },
            session_log_offset(1),
        )

    seeded_id = session_id("seeded-oversized-cut")
    with pytest.raises(ValueError, match=r"inherited event count exceeds its event log"):
        Session.create(seeded_id, [], seeded_header(seeded_id), session_log_offset(1))

    short_cut_id = session_id("seeded-short-cut")
    seed: list[SessionEvent] = [
        {"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}},
        {
            "type": "turn/end",
            "seq": session_seq(1),
            "time": 2,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    with pytest.raises(ValueError, match=r"constructor seed must equal its inherited prefix"):
        Session.create(short_cut_id, seed, seeded_header(short_cut_id), session_log_offset(1))


@pytest.mark.upstream(f"{SPEC}::{GROUP} > revalidates a supplied inherited event count (%s)")
@pytest.mark.parametrize("value", [-1, 0.5, MAX_SAFE_INTEGER + 1])
def test_revalidates_supplied_inherited_event_count(value: Any) -> None:
    sid = session_id(f"bad-inherited-count-{value}")
    with pytest.raises(TypeError, match=r"SessionLogOffset must be a non-negative safe integer"):
        Session.create(sid, [], seeded_header(sid), value)
