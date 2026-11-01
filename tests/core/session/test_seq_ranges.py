"""Ported from packages/core/session/tests/seq-ranges.spec.ts."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from deskagents.core.session import (
    SessionSeq,
    decode_seq_ranges,
    encode_seq_ranges,
    session_seq,
)
from deskagents.util.values import MAX_SAFE_INTEGER

SPEC = "packages/core/session/tests/seq-ranges.spec.ts"
GROUP = "sourceEventSeqs ranges"


def seqs(values: Sequence[int]) -> list[SessionSeq]:
    return [session_seq(value) for value in values]


@pytest.mark.upstream(f"{SPEC}::{GROUP} > round-trips %j")
@pytest.mark.parametrize(
    "values",
    [
        [],
        [5],
        [10, 11, 12, 13, 14],
        [16, 17, 100, 200, 201, 202, 203],
        [3, 2],
        [MAX_SAFE_INTEGER - 1, 0, MAX_SAFE_INTEGER - 2],
    ],
)
def test_round_trips(values: list[int]) -> None:
    assert decode_seq_ranges(encode_seq_ranges(seqs(values))) == values


@pytest.mark.upstream(f"{SPEC}::{GROUP} > encodes only profitable increasing runs")
def test_encodes_only_profitable_increasing_runs() -> None:
    assert encode_seq_ranges(seqs([1, 3, 4, 5, 7])) == [1, [3, 5], 7]
    assert encode_seq_ranges(seqs([1, 3, 4, 7])) == [1, 3, 4, 7]
    assert encode_seq_ranges(seqs([3, 2])) == [3, 2]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > does not impose a persistence-only source-event length limit"
)
def test_no_persistence_only_length_limit() -> None:
    values = [SessionSeq(index) for index in range(1_000_001)]
    assert encode_seq_ranges(values) == [[0, 1_000_000]]


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects malformed or impossible expansions")
def test_rejects_malformed_or_impossible_expansions() -> None:
    with pytest.raises(TypeError, match=r"must be an array"):
        decode_seq_ranges("nope")
    with pytest.raises(TypeError, match=r"non-negative safe integers"):
        decode_seq_ranges([-1])
    with pytest.raises(TypeError, match=r"\[start, end\] pairs"):
        decode_seq_ranges([[1]])
    with pytest.raises(TypeError, match=r"start <= end"):
        decode_seq_ranges([[4, 2]])
    with pytest.raises(TypeError, match=r"strictly increasing"):
        decode_seq_ranges([[2, 5], [4, 7]])
    with pytest.raises(TypeError, match=r"exceeds its event sequence"):
        decode_seq_ranges([0], 0)
    with pytest.raises(TypeError, match=r"exceeds its event sequence"):
        decode_seq_ranges([[0, 10]], 10)
