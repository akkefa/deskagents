"""Lossless range encoding for JSONL ``sourceEventSeqs`` arrays.

Upstream: packages/core/session/src/seq-ranges.ts
"""

from __future__ import annotations

from collections.abc import Sequence

from deskagents.util.values import MAX_SAFE_INTEGER, is_safe_integer

from .types import SessionSeq, session_seq

EncodedSeq = int | list[int]
"""A stored source sequence or an inclusive ``[start, end]`` consecutive range."""


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(values[i] > values[i - 1] for i in range(1, len(values)))


def encode_seq_ranges(values: Sequence[SessionSeq]) -> list[EncodedSeq]:
    """Replace profitable consecutive runs (three or more) with inclusive pairs."""
    if not _strictly_increasing(values):
        return list(values)
    encoded: list[EncodedSeq] = []
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[end] + 1:
            end += 1
        if end - start >= 2:
            encoded.append([values[start], values[end]])
        else:
            encoded.extend(values[start : end + 1])
        start = end + 1
    return encoded


def _assert_seq(value: object) -> int:
    if not is_safe_integer(value) or value < 0:  # type: ignore[operator]
        raise TypeError("sourceEventSeqs must contain non-negative safe integers")
    return int(value)  # type: ignore[call-overload,no-any-return]


def decode_seq_ranges(value: object, max_entries: int = MAX_SAFE_INTEGER) -> list[SessionSeq]:
    """Expand a parsed storage-form source sequence array."""
    if not isinstance(value, list | tuple):
        raise TypeError("sourceEventSeqs must be an array")
    decoded: list[SessionSeq] = []
    has_range = False
    for entry in value:
        if isinstance(entry, int | float) and not isinstance(entry, bool):
            seq = _assert_seq(entry)
            if len(decoded) >= max_entries:
                raise TypeError("sourceEventSeqs exceeds its event sequence")
            decoded.append(session_seq(seq))
            continue
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            raise TypeError("sourceEventSeqs range entries must be [start, end] pairs")
        start, end = _assert_seq(entry[0]), _assert_seq(entry[1])
        if end < start:
            raise TypeError("sourceEventSeqs ranges require start <= end")
        if end - start + 1 > max_entries - len(decoded):
            raise TypeError("sourceEventSeqs range exceeds its event sequence")
        decoded.extend(SessionSeq(seq) for seq in range(start, end + 1))
        has_range = True
    if has_range and not _strictly_increasing(decoded):
        raise TypeError("sourceEventSeqs ranges must be strictly increasing")
    return decoded
