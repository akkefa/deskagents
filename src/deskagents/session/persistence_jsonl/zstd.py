"""Zstandard frame primitives for the JSONL persistence backend.

Upstream: packages/session/session-persistence-jsonl/src/zstd.ts

The artifact is a concatenation of independently decodable, checksummed frames:
one header frame followed by one frame per durable event batch.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import zstandard

ZSTD_MAGIC = 0xFD2FB528


@dataclass(frozen=True)
class ZstdFrameRange:
    """Inclusive start and exclusive end of one complete frame."""

    start: int
    end: int


@dataclass(frozen=True)
class ZstdFrameScan:
    """Complete frames plus the start of an incomplete final frame."""

    frames: tuple[ZstdFrameRange, ...]
    torn_start: int | None = None


def scan_zstd_frames(source: bytes, max_frames: int | None = None) -> ZstdFrameScan:
    """Locate complete frames without decompressing their blocks."""
    frames: list[ZstdFrameRange] = []
    offset = 0
    limit = float("inf") if max_frames is None else max_frames

    while offset < len(source):
        start = offset
        if len(source) - offset < 4:
            return ZstdFrameScan(tuple(frames), start)
        if int.from_bytes(source[offset : offset + 4], "little") != ZSTD_MAGIC:
            raise ValueError(f"corrupt Zstandard session log: invalid frame magic at byte {offset}")
        offset += 4

        if offset == len(source):
            return ZstdFrameScan(tuple(frames), start)
        descriptor = source[offset]
        offset += 1
        if descriptor & 0x18:
            raise ValueError(
                f"corrupt Zstandard session log: reserved frame-header bit at byte {offset - 1}"
            )

        content_size_flag = descriptor >> 6
        single_segment = bool(descriptor & 0x20)
        checksum = bool(descriptor & 0x04)
        dictionary_flag = descriptor & 0x03
        dictionary_bytes = 4 if dictionary_flag == 3 else dictionary_flag
        content_size_bytes = (
            (1 if content_size_flag == 0 and single_segment else 0)
            if content_size_flag == 0
            else 1 << content_size_flag
        )
        remaining_header_bytes = (
            (0 if single_segment else 1) + dictionary_bytes + content_size_bytes
        )
        if len(source) - offset < remaining_header_bytes:
            return ZstdFrameScan(tuple(frames), start)
        offset += remaining_header_bytes

        while True:
            if len(source) - offset < 3:
                return ZstdFrameScan(tuple(frames), start)
            block_header = int.from_bytes(source[offset : offset + 3], "little")
            offset += 3
            last_block = bool(block_header & 1)
            block_type = (block_header >> 1) & 0x03
            block_size = block_header >> 3
            if block_type == 0x03:
                raise ValueError(
                    f"corrupt Zstandard session log: reserved block type at byte {offset - 3}"
                )
            payload_bytes = 1 if block_type == 0x01 else block_size
            if len(source) - offset < payload_bytes:
                return ZstdFrameScan(tuple(frames), start)
            offset += payload_bytes
            if last_block:
                break

        if checksum:
            if len(source) - offset < 4:
                return ZstdFrameScan(tuple(frames), start)
            offset += 4
        frames.append(ZstdFrameRange(start, offset))
        if len(frames) == limit:
            return ZstdFrameScan(tuple(frames))

    return ZstdFrameScan(tuple(frames))


def compress_zstd_frame(source: bytes | str) -> bytes:
    """Compress one independently decodable frame with a content checksum."""
    data = source.encode("utf-8", "surrogatepass") if isinstance(source, str) else source
    return zstandard.ZstdCompressor(write_checksum=True).compress(data)


def decompress_zstd_frame(source: bytes) -> bytes:
    """Decompress one complete frame and validate its checksum."""
    decoder = zstandard.ZstdDecompressor().decompressobj()
    plaintext = decoder.decompress(source)
    if not decoder.eof or decoder.unused_data:
        raise zstandard.ZstdError("input is not exactly one complete Zstandard frame")
    return plaintext + decoder.flush()


def decode_zstd_frames(source: bytes, frames: Sequence[ZstdFrameRange]) -> Iterator[bytes]:
    """Decode complete frames in source order, validating every checksum."""
    for frame in frames:
        try:
            yield decompress_zstd_frame(source[frame.start : frame.end])
        except zstandard.ZstdError as error:
            raise ValueError(
                f"corrupt Zstandard session log: frame at byte {frame.start} failed validation"
            ) from error


def decompress_zstd_prefix(source: bytes) -> bytes:
    """Recover plaintext emitted before EOF from a structurally torn final frame."""
    decoder = zstandard.ZstdDecompressor().decompressobj()
    try:
        return decoder.decompress(source)
    except zstandard.ZstdError:
        # A cut before a complete compressed block may produce no usable plaintext.
        return b""
