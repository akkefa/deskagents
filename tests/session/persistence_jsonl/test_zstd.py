"""Zstandard frame and default-backend coverage.

Ported from packages/session/session-persistence-jsonl/tests/zstd.spec.ts and
zstd.compat.spec.ts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core.session import SESSION_FORMAT_VERSION, SessionHeader, SessionId
from deskagents.session import persistence_jsonl
from deskagents.session.persistence import SessionPersistence
from deskagents.session.persistence_jsonl import log_path
from deskagents.session.persistence_jsonl.zstd import (
    ZstdFrameRange,
    compress_zstd_frame,
    decode_zstd_frames,
    decompress_zstd_frame,
    scan_zstd_frames,
)
from deskagents.testing.persistence_contract import one_turn_log, second_turn
from deskagents.util.values import stringify_json

SPEC = "packages/session/session-persistence-jsonl/tests/zstd.spec.ts"
COMPAT_SPEC = "packages/session/session-persistence-jsonl/tests/zstd.compat.spec.ts"
MAGIC = b"\x28\xb5\x2f\xfd"


def append_bytes(path: Path, content: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(content)


def meta(id_: str, cwd: str = "/work") -> SessionHeader:
    return {
        "version": SESSION_FORMAT_VERSION,
        "id": SessionId(id_),
        "createdAt": 1_000,
        "cwd": cwd,
        "isSeeded": False,
    }


async def mount(root: Path, compression: str | None = None) -> tuple[Context, SessionPersistence]:
    ctx = Context()
    config: dict[str, Any] = {"root": str(root)}
    if compression is not None:
        config["compression"] = compression
    await ctx.plugin(persistence_jsonl, config)
    service = ctx.get("sessionPersistence")
    assert isinstance(service, SessionPersistence)
    return ctx, service


@pytest.mark.upstream(
    f"{SPEC}::Zstandard frame structure > scans concatenated checksummed frames "
    "and honors a frame limit"
)
def test_scans_concatenated_checksummed_frames() -> None:
    first = compress_zstd_frame("header\n")
    second = compress_zstd_frame("event\n")
    stream = first + second

    assert scan_zstd_frames(b"").frames == ()
    assert scan_zstd_frames(stream).frames == (
        ZstdFrameRange(0, len(first)),
        ZstdFrameRange(len(first), len(stream)),
    )
    assert scan_zstd_frames(stream, 1).frames == (ZstdFrameRange(0, len(first)),)
    assert first[:4] == MAGIC
    assert first[4] & 0x04 == 0x04
    assert decompress_zstd_frame(first) == b"header\n"
    assert b"".join(decode_zstd_frames(stream, scan_zstd_frames(stream).frames)) == (
        b"header\nevent\n"
    )


@pytest.mark.upstream(
    f"{COMPAT_SPEC}::JSONL Zstandard compatibility > round-trips concatenated "
    "checksummed frames through the built-in Node API"
)
def test_round_trips_concatenated_frames() -> None:
    encoded = compress_zstd_frame('{"type":"session"}\n') + compress_zstd_frame(
        '{"type":"turn/start"}\n'
    )
    scan = scan_zstd_frames(encoded)
    assert scan.torn_start is None
    assert len(scan.frames) == 2
    assert all(encoded[item.start : item.start + 4] == MAGIC for item in scan.frames)
    assert b'"type":"turn/start"' in b"".join(decode_zstd_frames(encoded, scan.frames))
    event = encoded[scan.frames[1].start : scan.frames[1].end]
    assert scan_zstd_frames(event[:-1]).torn_start == 0


@pytest.mark.upstream(
    f"{SPEC}::Zstandard frame structure > distinguishes incomplete frame regions "
    "from invalid complete structure"
)
def test_distinguishes_torn_frames_from_invalid_structure() -> None:
    assert scan_zstd_frames(MAGIC[:2]).torn_start == 0
    assert scan_zstd_frames(MAGIC).torn_start == 0
    with pytest.raises(ValueError, match="invalid frame magic"):
        scan_zstd_frames(b"\0\0\0\0")
    with pytest.raises(ValueError, match="reserved frame-header bit"):
        scan_zstd_frames(MAGIC + b"\x08")


@pytest.mark.upstream(
    f"{SPEC}::JsonlSessionPersistence: default Zstandard encoding > writes .jsonl.zstd "
    "by default with one header frame and one first-batch frame"
)
@pytest.mark.asyncio
async def test_default_backend_writes_header_and_batch_frames(tmp_path: Path) -> None:
    ctx, backend = await mount(tmp_path)
    header = meta("default-zstd")
    writer = await backend.create(header)
    await writer.append(one_turn_log())
    await writer.close()

    path = Path(log_path(str(tmp_path), header["cwd"], header["id"], "zstd"))
    content = await asyncio.to_thread(path.read_bytes)
    assert content[:4] == MAGIC
    assert len(scan_zstd_frames(content).frames) == 2
    raw_path = Path(log_path(str(tmp_path), header["cwd"], header["id"], "none"))
    assert not await asyncio.to_thread(raw_path.exists)
    reader = await backend.open(header["id"], "read")
    assert list((await reader.read()).events) == one_turn_log()
    await reader.close()
    await ctx.fiber.dispose()


@pytest.mark.upstream(
    f"{SPEC}::JsonlSessionPersistence: default Zstandard encoding > appends one frame "
    "per durable batch without rewriting prior bytes"
)
@pytest.mark.asyncio
async def test_append_adds_one_frame_without_rewriting_prefix(tmp_path: Path) -> None:
    ctx, backend = await mount(tmp_path)
    header = meta("append-frame")
    writer = await backend.create(header)
    await writer.append(one_turn_log())
    path = Path(log_path(str(tmp_path), header["cwd"], header["id"], "zstd"))
    before = await asyncio.to_thread(path.read_bytes)
    continuation = second_turn()
    await writer.append(continuation)
    await writer.close()

    after = await asyncio.to_thread(path.read_bytes)
    assert after[: len(before)] == before
    assert len(scan_zstd_frames(after).frames) == 3
    reader = await backend.open(header["id"], "read")
    assert list((await reader.read()).events) == [*one_turn_log(), *continuation]
    await reader.close()
    await ctx.fiber.dispose()


@pytest.mark.upstream(
    f"{SPEC}::JsonlSessionPersistence: default Zstandard encoding > rejects a complete "
    "frame containing a torn JSONL record"
)
@pytest.mark.asyncio
async def test_rejects_complete_frame_with_torn_json_record(tmp_path: Path) -> None:
    ctx, backend = await mount(tmp_path)
    header = meta("bad-json")
    writer = await backend.create(header)
    await writer.append(one_turn_log())
    await writer.close()
    path = Path(log_path(str(tmp_path), header["cwd"], header["id"], "zstd"))
    await asyncio.to_thread(append_bytes, path, compress_zstd_frame('{"type":"turn/start"'))

    with pytest.raises(Exception, match="complete frame contains a torn JSONL record"):
        await backend.open(header["id"], "read")
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_recovers_records_from_a_frame_torn_in_its_checksum(tmp_path: Path) -> None:
    ctx, backend = await mount(tmp_path)
    header = meta("torn-checksum")
    writer = await backend.create(header)
    await writer.append(one_turn_log())
    await writer.close()
    path = Path(log_path(str(tmp_path), header["cwd"], header["id"], "zstd"))
    recovered = second_turn()
    plaintext = ("\n".join(stringify_json(event) for event in recovered) + "\n").encode()
    frame = compress_zstd_frame(plaintext)
    await asyncio.to_thread(append_bytes, path, frame[:-1])

    reader = await backend.open(header["id"], "read")
    assert list((await reader.read()).events) == [*one_turn_log(), *recovered]
    await reader.close()

    following = [
        {"type": "turn/start", "seq": 8, "time": 11, "data": {"turn": 3}},
        {
            "type": "turn/end",
            "seq": 9,
            "time": 12,
            "data": {"turn": 3, "reason": {"kind": "completed"}},
        },
    ]
    repair = await backend.open(header["id"], "write")
    await repair.append(following)
    await repair.close()
    repaired = await asyncio.to_thread(path.read_bytes)
    assert scan_zstd_frames(repaired).torn_start is None
    final = await backend.open(header["id"], "read")
    assert list((await final.read()).events) == [*one_turn_log(), *recovered, *following]
    await final.close()
    await ctx.fiber.dispose()
