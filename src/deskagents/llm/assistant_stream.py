"""Compact, lossless assistant-stream records and record-level query helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from deskagents.util.values import deep_freeze, is_safe_integer, snapshot_json_value

from .assembler import BlockAssembler


def _safe_time(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not is_safe_integer(value):
        raise TypeError(f"Assistant stream time must be a safe integer, got {value!r}")
    return value


def _safe_index(value: object, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not is_safe_integer(value)
        or value < 0
    ):
        raise TypeError(f"{label} index must be a non-negative safe integer")
    return value


def _chunk(value: object) -> dict[str, Any]:
    snapshot = snapshot_json_value(value)
    if not isinstance(snapshot, dict):
        raise TypeError("Assistant stream chunk must be losslessly JSON-serializable")
    return snapshot


class AssistantStreamAccumulator:
    """Incrementally pack adjacent compatible delta chunks into compact runs."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def push(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        time = _safe_time(value.get("time"))
        chunk = _chunk(value.get("chunk"))
        kind = chunk.get("type")
        previous = self._records[-1] if self._records else None
        if kind in ("text-delta", "reasoning-delta"):
            index = _safe_index(chunk.get("index"), str(kind))
            text = chunk.get("text")
            if not isinstance(text, str):
                raise TypeError(f"{kind} text must be a string")
            record_type = "text-chunks" if kind == "text-delta" else "reasoning-chunks"
            if (
                previous is not None
                and previous["type"] == record_type
                and previous["index"] == index
                and is_safe_integer(time - previous["_lastTime"])
            ):
                previous["dt"].append(time - previous["_lastTime"])
                previous["texts"].append(text)
                previous["_lastTime"] = time
            else:
                self._records.append(
                    {
                        "type": record_type,
                        "time0": time,
                        "index": index,
                        "dt": [],
                        "texts": [text],
                        "_lastTime": time,
                    }
                )
        elif kind == "tool-call-delta":
            index = _safe_index(chunk.get("index"), "tool-call-delta")
            call_id, name, arguments = (
                chunk.get("id"),
                chunk.get("name"),
                chunk.get("argumentsDelta"),
            )
            if not isinstance(call_id, str):
                raise TypeError("tool-call-delta id must be a string")
            if "name" in chunk and not isinstance(name, str):
                raise TypeError("tool-call-delta name must be a string")
            if not isinstance(arguments, str):
                raise TypeError("tool-call-delta argumentsDelta must be a string")
            same = (
                previous is not None
                and previous["type"] == "tool-call-chunks"
                and previous["index"] == index
                and previous["id"] == call_id
                and previous.get("name") == name
                and ("name" in previous) == ("name" in chunk)
                and is_safe_integer(time - previous["_lastTime"])
            )
            if not call_id or name == "":
                self._records.append({"type": "chunk", "time": time, "chunk": chunk})
            elif same and previous is not None:
                previous["dt"].append(time - previous["_lastTime"])
                previous["args"].append(arguments)
                previous["_lastTime"] = time
            else:
                record = {
                    "type": "tool-call-chunks",
                    "time0": time,
                    "index": index,
                    "dt": [],
                    "id": call_id,
                    "args": [arguments],
                    "_lastTime": time,
                }
                if "name" in chunk:
                    record["name"] = name
                self._records.append(record)
        elif kind in ("block-start", "block-end", "usage", "finish"):
            self._records.append({"type": "chunk", "time": time, "chunk": chunk})
        else:
            raise AssertionError(
                f"unreachable variant in AssistantStreamAccumulator.push: {chunk!r}"
            )
        return deep_freeze({"time": time, "chunk": chunk})

    def snapshot(self) -> Sequence[Mapping[str, Any]]:
        durable = [
            {key: item for key, item in record.items() if key != "_lastTime"}
            for record in self._records
        ]
        return deep_freeze(durable)


def _validate_run(record: Mapping[str, Any], members: Sequence[object], label: str) -> None:
    time = _safe_time(record.get("time0"))
    _safe_index(record.get("index"), label)
    dt = record.get("dt")
    if not isinstance(dt, list | tuple) or any(
        not isinstance(gap, int) or isinstance(gap, bool) or not is_safe_integer(gap) for gap in dt
    ):
        raise TypeError(f"{label} dt must contain safe integers")
    if len(dt) != len(members) - 1:
        raise TypeError(f"{label} dt length must be one less than its members")
    for gap in dt:
        time += gap
        if not is_safe_integer(time):
            raise TypeError(f"{label} member times must stay safe integers")


def expand_assistant_stream(
    stream: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record in stream:
        kind = record.get("type")
        if kind == "chunk":
            if set(record) != {"type", "time", "chunk"}:
                raise TypeError(
                    "chunk Assistant stream record must contain exactly type, time, chunk"
                )
            chunks.append(
                {"time": _safe_time(record.get("time")), "chunk": _chunk(record["chunk"])}
            )
            continue
        member_key = "args" if kind == "tool-call-chunks" else "texts"
        expected = {"type", "time0", "index", "dt", member_key}
        if kind == "tool-call-chunks":
            expected.add("id")
            if "name" in record:
                expected.add("name")
        if (
            kind not in ("text-chunks", "reasoning-chunks", "tool-call-chunks")
            or set(record) != expected
        ):
            raise TypeError(f"Unsupported Assistant stream record {kind!r}")
        members = record[member_key]
        if (
            not isinstance(members, list | tuple)
            or not members
            or any(not isinstance(item, str) for item in members)
        ):
            raise TypeError(f"{kind} {member_key} must be a non-empty string array")
        if kind == "tool-call-chunks":
            if not isinstance(record.get("id"), str) or not record["id"]:
                raise TypeError("tool-call-chunks id must be a non-empty string")
            if "name" in record and (not isinstance(record["name"], str) or not record["name"]):
                raise TypeError("tool-call-chunks name must be a non-empty string")
        _validate_run(record, members, str(kind))
        time = _safe_time(record["time0"])
        for position, fragment in enumerate(members):
            if position:
                time += record["dt"][position - 1]
            if kind == "text-chunks":
                chunk = {"type": "text-delta", "index": record["index"], "text": fragment}
            elif kind == "reasoning-chunks":
                chunk = {"type": "reasoning-delta", "index": record["index"], "text": fragment}
            else:
                chunk = {
                    "type": "tool-call-delta",
                    "index": record["index"],
                    "id": record["id"],
                    "argumentsDelta": fragment,
                }
                if "name" in record:
                    chunk["name"] = record["name"]
            chunks.append({"time": time, "chunk": chunk})
    return deep_freeze(chunks)


def is_token_delta(chunk: Mapping[str, Any]) -> bool:
    kind = chunk.get("type")
    if kind in ("text-delta", "reasoning-delta"):
        return chunk.get("text") != ""
    return kind == "tool-call-delta" and (chunk.get("argumentsDelta") != "" or "name" in chunk)


def is_visible_chunk(chunk: Mapping[str, Any]) -> bool:
    kind = chunk.get("type")
    if kind in ("text-delta", "reasoning-delta"):
        return isinstance(chunk.get("text"), str) and bool(chunk["text"].strip())
    if kind == "block-start":
        return chunk.get("blockType") not in ("text", "reasoning", "tool-call")
    if kind == "block-end":
        block = chunk.get("block")
        if not isinstance(block, Mapping) or block.get("type") == "tool-call":
            return False
        if block.get("type") in ("text", "reasoning"):
            return isinstance(block.get("text"), str) and bool(block["text"].strip())
        return True
    return False


def chunk_has_visible_text(chunk: Mapping[str, Any]) -> bool:
    if chunk.get("type") == "text-delta":
        return isinstance(chunk.get("text"), str) and bool(chunk["text"].strip())
    block = chunk.get("block")
    return (
        chunk.get("type") == "block-end"
        and isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and bool(block["text"].strip())
    )


def _first_member_time(record: Mapping[str, Any], *, visible: bool) -> int | None:
    if record["type"] == "tool-call-chunks":
        if visible:
            return None
        if "name" in record:
            return int(record["time0"])
        fragments = record["args"]
    else:
        fragments = record["texts"]
    time = int(record["time0"])
    for index, fragment in enumerate(fragments):
        if index:
            time += record["dt"][index - 1]
        if bool(fragment.strip()) if visible else fragment != "":
            return time
    return None


def assistant_stream_first_token_time(stream: Sequence[Mapping[str, Any]]) -> int | None:
    for record in stream:
        time = (
            record["time"]
            if record["type"] == "chunk" and is_token_delta(record["chunk"])
            else _first_member_time(record, visible=False)
            if record["type"] != "chunk"
            else None
        )
        if time is not None:
            return int(time)
    return None


def assistant_stream_has_visible_content(stream: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        is_visible_chunk(record["chunk"])
        if record["type"] == "chunk"
        else _first_member_time(record, visible=True) is not None
        for record in stream
    )


def assistant_stream_has_visible_text(stream: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        (
            record["type"] == "text-chunks"
            and any(isinstance(text, str) and bool(text.strip()) for text in record["texts"])
        )
        or (record["type"] == "chunk" and chunk_has_visible_text(record["chunk"]))
        for record in stream
    )


def last_assistant_stream_chunk(
    stream: Sequence[Mapping[str, Any]], type_: str
) -> Mapping[str, Any] | None:
    for record in reversed(stream):
        if record["type"] == "chunk" and record["chunk"].get("type") == type_:
            return cast(Mapping[str, Any], record["chunk"])
    return None


def assistant_stream_chunks(
    stream: Sequence[Mapping[str, Any]], type_: str
) -> Sequence[Mapping[str, Any]]:
    return tuple(
        record["chunk"]
        for record in stream
        if record["type"] == "chunk" and record["chunk"].get("type") == type_
    )


def join_assistant_stream_text(stream: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for record in stream:
        if record["type"] == "text-chunks":
            parts.extend(record["texts"])
        elif record["type"] == "chunk" and record["chunk"].get("type") == "text-delta":
            parts.append(record["chunk"]["text"])
    return "".join(parts)


def assemble_assistant_stream(
    stream: Sequence[Mapping[str, Any]], assembler: BlockAssembler | None = None
) -> BlockAssembler:
    target = assembler or BlockAssembler()
    for record in stream:
        kind = record["type"]
        if kind == "chunk":
            target.push(dict(record["chunk"]))
        elif kind == "text-chunks":
            target.push(
                {"type": "text-delta", "index": record["index"], "text": "".join(record["texts"])}
            )
        elif kind == "reasoning-chunks":
            target.push(
                {
                    "type": "reasoning-delta",
                    "index": record["index"],
                    "text": "".join(record["texts"]),
                }
            )
        else:
            chunk = {
                "type": "tool-call-delta",
                "index": record["index"],
                "id": record["id"],
                "argumentsDelta": "".join(record["args"]),
            }
            if "name" in record:
                chunk["name"] = record["name"]
            target.push(chunk)
    return target
