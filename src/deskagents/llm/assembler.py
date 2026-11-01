"""Incremental model chunk-to-message assembly.

Attachment blocks are accepted when supplied by a completed ``block-end`` chunk.
Creating or projecting attachment-specific blocks is intentionally deferred to
the external attachment plugin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ContentBlock = Mapping[str, Any]
Message = Mapping[str, Any]
TokenUsage = Mapping[str, Any]


@dataclass
class _Partial:
    block_type: str
    text: str = ""
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments: str = ""
    block: ContentBlock | None = None


class BlockAssembler:
    """Assemble interleaved stream chunks in first-seen block-index order."""

    def __init__(self) -> None:
        self._partials: dict[int, _Partial] = {}
        self._order: list[int] = []
        self._usage: TokenUsage | None = None
        self._finish: dict[str, Any] | None = None
        self._replay_state: dict[str, Any] | None = None

    def _ensure(self, index: int, block_type: str) -> _Partial:
        partial = self._partials.get(index)
        if partial is None:
            partial = _Partial(block_type)
            self._partials[index] = partial
            self._order.append(index)
        return partial

    def push(self, chunk: dict[str, Any]) -> None:
        kind = chunk.get("type")
        if kind == "block-start":
            self._ensure(chunk["index"], chunk["blockType"])
        elif kind in ("text-delta", "reasoning-delta"):
            partial = self._ensure(chunk["index"], "text" if kind == "text-delta" else "reasoning")
            if partial.block is None:
                partial.text += chunk["text"]
        elif kind == "tool-call-delta":
            partial = self._ensure(chunk["index"], "tool-call")
            if partial.block is None:
                if "id" in chunk:
                    partial.tool_call_id = chunk["id"]
                if chunk.get("name"):
                    partial.tool_call_name = chunk["name"]
                partial.tool_call_arguments += chunk["argumentsDelta"]
        elif kind == "block-end":
            partial = self._ensure(chunk["index"], chunk["block"]["type"])
            if partial.block is None:
                partial.block = chunk["block"]
        elif kind == "usage":
            self._usage = chunk["usage"]
        elif kind == "finish":
            self._finish = chunk["reason"]
            self._replay_state = chunk.get("replayState")
        else:
            raise AssertionError(f"unreachable variant in BlockAssembler.push: {chunk!r}")

    def _must_get(self, index: int) -> _Partial:
        try:
            return self._partials[index]
        except KeyError as error:
            raise RuntimeError(
                f"BlockAssembler invariant violated: no partial for index {index}"
            ) from error

    def _assemble(self, partial: _Partial, index: int) -> ContentBlock:
        if partial.block is not None:
            return partial.block
        if partial.block_type == "text":
            return {"type": "text", "text": partial.text}
        if partial.block_type == "reasoning":
            return {"type": "reasoning", "text": partial.text}
        if partial.block_type == "tool-call":
            return {
                "type": "tool-call",
                "id": partial.tool_call_id or f"call-{index}",
                "name": partial.tool_call_name or "",
                "arguments": partial.tool_call_arguments,
            }
        raise ValueError(f'cannot assemble incomplete block of type "{partial.block_type}"')

    def _assembled(self) -> tuple[list[ContentBlock], dict[str, Any] | None]:
        all_blocks = [self._assemble(self._must_get(index), index) for index in self._order]
        kept = (
            [block.get("type") != "tool-call" for block in all_blocks]
            if self.finish.get("kind") == "max-tokens"
            else None
        )
        blocks = (
            all_blocks
            if kept is None
            else [block for block, include in zip(all_blocks, kept, strict=True) if include]
        )
        envelope = self._replay_state
        if envelope is None or "blocks" not in envelope:
            return blocks, envelope
        metadata = envelope["blocks"]
        if not isinstance(metadata, list | tuple) or len(metadata) != len(all_blocks):
            return blocks, None
        if kept is None or len(blocks) == len(all_blocks):
            return blocks, envelope
        return blocks, {
            "response": envelope.get("response"),
            "blocks": [item for item, include in zip(metadata, kept, strict=True) if include],
        }

    def blocks(self) -> list[ContentBlock]:
        return self._assembled()[0]

    def interrupted_blocks(self) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        for index in self._order:
            partial = self._must_get(index)
            kind = partial.block.get("type") if partial.block is not None else partial.block_type
            if kind not in ("text", "reasoning"):
                continue
            block = self._assemble(partial, index)
            if isinstance(block.get("text"), str) and block["text"].strip():
                blocks.append(block)
        return blocks

    @property
    def usage(self) -> TokenUsage | None:
        return self._usage

    @property
    def finish(self) -> dict[str, Any]:
        return self._finish or {"kind": "stop"}

    @property
    def replay_state(self) -> dict[str, Any] | None:
        return self._assembled()[1]

    def message(self, source: dict[str, Any]) -> Message:
        # Importing the public constructor here preserves its generated identity
        # without creating a package-initialization cycle.
        from . import create_assistant_message

        return create_assistant_message(content=self.blocks(), source=source)
