"""The fixed-density token estimator shared by projections and measurement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from deskagents.core.session import EpochHeader
from deskagents.llm import ContentBlock, Message
from deskagents.util.values import stringify_json

CHARS_PER_TOKEN = 4
BLOCK_OVERHEAD = 4
ROLE_OVERHEAD = 4


def _density(characters: int) -> int:
    return math.ceil(characters / CHARS_PER_TOKEN)


def estimate_structural_block(block: ContentBlock) -> int:
    """Price one block by its compact JSON representation.

    The ``offloaded`` image marker is plugin bookkeeping, not request content.
    """
    value: object = block
    if block.get("type") == "image" and "offloaded" in block:
        value = {key: item for key, item in block.items() if key != "offloaded"}
    return BLOCK_OVERHEAD + _density(len(stringify_json(value)))


def estimate_content(blocks: Sequence[ContentBlock]) -> int:
    """Price content blocks under the pinned upstream heuristic."""
    tokens = 0
    for block in blocks:
        kind = block.get("type")
        if kind in ("text", "reasoning"):
            tokens += _density(len(block["text"])) + BLOCK_OVERHEAD
        elif kind == "tool-call":
            tokens += (
                _density(len(block["name"])) + _density(len(block["arguments"])) + BLOCK_OVERHEAD
            )
        else:
            tokens += estimate_structural_block(block)
    return tokens


def estimate_system_message(message: Message) -> int:
    """Price a rendered system prompt without content-block framing."""
    content = message["content"]
    if not content:
        return 0
    characters = sum(
        len(block["text"]) if block.get("type") == "text" else len(stringify_json(block))
        for block in content
    )
    return _density(characters) + ROLE_OVERHEAD


def estimate_message(message: Message) -> int:
    """Price one model-visible message."""
    if message["role"] == "system":
        return estimate_system_message(message)
    return estimate_content(message["content"]) + ROLE_OVERHEAD


def estimate_tools_tokens(header: EpochHeader | None) -> int:
    """Price the latest canonical request envelope's tool schemas."""
    if header is None:
        return 0
    tools = header.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, str | bytes) or not tools:
        return 0
    return _density(len(stringify_json(tools))) + BLOCK_OVERHEAD


def usage_tokens(usage: Mapping[str, object]) -> int:
    """Sum disjoint provider buckets without adding reasoning output twice."""
    return (
        cast(int, usage["inputTokens"])
        + cast(int, usage.get("cacheReadTokens", 0))
        + cast(int, usage.get("cacheWriteTokens", 0))
        + cast(int, usage["outputTokens"])
    )
