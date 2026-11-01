"""Stateful reconstruction of historical tool definitions for request projection.

Upstream: packages/core/session/src/tool-history.ts
"""

from __future__ import annotations

from collections.abc import Sequence

from deskagents.llm import ToolHistory, ToolSchema
from deskagents.util.values import deep_freeze, stringify_json

from .types import SessionEvent, SessionSeq


class ToolHistoryProjection:
    """Folds committed headers and developer messages independently of model capability."""

    def __init__(self) -> None:
        self._headers: dict[SessionSeq, Sequence[ToolSchema]] = {}
        self._declared: dict[str, ToolSchema] = {}
        self._active: Sequence[ToolSchema] = ()
        self._available: set[str] = set()
        self._baseline_seq: SessionSeq | None = None
        self._history: ToolHistory = deep_freeze({"tools": [], "updates": []})

    def apply(self, event: SessionEvent) -> None:
        """Consume the next committed event in log order."""
        if event["type"] == "request/header":
            data = event["data"]
            tools = data["header"].get("tools") or []
            self._headers[event["seq"]] = tools
            redeclared = any(
                (before := self._declared.get(tool["name"])) is not None
                and stringify_json(before) != stringify_json(tool)
                for tool in tools
            )
            if (
                self._baseline_seq is None
                or data["reason"] == "series"
                or data.get("startsSeries")
                or redeclared
            ):
                self._baseline_seq = event["seq"]
                self._declared = {tool["name"]: tool for tool in tools}
                self._history = deep_freeze({"tools": list(tools), "updates": []})
                self._available = {tool["name"] for tool in tools}
            self._active = tools
        elif event["type"] == "developer/message":
            data = event["data"]
            message = data["message"]
            header_seq = data.get("headerSeq")
            definitions = [] if header_seq is None else self._headers.get(header_seq)
            additions = []
            for block in message["content"]:
                if block["type"] != "tool-addition":
                    continue
                tool = next((t for t in definitions or () if t["name"] == block["toolName"]), None)
                if tool is None:
                    raise ValueError(f"tool history: missing definition for {block['toolName']}")
                additions.append(tool)
            for tool in additions:
                self._declared[tool["name"]] = tool
            for block in message["content"]:
                if block["type"] == "tool-addition":
                    self._available.add(block["toolName"])
                elif block["type"] == "tool-removal":
                    self._available.discard(block["toolName"])
            self._history = deep_freeze(
                {
                    "tools": self._history["tools"],
                    "updates": [
                        *self._history["updates"],
                        {"messageId": message["id"], "additions": additions},
                    ],
                }
            )

    def snapshot(self) -> ToolHistory:
        """Read an immutable snapshot; later events do not change it."""
        if len(self._active) != len(self._available) or any(
            tool["name"] not in self._available for tool in self._active
        ):
            return deep_freeze({"tools": list(self._active), "updates": []})
        return self._history
