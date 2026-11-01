"""Provider-request content projections.

Upstream: packages/llm/llm/src/content.ts at ce7073d.

Core owns tool declaration history projection. Attachment path and image projections stay
with the external attachment plugin and can wrap the same adapter boundary later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["project_tool_updates"]

Message = Mapping[str, Any]
ToolSchema = Mapping[str, Any]
ToolHistory = Mapping[str, Any]


def _without_developer_messages(messages: Sequence[Message]) -> Sequence[Message]:
    retained = [message for message in messages if message.get("role") != "developer"]
    return messages if len(retained) == len(messages) else retained


def _tool_declarations(
    tools: Sequence[ToolSchema] | None,
    mode: str,
    history: ToolHistory,
) -> dict[str, ToolSchema]:
    declarations = {tool["name"]: tool for tool in history["tools"]}
    for update in history["updates"]:
        for tool in update["additions"]:
            name = tool["name"]
            if name not in declarations:
                declarations[name] = {**tool, "deferLoading": True}
    if mode == "addition-only":
        active = {tool["name"] for tool in (tools or ())}
        declarations = {
            name: declaration for name, declaration in declarations.items() if name in active
        }
    elif mode != "in-history":
        raise ValueError(f'unsupported tool update mode "{mode}"')
    return declarations


def project_tool_updates(
    messages: Sequence[Message],
    tools: Sequence[ToolSchema] | None,
    tool_update: str | None,
    history: ToolHistory | None = None,
) -> dict[str, Any]:
    """Project folded tool history into one route's declaration/update capabilities."""
    if tool_update is None:
        immediate: Sequence[ToolSchema] | None = tools
        if tools is not None and any(tool.get("deferLoading") is True for tool in tools):
            immediate = [
                {key: value for key, value in tool.items() if key != "deferLoading"}
                for tool in tools
            ]
        return {
            "messages": _without_developer_messages(messages),
            "tools": immediate,
        }

    if history is None:
        return {
            "messages": _without_developer_messages(messages),
            "tools": tools,
        }

    message_ids = {
        message["id"]
        for message in messages
        if message.get("role") == "developer" and "id" in message
    }
    if any(update["messageId"] not in message_ids for update in history["updates"]):
        return {
            "messages": _without_developer_messages(messages),
            "tools": tools,
        }

    declarations = _tool_declarations(tools, tool_update, history)
    update_ids = {update["messageId"] for update in history["updates"]}
    offered = {tool["name"] for tool in history["tools"] if tool.get("deferLoading") is not True}
    projected: list[Message] = []
    changed = False
    for message in messages:
        if message.get("role") != "developer":
            projected.append(message)
            continue
        if message.get("id") not in update_ids:
            changed = True
            continue
        content: list[Mapping[str, Any]] = []
        for block in message["content"]:
            block_type = block.get("type")
            if block_type == "tool-addition":
                name = block.get("toolName")
                if name not in declarations or name in offered:
                    changed = True
                    continue
                offered.add(name)
            elif block_type == "tool-removal":
                name = block.get("toolName")
                if tool_update != "in-history" or name not in offered:
                    changed = True
                    continue
                offered.remove(name)
            content.append(block)
        if not content:
            changed = True
            continue
        if len(content) == len(message["content"]):
            projected.append(message)
        else:
            changed = True
            projected.append({**message, "content": content})

    return {
        "messages": messages if not changed and len(projected) == len(messages) else projected,
        "tools": list(declarations.values()),
    }
