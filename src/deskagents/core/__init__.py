"""Upstream's packages/core group: the agent API spine."""

from .agent import AgentModel, AgentRegistry, assemble_context_for, fold_consumed_work
from .agent_loop import AgentHandle, AgentLoop, ReactLoopAgent, ReactLoopInbox
from .system_prompt import (
    SystemPrompt,
    join_context_sections,
    render_context_sections,
    render_context_snapshot,
    render_prompt,
)
from .tools import NativeToolExecutor, ToolsService, define_tool

__all__ = [
    "AgentHandle",
    "AgentLoop",
    "AgentModel",
    "AgentRegistry",
    "NativeToolExecutor",
    "ReactLoopAgent",
    "ReactLoopInbox",
    "SystemPrompt",
    "ToolsService",
    "assemble_context_for",
    "define_tool",
    "fold_consumed_work",
    "join_context_sections",
    "render_context_sections",
    "render_context_snapshot",
    "render_prompt",
]
