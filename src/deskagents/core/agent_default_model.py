"""Deployment default model selection service.

Upstream: packages/core/agent-default-model at ce7073d.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deepseek_cordis import Context, Service


class AgentDefaultModelConfig(Service):
    name = "agentDefaultModel"

    def __init__(self, ctx: Context, config: Mapping[str, Any]) -> None:
        super().__init__(ctx, config)
        self._selection = {
            key: config[key] for key in ("provider", "model", "reasoningEffort") if key in config
        }
        if not self._selection.get("provider") or not self._selection.get("model"):
            raise ValueError("default provider and model are required")

    def current_selection(self) -> Mapping[str, Any]:
        return dict(self._selection)

    async def save_selection(self, selection: Mapping[str, Any]) -> None:
        self._selection = {
            key: selection[key]
            for key in ("provider", "model", "reasoningEffort")
            if key in selection
        }


__all__ = ["AgentDefaultModelConfig"]
