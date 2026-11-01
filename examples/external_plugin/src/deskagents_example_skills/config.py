"""Provider: contribute the skills listed in this plugin's bundle-row configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from deepseek_cordis import Context

from deskagents_example_skills import Skill, SkillRegistry

inject = ["skills"]


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> None:
    registry = cast(SkillRegistry, ctx.get("skills"))
    for skill in (config or {}).get("skills", ()):
        registry.register(cast(Skill, dict(skill)), ctx=ctx)
