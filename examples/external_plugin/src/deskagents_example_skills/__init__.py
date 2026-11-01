"""Service Definition of the example skill family: one registry behind ``ctx.get("skills")``.

It follows upstream's three seam roles: this module defines the service, providers
(``config``) contribute skills, and consumers (``tool_skill``) read them. Providers
and consumers depend on this definition only, never on each other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict

from deepseek_cordis import Context, Service, current_context


class Skill(TypedDict):
    name: str
    description: str
    instructions: str


class SkillRegistry(Service):
    name = "skills"

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, config)
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill, *, ctx: Context | None = None) -> Callable[[], object]:
        """Add a skill owned by the calling plugin; it is removed when that plugin unloads."""
        owner = ctx or current_context() or self.ctx

        def setup() -> Callable[[], None]:
            if skill["name"] in self._skills:
                raise ValueError(f'skill "{skill["name"]}" is already registered')
            self._skills[skill["name"]] = skill

            def cleanup() -> None:
                self._skills.pop(skill["name"], None)

            return cleanup

        return owner.effect(setup, f"skills.register({skill['name']})")

    def list(self) -> list[Skill]:
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)


__all__ = ["Skill", "SkillRegistry"]
