"""Ordered system-prompt, runtime-context, variable, and tool-schema registry.

Upstream: packages/core/system-prompt at ce7073d.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from deepseek_cordis import Context, Service, current_context

from deskagents.core.scope import (
    AnonymousEntries,
    NamedEntries,
    ScopedLayers,
    ScopeKey,
    scope_target,
)
from deskagents.util.values import snapshot_json_value

PERSONA_PREFIX_SECTION = "deployment:persona-prefix"
PERSONA_SUFFIX_SECTION = "deployment:persona-suffix"
TOOL_ORDER_REST = "<unlisted-tools>"

SECTION_ORDERS = {
    "HARNESS_IDENTITY": -1000,
    "DEPLOYMENT_PERSONA_PREFIX": 0,
    "PLAN_POLICY": 500,
    "TEAM_POLICY": 600,
    "PTC_ONLY": 800,
    "FILE_REFERENCE": 900,
    "TOOL_BASH": 1000,
    "TOOL_PWSH": 1010,
    "TOOL_READ": 1100,
    "TOOL_WRITE": 1200,
    "TOOL_EDIT": 1300,
    "TOOL_APPLY_PATCH": 1350,
    "TOOL_GLOB": 1400,
    "TOOL_GREP": 1500,
    "TOOL_JOBS": 1600,
    "TOOL_PTY": 1700,
    "TOOL_WEB_SEARCH": 2000,
    "TOOL_WEB_FETCH": 2100,
    "TOOL_LSP": 2200,
    "TOOL_SESSION_QUERY": 2300,
    "TOOL_GOAL": 2400,
    "TOOL_SCHEDULE": 2500,
    "TOOL_WORKFLOW": 2600,
    "TOOL_RALPH": 2700,
    "TOOL_SUBAGENT": 2800,
    "TOOL_REPORT": 2900,
    "TOOL_COMPUTER_USE": 3000,
    "MCP_SERVERS": 3100,
    "TOOLS_SDK": 5000,
    "DELIVERABLE_FILE_REFERENCES": 9000,
    "STRUCTURED_OUTPUT": 9900,
    "HARNESS_SOURCE": 10000,
    "WEB_SURFACE": 10100,
    "DEPLOYMENT_PERSONA_SUFFIX": 10200,
}
CONTEXT_ORDERS = {
    "WORKING_DIRECTORY": 100,
    "SANDBOX_POLICY": 110,
    "APPROVAL_POLICY": 115,
    "SUBAGENT_DELEGATION": 120,
    "TEAM_IDENTITY": 125,
    "HARNESS_SOURCE": 200,
    "WEB_SURFACE": 210,
}
_VARIABLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_GROUP = re.compile(r"\{\{([^{}]*)\}\}")


class _Layer:
    def __init__(self, scope: ScopeKey | None) -> None:
        suffix = " globally" if scope is None else " in this scope"
        self.sections: NamedEntries[Mapping[str, Any]] = NamedEntries(
            lambda name: ValueError(f'prompt section "{name}" exists{suffix}')
        )
        self.contexts: NamedEntries[Mapping[str, Any]] = NamedEntries(
            lambda name: ValueError(f'prompt context "{name}" exists{suffix}')
        )
        self.variables: NamedEntries[Callable[[Mapping[str, Any]], str | None]] = NamedEntries(
            lambda name: ValueError(f'prompt variable "{name}" exists{suffix}')
        )
        self.runtime_context_suppressors: AnonymousEntries[bool] = AnonymousEntries()
        self.tool_providers: AnonymousEntries[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = (
            AnonymousEntries()
        )

    def is_empty(self) -> bool:
        return all(
            entries.is_empty()
            for entries in (
                self.sections,
                self.contexts,
                self.variables,
                self.runtime_context_suppressors,
                self.tool_providers,
            )
        )


def _interpolate(item: Mapping[str, Any], variables: Mapping[str, str | None], kind: str) -> str:
    text = cast(str, item["text"])
    result, last = "", 0
    for match in _GROUP.finditer(text):
        name = match.group(1)
        if not _VARIABLE_NAME.fullmatch(name):
            raise ValueError(
                f'malformed prompt variable reference "{{{{{name}}}}}" in {kind} "{item["name"]}"'
            )
        if name not in variables:
            known = ", ".join(variables) or "(none)"
            raise ValueError(
                f'unknown prompt variable "{{{{{name}}}}}" in {kind} "{item["name"]}"; '
                f"registered variables: {known}"
            )
        value = variables[name]
        if value is None:
            raise ValueError(
                f'prompt variable "{{{{{name}}}}}" has no value for this assembly '
                f'({kind} "{item["name"]}")'
            )
        result += text[last : match.start()] + value
        last = match.end()
    return result + text[last:]


def render_prompt(assembly: Mapping[str, Any]) -> str:
    return "\n\n".join(
        text
        for section in assembly["sections"]
        if (
            text := (
                section["text"]
                if section.get("interpolate") is False
                else _interpolate(section, assembly["variables"], "section")
            )
        )
    )


def render_context_sections(assembly: Mapping[str, Any]) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for context in assembly["contexts"]:
        text = (
            context["text"]
            if context.get("interpolate") is False
            else _interpolate(context, assembly["variables"], "context")
        )
        if text:
            rendered.append({"name": context["name"], "text": text})
    return rendered


def join_context_sections(sections: Sequence[Mapping[str, str]]) -> str:
    body = "\n\n".join(section["text"] for section in sections)
    return (
        "Current runtime context. This snapshot supersedes earlier runtime-context "
        f"snapshots.\n\n{body}"
        if body
        else ""
    )


def render_context_snapshot(assembly: Mapping[str, Any]) -> str:
    return join_context_sections(render_context_sections(assembly))


class SystemPrompt(Service):
    name = "systemPrompt"

    def __init__(self, ctx: Context, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(ctx, config)
        config = config or {}
        self._layers = ScopedLayers(_Layer, self._emit_change)
        order = config.get("toolOrder")
        if order is not None:
            if len(set(order)) != len(order):
                raise ValueError("toolOrder lists a name more than once")
            if TOOL_ORDER_REST not in order:
                raise ValueError(f'toolOrder must contain the "{TOOL_ORDER_REST}" rest entry')
        self._tool_order: Sequence[str] | None = order
        if config.get("includeHarnessIdentity", True):
            self.section(
                {
                    "name": "harness:identity",
                    "order": self.get_section_order("HARNESS_IDENTITY"),
                    "text": "You are an AI agent powered by DeepSeek Harness.",
                },
                ctx=ctx,
            )
        self.section(
            {
                "name": PERSONA_PREFIX_SECTION,
                "order": self.get_section_order("DEPLOYMENT_PERSONA_PREFIX"),
                "text": config.get("personaPrefix", ""),
            },
            ctx=ctx,
        )
        self.section(
            {
                "name": PERSONA_SUFFIX_SECTION,
                "order": self.get_section_order("DEPLOYMENT_PERSONA_SUFFIX"),
                "text": config.get("personaSuffix", ""),
            },
            ctx=ctx,
        )
        if not config.get("includeRuntimeContext", True):
            self.suppress_runtime_context(ctx=ctx)

    def caller(self, ctx: Context | None = None) -> Context:
        candidate = ctx or current_context()
        return candidate if candidate is not None and candidate.root is self.ctx.root else self.ctx

    def _emit_change(self) -> None:
        self.ctx.emit("system-prompt/change")

    def section(
        self, section: Mapping[str, Any], *, ctx: Context | None = None
    ) -> Callable[[], object]:
        if not math.isfinite(section["order"]):
            raise TypeError(f'prompt section "{section["name"]}" order must be finite')
        return self._layers.effect(
            self.caller(ctx),
            lambda layer: layer.sections.insert(section["name"], section),
            label="systemPrompt.section()",
        )

    def context(
        self, context: Mapping[str, Any], *, ctx: Context | None = None
    ) -> Callable[[], object]:
        if not math.isfinite(context["order"]):
            raise TypeError(f'prompt context "{context["name"]}" order must be finite')
        return self._layers.effect(
            self.caller(ctx),
            lambda layer: layer.contexts.insert(context["name"], context),
            label="systemPrompt.context()",
        )

    def suppress_runtime_context(self, *, ctx: Context | None = None) -> Callable[[], object]:
        return self._layers.effect(
            self.caller(ctx),
            lambda layer: layer.runtime_context_suppressors.append(True),
            label="systemPrompt.suppressRuntimeContext()",
        )

    def tools(
        self,
        provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        ctx: Context | None = None,
    ) -> Callable[[], object]:
        return self._layers.effect(
            self.caller(ctx),
            lambda layer: layer.tool_providers.append(provider),
            label="systemPrompt.tools()",
        )

    def variable(
        self,
        name: str,
        provider: Callable[[Mapping[str, Any]], str | None],
        *,
        ctx: Context | None = None,
    ) -> Callable[[], object]:
        if not _VARIABLE_NAME.fullmatch(name):
            raise ValueError(f'invalid prompt variable name "{name}"')
        return self._layers.effect(
            self.caller(ctx),
            lambda layer: layer.variables.insert(name, provider),
            label="systemPrompt.variable()",
        )

    def get_section_order(self, name: str) -> int:
        return SECTION_ORDERS[name]

    def get_context_order(self, name: str) -> int:
        return CONTEXT_ORDERS[name]

    def _suppressed(self, scope: ScopeKey | None) -> bool:
        return not self._layers.global_.runtime_context_suppressors.is_empty() or any(
            not layer.runtime_context_suppressors.is_empty()
            for layer in self._layers.chain_layers(scope)
        )

    @staticmethod
    def _resolve(entry: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
        text = entry["text"](context) if callable(entry["text"]) else entry["text"]
        return {
            "name": entry["name"],
            "text": text,
            **({"interpolate": entry["interpolate"]} if "interpolate" in entry else {}),
        }

    def refresh_context(
        self, assembly: Mapping[str, Any], context: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        context = context or {}
        providers = self._layers.merge(context.get("scope"), lambda layer: layer.contexts)
        suppressed = self._suppressed(context.get("scope"))
        required = {
            name: entry for name, entry in providers.items() if entry.get("required") is True
        }
        contexts: list[Mapping[str, Any]] = []
        for accepted in assembly["contexts"]:
            provider = providers.get(accepted["name"])
            required.pop(accepted["name"], None)
            if suppressed and (provider or {}).get("required") is not True:
                continue
            contexts.append(accepted if provider is None else self._resolve(provider, context))
        for provider in required.values():
            contexts.append(self._resolve(provider, context))
        contexts.sort(key=lambda item: providers.get(item["name"], {}).get("order", 0))
        return {**assembly, "contexts": contexts}

    async def assemble(self, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        context = context or {}
        scope = context.get("scope")
        layers = self._layers.chain_layers(scope)
        variables: dict[str, str | None] = {}
        for name, variable_provider in self._layers.global_.variables.entries():
            variables[name] = variable_provider(context)
        for layer in layers:
            for name, variable_provider in layer.variables.entries():
                variables[name] = variable_provider(context)
        sections = self._layers.merge(scope, lambda layer: layer.sections)
        contexts = self._layers.merge(scope, lambda layer: layer.contexts)
        complete = [item for item in sections.values() if item.get("complete") is True]
        if len(complete) > 1:
            raise ValueError("multiple complete prompt sections are active")
        schemas: list[Mapping[str, Any]] = []
        known: set[str] = set()
        providers = list(self._layers.global_.tool_providers.values())
        for layer in layers:
            providers.extend(layer.tool_providers.values())
        for tool_provider in providers:
            tool_result = tool_provider(context)
            for schema in tool_result["schemas"]:
                detached = snapshot_json_value(schema)
                if not isinstance(detached, dict):
                    raise ValueError("tool schema must be lossless JSON")
                schemas.append(detached)
            known.update(
                tool_result.get("knownNames", [schema["name"] for schema in tool_result["schemas"]])
            )
        schemas = self._order_tools(schemas, known)
        assembled_sections = [
            self._resolve(item, context)
            for item in sorted(sections.values(), key=lambda item: (item["order"], item["name"]))
        ]
        suppressed = self._suppressed(scope)
        assembled_contexts = [
            self._resolve(item, context)
            for item in sorted(contexts.values(), key=lambda item: item["order"])
            if not suppressed or item.get("required") is True
        ]
        presentation = {
            "sections": assembled_sections,
            "contexts": assembled_contexts,
            "variables": variables,
        }
        transformed = self.ctx.waterfall(
            "system-prompt/assemble",
            presentation,
            context,
            next_=lambda: presentation,
            filter_=scope_target(self, scope),
        )
        if inspect.isawaitable(transformed):
            transformed = await transformed
        transformed = cast(Mapping[str, Any], transformed)
        assembly_result = {**transformed, "tools": schemas}
        if complete:
            assembly_result["sections"] = [self._resolve(complete[0], context)]
        if suppressed:
            required_names = {
                item["name"] for item in contexts.values() if item.get("required") is True
            }
            assembly_result["contexts"] = [
                item for item in assembly_result["contexts"] if item["name"] in required_names
            ]
        return assembly_result

    def _order_tools(
        self, tools: list[Mapping[str, Any]], known: set[str]
    ) -> list[Mapping[str, Any]]:
        if any(tool["name"] == TOOL_ORDER_REST for tool in tools):
            raise ValueError(f'reserved tool name "{TOOL_ORDER_REST}"')
        if self._tool_order is None:
            return sorted(tools, key=lambda tool: tool["name"])
        unknown = [
            name for name in self._tool_order if name != TOOL_ORDER_REST and name not in known
        ]
        if unknown:
            raise ValueError(f"toolOrder lists unregistered tools {unknown}")
        by_name = {tool["name"]: tool for tool in tools}
        listed = set(self._tool_order)
        rest = sorted(
            (tool for tool in tools if tool["name"] not in listed),
            key=lambda tool: tool["name"],
        )
        ordered: list[Mapping[str, Any]] = []
        for name in self._tool_order:
            if name == TOOL_ORDER_REST:
                ordered.extend(rest)
            elif name in by_name:
                ordered.append(by_name[name])
        return ordered
