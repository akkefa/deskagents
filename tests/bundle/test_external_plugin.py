"""A plugin family from its own distribution mounts through bundle rows and drives a turn.

The example under examples/external_plugin is treated as an installed distribution:
its source joins ``sys.path`` and its declared ``deskagents.plugins`` entry points
are what ``importlib.metadata`` reports.
"""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from deepseek_cordis import Context

import deskagents.bundle
from deskagents.bundle import MINIMAL_BUNDLE, BundleLoader, apply_row_patches, load_bundle_rows
from deskagents.core import AgentRegistry, SystemPrompt, ToolsService, define_tool
from deskagents.llm import LlmRuntime, create_user_message
from deskagents.testing import (
    PluginContractBackend,
    ScriptedAdapter,
    run_plugin_contract,
)

PLUGIN = Path(__file__).resolve().parents[2] / "examples" / "external_plugin"


def declared_entry_points() -> list[importlib.metadata.EntryPoint]:
    project = tomllib.loads((PLUGIN / "pyproject.toml").read_text())["project"]
    group = deskagents.bundle.PLUGIN_ENTRY_POINTS
    return [
        importlib.metadata.EntryPoint(name, value, group)
        for name, value in project["entry-points"][group].items()
    ]


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> list[importlib.metadata.EntryPoint]:
    monkeypatch.syspath_prepend(str(PLUGIN / "src"))
    declared = declared_entry_points()

    def entry_points(*, group: str, name: str) -> importlib.metadata.EntryPoints:
        return importlib.metadata.EntryPoints(
            point for point in declared if point.group == group and point.name == name
        )

    monkeypatch.setattr(deskagents.bundle, "entry_points", entry_points)
    return declared


def skill_turn() -> ScriptedAdapter:
    call = {"type": "tool-call", "id": "call-1", "name": "skill"}
    return ScriptedAdapter(
        [
            [
                {
                    "type": "block-end",
                    "index": 0,
                    "block": {**call, "arguments": '{"name":"release-notes"}'},
                },
                {"type": "finish", "reason": {"kind": "stop"}},
            ],
            [
                {"type": "text-delta", "index": 0, "text": "done"},
                {"type": "finish", "reason": {"kind": "stop"}},
            ],
        ]
    )


async def external_plugin_backend() -> PluginContractBackend:
    patches = yaml.safe_load((PLUGIN / "bundle.yml").read_text())
    ctx = Context()
    loader = BundleLoader(ctx)
    await loader.load_file(MINIMAL_BUNDLE)
    tools = ctx.get("tools")
    prompt = ctx.get("systemPrompt")
    llm = ctx.get("llm")
    agents = ctx.get("agents")
    assert isinstance(tools, ToolsService)
    assert isinstance(prompt, SystemPrompt)
    assert isinstance(llm, LlmRuntime)
    assert isinstance(agents, AgentRegistry)
    llm.register_adapter(["scripted"], skill_turn(), ctx=ctx)
    families: list[Any] = []

    async def mount() -> None:
        families.append(await loader.load(apply_row_patches([], patches)))

    async def dispose_plugin() -> None:
        await families.pop().dispose()

    async def assert_active() -> None:
        assert [tool["name"] for tool in tools.definitions()] == ["skill"]
        assembly = await prompt.assemble({})
        assert "skills" in [section["name"] for section in assembly["sections"]]

    async def exercise() -> None:
        handle = await agents.create(
            {
                "sessionId": "plugin-contract",
                "agentOptions": {"provider": "scripted", "model": "scripted"},
            }
        )
        try:
            handle.agent.followup(
                create_user_message(
                    content=[{"type": "text", "text": "notes"}],
                    source={"kind": "user"},
                )
            )
            await handle.agent.when_idle()
            results = [
                event["data"]
                for event in handle.agent.session.snapshot_events()
                if event["type"] == "tool/result"
            ]
            assert results[-1]["message"]["content"][0]["text"].startswith("Group changes")
        finally:
            await handle.dispose()

    async def assert_disposed() -> None:
        assert tools.definitions() == []
        assembly = await prompt.assemble({})
        assert "skills" not in [section["name"] for section in assembly["sections"]]

    async def dispose() -> None:
        while families:
            await families.pop().dispose()
        await loader.dispose()
        await ctx.fiber.dispose()

    return PluginContractBackend(
        ctx=ctx,
        mount=mount,
        dispose_plugin=dispose_plugin,
        dispose=dispose,
        services=["skills"],
        assert_active=assert_active,
        exercise=exercise,
        assert_disposed=assert_disposed,
    )


@pytest.mark.asyncio
async def test_external_family_passes_the_reusable_plugin_contract(
    installed: list[importlib.metadata.EntryPoint],
) -> None:
    del installed
    results = await run_plugin_contract(external_plugin_backend)
    failures = [result.detail for result in results if result.outcome == "failed"]
    assert failures == []
    assert [result.outcome for result in results] == ["passed"] * 4


class LeakyPlugin:
    """Registers a tool and a listener on the host instead of on its own context."""

    inject = ["tools"]

    def __init__(self, host: Context) -> None:
        self.host = host

    def apply(self, ctx: Context, config: object = None) -> None:
        async def execute(args: Mapping[str, Any], execution: object) -> str:
            return "leaked"

        tools = ctx.get("tools")
        assert isinstance(tools, ToolsService)
        tools.register(
            define_tool(
                name="leaked",
                description="leaked",
                parameters={},
                output={
                    "schema": {"type": "string"},
                    "render": lambda args, value: [{"type": "text", "text": value}],
                },
                execute=execute,
            ),
            ctx=self.host,
        )
        self.host.on("tools/result", lambda execution, result: None)


@pytest.mark.asyncio
async def test_the_plugin_contract_detects_registrations_left_on_the_host() -> None:
    async def factory() -> PluginContractBackend:
        ctx = Context()
        loader = BundleLoader(ctx, {"test.leaky": LeakyPlugin(ctx)})
        await loader.load_file(MINIMAL_BUNDLE)
        families: list[Any] = []

        async def mount() -> None:
            families.append(await loader.load([{"id": "leaky", "name": "test.leaky"}]))

        async def dispose_plugin() -> None:
            await families.pop().dispose()

        async def dispose() -> None:
            await loader.dispose()
            await ctx.fiber.dispose()

        return PluginContractBackend(
            ctx=ctx, mount=mount, dispose_plugin=dispose_plugin, dispose=dispose
        )

    results = {result.title: result for result in await run_plugin_contract(factory)}
    assert results["mounts and exposes its declared capabilities"].outcome == "passed"
    unload = results["unloads without leaving host registrations, listeners or effects"]
    assert unload.outcome == "failed"
    message = str(unload.error)
    assert "tools: [] -> ['leaked']" in message
    assert "'tools/result': 1" in message
    assert results["mounts again after unloading"].outcome == "failed"


@pytest.mark.asyncio
async def test_installed_plugin_family_extends_the_core_bundle(
    installed: list[importlib.metadata.EntryPoint],
) -> None:
    patches = yaml.safe_load((PLUGIN / "bundle.yml").read_text())
    rows = apply_row_patches(load_bundle_rows(MINIMAL_BUNDLE), patches)
    ctx = Context()
    loader = BundleLoader(ctx)
    await loader.load(rows)
    llm = ctx.get("llm")
    assert isinstance(llm, LlmRuntime)
    adapter = skill_turn()
    llm.register_adapter(["scripted"], adapter)
    agents = ctx.get("agents")
    assert isinstance(agents, AgentRegistry)
    handle = await agents.create(
        {"sessionId": "skills", "agentOptions": {"provider": "scripted", "model": "scripted"}}
    )
    handle.agent.followup(
        create_user_message(content=[{"type": "text", "text": "notes"}], source={"kind": "user"})
    )
    await handle.agent.when_idle()

    first: Mapping[str, Any] = adapter.requests[0]
    assert [tool["name"] for tool in first["tools"]] == ["skill"]
    assert "- release-notes: Draft release notes" in json.dumps(first["messages"])
    results = [
        event["data"]
        for event in handle.agent.session.snapshot_events()
        if event["type"] == "tool/result"
    ]
    assert results[0]["message"]["content"] == [
        {"type": "text", "text": "Group changes by user impact and link each pull request."}
    ]
    await handle.dispose()
    await loader.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_disposing_the_family_removes_its_tool_and_prompt_section(
    installed: list[importlib.metadata.EntryPoint],
) -> None:
    ctx = Context()
    loader = BundleLoader(ctx)
    await loader.load_file(MINIMAL_BUNDLE)
    family = await loader.load(
        apply_row_patches([], yaml.safe_load((PLUGIN / "bundle.yml").read_text()))
    )
    tools, prompt = ctx.get("tools"), ctx.get("systemPrompt")
    assert isinstance(tools, ToolsService) and isinstance(prompt, SystemPrompt)
    assert [tool["name"] for tool in tools.definitions()] == ["skill"]
    await family.dispose()
    assert tools.definitions() == []
    assembly = await prompt.assemble({})
    assert "skills" not in [section["name"] for section in assembly["sections"]]
    await loader.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_an_unknown_name_says_how_to_provide_it(
    installed: list[importlib.metadata.EntryPoint],
) -> None:
    loader = BundleLoader(Context())
    with pytest.raises(ValueError, match='install a distribution declaring it in "deskagents'):
        await loader.load([{"id": "web", "name": "deskagents_web.tool_web"}])


def test_a_name_declared_twice_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    group = deskagents.bundle.PLUGIN_ENTRY_POINTS
    twice = importlib.metadata.EntryPoints(
        importlib.metadata.EntryPoint("shared", value, group) for value in ("a:apply", "b:apply")
    )
    monkeypatch.setattr(deskagents.bundle, "entry_points", lambda *, group, name: twice)
    with pytest.raises(ValueError, match="declared by several distributions: a:apply, b:apply"):
        deskagents.bundle.installed_plugin("shared")
