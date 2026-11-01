"""Reusable lifecycle contract for external deskagents plugin families.

The contract is capability-neutral. A plugin package supplies a fresh host and the
steps that mount and unmount its family on that host. The contract itself compares
the host before mounting and after unloading: host-owned effects, listeners of the
harness events, and the contents of the core registries. A registration that
outlives its plugin therefore fails the contract even when the plugin's own probes
pass. Capability probes add the family's own checks on top.
"""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from deepseek_cordis import Context

__all__ = [
    "HARNESS_EVENTS",
    "PLUGIN_CONTRACT_SUITE",
    "PluginContractBackend",
    "PluginContractFactory",
    "PluginContractResult",
    "host_snapshot",
    "plugin_contract_cases",
    "run_plugin_contract",
]

PLUGIN_CONTRACT_SUITE = "External plugin family contract"

HARNESS_EVENTS = (
    "agent/assistant-stream",
    "agent/created",
    "agent/disposed",
    "agent/error",
    "agent/inbox/claimed",
    "agent/inbox/spliced",
    "agent/pre-step",
    "agent/request",
    "agent/request-error",
    "agent/status",
    "agent/turn-stopping",
    "approval/request",
    "internal/plugin",
    "internal/status",
    "llm/adapters-updated",
    "llm/models-updated",
    "llm/stream",
    "session/created",
    "session/disposed",
    "session/event",
    "session/flush",
    "tools/change",
    "tools/execute",
    "tools/post-execute",
    "tools/pre-execute",
    "tools/result",
)
"""Events whose listener counts the contract compares; extend with ``events``."""

PluginContractProbe = Callable[[], object]


async def _noop() -> None:
    return None


@dataclass
class PluginContractBackend:
    """One fresh host and the steps that mount and unmount a plugin family on it.

    ``ctx`` is the host's root context, created with the core bundle (or whatever the
    family needs) but without the family. ``mount`` loads the family and
    ``dispose_plugin`` unloads only it; both may be called again for a remount.
    ``dispose`` tears down the complete host and must be safe after either.
    ``services`` names the services the family provides; ``events`` adds event names
    whose listener counts the contract compares.
    """

    ctx: Context
    mount: PluginContractProbe
    dispose_plugin: PluginContractProbe
    dispose: PluginContractProbe
    services: Sequence[str] = ()
    events: Sequence[str] = ()
    assert_active: PluginContractProbe = _noop
    exercise: PluginContractProbe = _noop
    assert_disposed: PluginContractProbe = _noop


PluginContractFactory = Callable[[], Awaitable[PluginContractBackend]]
PluginContractCheck = Callable[[PluginContractFactory], Awaitable[None]]


@dataclass(frozen=True)
class PluginContractResult:
    title: str
    outcome: Literal["passed", "failed"]
    error: BaseException | None = None
    detail: str = ""


async def _invoke(callback: PluginContractProbe) -> object:
    result = callback()
    return await result if inspect.isawaitable(result) else result


async def _read(read: Callable[[], object]) -> object:
    try:
        return await _invoke(read)
    except Exception as error:
        return f"unreadable: {type(error).__name__}"


async def host_snapshot(ctx: Context, events: Sequence[str] = ()) -> dict[str, object]:
    """What a family must leave unchanged once it unloads, read through public APIs."""
    snapshot: dict[str, object] = {
        "host effects": sorted(effect.label for effect in ctx.fiber.get_effects()),
        "listeners": {
            name: len(ctx.events.dispatch("emit", name))
            for name in sorted({*HARNESS_EVENTS, *events})
        },
    }
    tools: Any = ctx.get("tools")
    if tools is not None:
        snapshot["tools"] = await _read(
            lambda: sorted(definition["name"] for definition in tools.definitions())
        )
    prompt: Any = ctx.get("systemPrompt")
    if prompt is not None:

        async def sections() -> list[str]:
            assembly = await prompt.assemble({})
            return [section["name"] for section in assembly["sections"]]

        snapshot["prompt sections"] = await _read(sections)
    llm: Any = ctx.get("llm")
    if llm is not None:
        snapshot["model providers"] = await _read(
            lambda: sorted(provider["id"] for provider in llm.list_providers())
        )
    agents: Any = ctx.get("agents")
    if agents is not None:
        snapshot["agents"] = await _read(lambda: sorted(agent.id for agent in agents.list()))
    sessions: Any = ctx.get("sessions")
    if sessions is not None:
        snapshot["sessions"] = await _read(
            lambda: sorted(session.id for session in sessions.list())
        )
    projections: Any = ctx.get("sessionProjections")
    if projections is not None:
        snapshot["projection keys"] = await _read(lambda: sorted(projections.summary_keys()))
    return snapshot


def _difference(before: Mapping[str, object], after: Mapping[str, object]) -> str:
    lines = [
        f"{key}: {before.get(key)!r} -> {after.get(key)!r}"
        for key in sorted({*before, *after})
        if before.get(key) != after.get(key)
    ]
    return "; ".join(lines)


def _assert_services(backend: PluginContractBackend, present: bool) -> None:
    for name in backend.services:
        provided = backend.ctx.get(name) is not None
        state = "missing while mounted" if present else "still provided after unload"
        assert provided is present, f'service "{name}" is {state}'


async def _assert_unchanged(
    backend: PluginContractBackend, before: Mapping[str, object], when: str
) -> None:
    after = await host_snapshot(backend.ctx, backend.events)
    assert after == before, f"{when} left the host changed: {_difference(before, after)}"


async def _mounts(factory: PluginContractFactory) -> None:
    backend = await factory()
    try:
        await _invoke(backend.mount)
        _assert_services(backend, True)
        await _invoke(backend.assert_active)
    finally:
        await _invoke(backend.dispose)


async def _exercises(factory: PluginContractFactory) -> None:
    backend = await factory()
    try:
        await _invoke(backend.mount)
        await _invoke(backend.assert_active)
        await _invoke(backend.exercise)
        await _invoke(backend.assert_active)
    finally:
        await _invoke(backend.dispose)


async def _unloads(factory: PluginContractFactory) -> None:
    backend = await factory()
    try:
        before = await host_snapshot(backend.ctx, backend.events)
        await _invoke(backend.mount)
        await _invoke(backend.exercise)
        await _invoke(backend.dispose_plugin)
        await _assert_unchanged(backend, before, "unloading the family")
        _assert_services(backend, False)
        await _invoke(backend.assert_disposed)
    finally:
        await _invoke(backend.dispose)


async def _remounts(factory: PluginContractFactory) -> None:
    backend = await factory()
    try:
        before = await host_snapshot(backend.ctx, backend.events)
        await _invoke(backend.mount)
        await _invoke(backend.dispose_plugin)
        await _invoke(backend.mount)
        _assert_services(backend, True)
        await _invoke(backend.assert_active)
        await _invoke(backend.dispose_plugin)
        await _assert_unchanged(backend, before, "a mount, unload and remount")
    finally:
        await _invoke(backend.dispose)


_CASES: tuple[tuple[str, PluginContractCheck], ...] = (
    ("mounts and exposes its declared capabilities", _mounts),
    ("executes through public harness seams", _exercises),
    ("unloads without leaving host registrations, listeners or effects", _unloads),
    ("mounts again after unloading", _remounts),
)


def plugin_contract_cases() -> tuple[tuple[str, PluginContractCheck], ...]:
    """Return stable case titles and their pytest-independent checks."""
    return _CASES


async def run_plugin_contract(
    factory: PluginContractFactory,
) -> list[PluginContractResult]:
    """Run every external-plugin check and collect outcomes."""
    results: list[PluginContractResult] = []
    for title, check in _CASES:
        try:
            await check(factory)
        except Exception as error:
            results.append(
                PluginContractResult(
                    title,
                    "failed",
                    error,
                    "".join(traceback.format_exception(error)),
                )
            )
        else:
            results.append(PluginContractResult(title, "passed"))
    return results
