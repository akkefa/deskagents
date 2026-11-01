from __future__ import annotations

import pytest
from deepseek_cordis import Context, FiberState, Service

import deskagents.bundle
from deskagents.bundle import (
    MINIMAL_BUNDLE,
    BundleBuilder,
    BundleLoader,
    apply_row_patches,
    load_bundle_rows,
)
from deskagents.session import SessionTitleService


def test_row_patches_preserve_unmentioned_fields() -> None:
    rows = [{"id": "service", "name": "example.plugin", "config": {"old": True}}]
    assert apply_row_patches(rows, [{"id": "service", "config": {"new": True}}]) == [
        {"id": "service", "name": "example.plugin", "config": {"new": True}}
    ]


def test_builder_produces_loader_rows() -> None:
    assert BundleBuilder().plugin(
        "disabled", "example.plugin", config={"value": 1}, disabled=True
    ).rows() == [
        {
            "id": "disabled",
            "name": "example.plugin",
            "config": {"value": 1},
            "disabled": True,
        }
    ]


@pytest.mark.asyncio
async def test_minimal_bundle_mounts_through_cordis_loader() -> None:
    rows = load_bundle_rows(MINIMAL_BUNDLE)
    assert [row["id"] for row in rows][-2:] == [
        "agent-loop",
        "session-checkpoint-policy",
    ]
    ctx = Context()
    loader = BundleLoader(ctx)
    batch = await loader.load(rows)
    assert set(batch.fibers) == {row["id"] for row in rows}
    assert isinstance(ctx.get("sessionTitle"), SessionTitleService)
    await loader.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_disabled_bundle_row_does_not_apply_its_plugin() -> None:
    applied: list[object] = []

    def plugin(ctx: Context, config: object) -> None:
        del ctx
        applied.append(config)

    ctx = Context()
    loader = BundleLoader(ctx, {"test.disabled": plugin})
    batch = await loader.load(
        [
            {
                "id": "disabled",
                "name": "test.disabled",
                "config": {"value": 1},
                "disabled": True,
            }
        ]
    )
    assert applied == []
    assert "disabled" not in batch.fibers
    await loader.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_explicit_plugin_map_wins_before_installed_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[str] = []

    def explicit(ctx: Context, config: object) -> None:
        del ctx, config
        applied.append("explicit")

    def installed(ctx: Context, config: object) -> None:
        del ctx, config
        applied.append("installed")

    monkeypatch.setattr(deskagents.bundle, "installed_plugin", lambda name: installed)
    ctx = Context()
    loader = BundleLoader(ctx, {"shared": explicit})
    await loader.load([{"id": "shared", "name": "shared"}])
    assert applied == ["explicit"]
    await loader.dispose()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_service_and_apply_plugins_inject_and_validate_config() -> None:
    seen: list[object] = []

    class Probe(Service):
        name = "probe"

    class Consumer:
        inject = ["probe"]

        class Config:
            @staticmethod
            def validate(value: object) -> object:
                assert isinstance(value, dict)
                return {"value": int(value["value"])}

        def apply(self, ctx: Context, config: object) -> None:
            assert isinstance(ctx.get("probe"), Probe)
            seen.append(config)

    ctx = Context()
    loader = BundleLoader(
        ctx,
        {
            "test.probe": Probe,
            "test.consumer": Consumer(),
        },
    )
    batch = await loader.load(
        [
            {
                "id": "consumer",
                "name": "test.consumer",
                "config": {"value": "7"},
            },
            {"id": "probe", "name": "test.probe"},
        ]
    )
    assert seen == [{"value": 7}]
    assert batch.fibers["consumer"].state is FiberState.ACTIVE
    assert isinstance(ctx.get("probe"), Probe)
    await batch.dispose()
    assert ctx.get("probe") is None
    await ctx.fiber.dispose()
