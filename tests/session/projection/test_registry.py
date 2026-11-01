"""Ported from packages/session/session-projection/tests/registry.spec.ts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Annotated, Any
from unittest.mock import Mock

import pytest
from deepseek_cordis import Context
from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    CreateSessionMeta,
    CreateSessionOptions,
    Session,
    SessionEvent,
    SessionHeader,
    SessionId,
    SessionLogOffset,
    SessionSeq,
    SessionStore,
)
from deskagents.session.projection import (
    ProjectionCheckpoint,
    ProjectionDefinition,
    ProjectionWire,
    SessionProjectionRegistry,
)

SPEC = "packages/session/session-projection/tests/registry.spec.ts"
DRIVE = "SessionProjectionRegistry drive"


class MarksView(TypedDict):
    marks: list[str]


MarksState = Any
"""``MarksView | None``; event data arrives as a frozen mapping."""

MARKS_VIEW: TypeAdapter[MarksView] = TypeAdapter(MarksView)
MARKS_STATE: TypeAdapter[MarksView | None] = TypeAdapter(MarksView | None)
COUNT: TypeAdapter[int] = TypeAdapter(Annotated[int, Field(strict=True, ge=0)])


class StableViewState(TypedDict):
    revision: Annotated[int, Field(strict=True, ge=0)]
    value: MarksView


STABLE_VIEW_STATE: TypeAdapter[StableViewState] = TypeAdapter(StableViewState)

RESTORE_HEADER: SessionHeader = {
    "version": SESSION_FORMAT_VERSION,
    "id": SessionId("projection-restore"),
    "createdAt": 0,
    "isSeeded": False,
}


def marks_unit() -> ProjectionDefinition[MarksState]:
    """Whole-value unit: latest test/mark event wins; unrelated events keep the state."""
    return ProjectionDefinition(
        key="test/marks",
        state_schema=MARKS_STATE,
        init=lambda _header, _inherited: None,
        apply=lambda state, event: event["data"] if event["type"] == "test/mark" else state,
        wire=ProjectionWire(
            view_schema=MARKS_VIEW,
            view=lambda state: state if state is not None else {"marks": []},
        ),
        state_version=1,
    )


def count_unit() -> ProjectionDefinition[int]:
    """Host-only counting unit over every event: the state changes on each apply."""
    return ProjectionDefinition(
        key="test/count",
        state_schema=COUNT,
        init=lambda _header, _inherited: 0,
        apply=lambda state, _event: state + 1,
        state_version=1,
    )


def stable_view_unit(view: Callable[[Any], object]) -> ProjectionDefinition[Any]:
    def apply(state: Any, event: SessionEvent) -> Any:
        if event["type"] == "turn/start":
            return {**state, "revision": state["revision"] + 1}
        if event["type"] == "test/mark":
            return {"revision": state["revision"] + 1, "value": event["data"]}
        return state

    return ProjectionDefinition(
        key="test/stable-view",
        state_schema=STABLE_VIEW_STATE,
        init=lambda _header, _inherited: {"revision": 0, "value": {"marks": []}},
        apply=apply,
        wire=ProjectionWire(view_schema=MARKS_VIEW, view=view),
        state_version=1,
    )


def cut_unit() -> ProjectionDefinition[int]:
    """Host-only unit whose initial state proves the exact inherited cut."""
    return ProjectionDefinition(
        key="test/cut",
        state_schema=COUNT,
        init=lambda _header, inherited: inherited,
        apply=lambda state, _event: state,
        state_version=1,
    )


def mark(session: Session, marks: list[str]) -> SessionEvent:
    return session.append("test/mark", {"marks": marks})


def projections_of(ctx: Context) -> SessionProjectionRegistry:
    service = ctx.get("sessionProjections")
    assert isinstance(service, SessionProjectionRegistry)
    return service


def event(type_: str, seq: int, time: int, data: object) -> SessionEvent:
    return {"type": type_, "seq": SessionSeq(seq), "time": time, "data": data}


def turn_end(seq: int, time: int, turn: int) -> SessionEvent:
    return event("turn/end", seq, time, {"turn": turn, "reason": {"kind": "completed"}})


STATE_SEQUENCES = (
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 1, 2),
    (0, 1, 0, 0),
    (0, 1, 0, 1),
    (0, 1, 0, 2),
    (0, 1, 1, 0),
    (0, 1, 1, 1),
    (0, 1, 1, 2),
    (0, 1, 2, 0),
    (0, 1, 2, 1),
    (0, 1, 2, 2),
    (0, 1, 2, 3),
)


def identity_sequences(length: int) -> list[list[int]]:
    sequences: list[list[int]] = []

    def visit(sequence: list[int], highest: int) -> None:
        if len(sequence) == length:
            sequences.append(sequence)
            return
        for value in range(highest + 2):
            visit([*sequence, value], max(highest, value))

    visit([0], 0)
    return sequences


def same_identities(left: list[Any], right: list[Any]) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right, strict=True))


def sequence_name(sequence: tuple[int, ...], prefix: str) -> str:
    return ",".join(f"{prefix}{value + 1}" for value in sequence)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > supplies the exact inherited cut to live, restored, and hydrated "
    "projection initialization"
)
async def test_supplies_exact_inherited_cut(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    cut = await registry.register(cut_unit())
    inherited = [
        event("turn/start", 0, 1, {"turn": 1}),
        turn_end(1, 2, 1),
    ]
    session = store.create(
        SessionId("projection-cut"),
        CreateSessionOptions(
            seed=inherited,
            inherited_event_count=SessionLogOffset(len(inherited)),
            meta=CreateSessionMeta(is_seeded=True),
        ),
    )

    assert cut.state_of(session) == len(inherited)
    restored = registry.restore(
        {}, inherited, SessionLogOffset(0), session.header, session.inherited_event_count
    )
    assert restored.checkpoint["test/cut"]["val"] == len(inherited)
    prepared = Session.create(
        SessionId("projection-cut-prepared"),
        inherited,
        {**session.header, "id": SessionId("projection-cut-prepared")},
        session.inherited_event_count,
    )
    assert registry.hydrate(prepared, {}, inherited, SessionLogOffset(0)).as_of_seq == 1
    assert cut.state_of(prepared) == len(inherited)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > drives a registered unit over committed events and snapshots the "
    "current value"
)
async def test_drives_registered_unit(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    await registry.register(marks_unit())
    mark(session, ["a"])
    mark(session, ["a", "b"])
    snapshot = registry.snapshot(session)
    assert snapshot.values["test/marks"] == {"marks": ["a", "b"]}
    assert snapshot.as_of_seq == session.seq - 1


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > builds the cell lazily from the full log for a unit registered after "
    "events flowed"
)
async def test_builds_cell_lazily(registry: SessionProjectionRegistry, store: SessionStore) -> None:
    session = store.create()
    mark(session, ["pre-registration"])
    await registry.register(marks_unit())
    assert registry.snapshot(session).values["test/marks"] == {"marks": ["pre-registration"]}
    # The lazily-built cell then continues on the live drive path.
    mark(session, ["after"])
    assert registry.snapshot(session).values["test/marks"] == {"marks": ["after"]}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > serves init-derived state and asOfSeq -1 for an empty log"
)
async def test_empty_log(registry: SessionProjectionRegistry, store: SessionStore) -> None:
    session = store.create()
    await registry.register(marks_unit())
    snapshot = registry.snapshot(session)
    assert snapshot.as_of_seq == -1
    assert snapshot.values["test/marks"] == {"marks": []}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > notifies onChanged with the validated view and the causing seq, and "
    "skips same-reference applies"
)
async def test_notifies_on_changed(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    await registry.register(marks_unit())
    seen: list[dict[str, Any]] = []
    registry.on_changed(
        lambda changed, key, value, seq: seen.append(
            {"key": key, "value": value, "seq": seq, "sessionId": str(changed.id)}
        )
    )
    marked = mark(session, ["a"])
    # Non-matching event: apply returns the same state, so no notification.
    session.append("turn/start", {"turn": 1})
    assert seen == [
        {
            "key": "test/marks",
            "value": {"marks": ["a"]},
            "seq": marked["seq"],
            "sessionId": str(session.id),
        }
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{DRIVE} > does not compute a view while no change listener exists")
async def test_no_view_without_listener(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    view = Mock(side_effect=lambda state: state["value"])
    handle = await registry.register(stable_view_unit(view))

    session.append("turn/start", {"turn": 1})
    session.append("turn/start", {"turn": 2})

    assert handle.state_of(session)["revision"] == 2
    view.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > publishes the first observed view and suppresses later same-reference views"
)
async def test_publishes_first_observed_view(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    view = Mock(side_effect=lambda state: state["value"])
    await registry.register(stable_view_unit(view))

    seen: list[object] = []
    registry.on_changed(
        lambda _session, key, value, _seq: seen.append(value) if key == "test/stable-view" else None
    )

    session.append("turn/start", {"turn": 1})
    session.append("turn/start", {"turn": 2})

    assert seen == [{"marks": []}]
    assert view.call_count == 2

    mark(session, ["changed"])
    assert seen == [{"marks": []}, {"marks": ["changed"]}]
    assert view.call_count == 3


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > publishes the first view after an unobserved state change"
)
async def test_publishes_first_view_after_unobserved_change(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    view = Mock(side_effect=lambda state: state["value"])
    await registry.register(stable_view_unit(view))
    first: list[object] = []
    stop = registry.on_changed(
        lambda _session, key, value, _seq: (
            first.append(value) if key == "test/stable-view" else None
        )
    )

    session.append("turn/start", {"turn": 1})
    stop()
    session.append("turn/start", {"turn": 2})
    assert view.call_count == 1

    resumed: list[object] = []
    registry.on_changed(
        lambda _session, key, value, _seq: (
            resumed.append(value) if key == "test/stable-view" else None
        )
    )
    session.append("turn/start", {"turn": 3})

    assert first == [{"marks": []}]
    assert resumed == [{"marks": []}]
    assert view.call_count == 2


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > matches every four-state identity sequence across listener gaps and "
    "raw-view identities"
)
async def test_four_state_identity_sequences(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    initial_state: Any = {"marks": ["initial"]}
    state_by_event: dict[str, Any] = {}
    # Upstream keys a Map by state identity; dicts are unhashable, so key by id() and keep
    # every state alive so no id is reused.
    view_by_state: dict[int, Any] = {}
    retained: list[Any] = []
    computed_views: list[Any] = []

    def apply(state: Any, event: SessionEvent) -> Any:
        if event["type"] != "test/mark":
            return state
        marks = event["data"]["marks"]
        token = marks[0] if marks else None
        if token is None or token not in state_by_event:
            return state
        return state_by_event[token]

    def view(state: Any) -> Any:
        value = view_by_state.get(id(state))
        if value is None:
            raise ValueError("test state lacks a raw view")
        computed_views.append(value)
        return value

    await registry.register(
        ProjectionDefinition(
            key="test/marks",
            state_schema=MARKS_STATE,
            init=lambda _header, _inherited: initial_state,
            apply=apply,
            wire=ProjectionWire(view_schema=MARKS_VIEW, view=view),
            state_version=1,
        )
    )

    failures: dict[str, Any] = {}
    mismatch_count = 0
    checked = 0
    for state_sequence in STATE_SEQUENCES:
        state_count = max(state_sequence) + 1
        for view_sequence in identity_sequences(state_count):
            for baseline_known in (False, True):
                for listener_mask in range(8):
                    scenario = str(checked)
                    checked += 1
                    states: list[Any] = [
                        {"marks": [f"state-{scenario}-{index}"]} for index in range(state_count)
                    ]
                    views: list[Any] = [{"marks": []} for _ in range(max(view_sequence) + 1)]
                    retained.extend(states)
                    for index in range(state_count):
                        view_by_state[id(states[index])] = views[view_sequence[index]]
                    for index, state_index in enumerate(state_sequence):
                        state_by_event[f"{scenario}:{index}"] = states[state_index]

                    session = store.create()
                    notifications: list[int] = []
                    stop: Callable[[], None] | None = None

                    def on_changed(
                        changed: Session,
                        key: str,
                        _value: object,
                        seq: SessionSeq,
                        session: Session = session,
                        notifications: list[int] = notifications,
                    ) -> None:
                        if changed is session and key == "test/marks":
                            notifications.append(seq)

                    def set_listening(
                        listening: bool, on_changed: Callable[..., None] = on_changed
                    ) -> None:
                        nonlocal stop
                        if listening and stop is None:
                            stop = registry.on_changed(on_changed)
                        elif not listening and stop is not None:
                            stop()
                            stop = None

                    set_listening(baseline_known)
                    mark(session, [f"{scenario}:0"])
                    computed_views.clear()
                    notifications.clear()

                    expected_views: list[Any] = []
                    expected_notifications: list[int] = []
                    comparable = views[view_sequence[state_sequence[0]]] if baseline_known else None
                    for index in range(1, len(state_sequence)):
                        listening = (listener_mask & (1 << (index - 1))) != 0
                        set_listening(listening)
                        changed = state_sequence[index] != state_sequence[index - 1]
                        if changed:
                            if listening:
                                current = views[view_sequence[state_sequence[index]]]
                                expected_views.append(current)
                                if comparable is None or comparable is not current:
                                    expected_notifications.append(index)
                                comparable = current
                            else:
                                comparable = None
                        mark(session, [f"{scenario}:{index}"])
                    set_listening(False)

                    if (
                        not same_identities(computed_views, expected_views)
                        or notifications != expected_notifications
                    ):
                        mismatch_count += 1
                        state_name = sequence_name(state_sequence, "v")
                        if state_name not in failures or (baseline_known and listener_mask == 7):
                            failures[state_name] = {
                                "state": state_name,
                                "view": ",".join(
                                    f"r{view_sequence[value] + 1}" for value in state_sequence
                                ),
                                "baseline": "known" if baseline_known else "unknown",
                                "listeners": ",".join(
                                    "off" if (listener_mask & (1 << index)) == 0 else "on"
                                    for index in range(3)
                                ),
                                "expectedViewCalls": len(expected_views),
                                "actualViewCalls": len(computed_views),
                                "expectedNotifications": expected_notifications,
                                "actualNotifications": list(notifications),
                            }
                    computed_views.clear()

    assert {
        "checked": checked,
        "mismatchCount": mismatch_count,
        "failures": list(failures.values()),
    } == {"checked": 960, "mismatchCount": 0, "failures": []}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > drives independently per session (cells are per-session watermarks)"
)
async def test_drives_independently_per_session(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    other = store.create()
    await registry.register(marks_unit())
    mark(session, ["one"])
    mark(other, ["two"])
    assert registry.snapshot(session).values["test/marks"] == {"marks": ["one"]}
    assert registry.snapshot(other).values["test/marks"] == {"marks": ["two"]}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > updates host-only units without publishing them to wire listeners"
)
async def test_host_only_units_not_published(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    await registry.register(marks_unit())
    count = await registry.register(count_unit())
    changed_keys: list[str] = []
    registry.on_changed(lambda _session, key, _value, _seq: changed_keys.append(key))
    session.append("turn/start", {"turn": 1})
    assert changed_keys == []
    assert count.state_of(session) == 1
    assert registry.snapshot(session).values == {"test/marks": {"marks": []}}


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{DRIVE} > shares one unit between registrants of the same key")
async def test_shares_one_unit(registry: SessionProjectionRegistry, store: SessionStore) -> None:
    session = store.create()
    await registry.register(marks_unit())

    # One definition already serves every session (cells are keyed by Session), and
    # registrants are per-session: an agent preset mounts the same tool package once per
    # agent.
    handle = await registry.register(marks_unit())
    assert callable(handle.state_of)
    mark(session, ["kept"])
    assert registry.snapshot(session).values["test/marks"] == {"marks": ["kept"]}


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{DRIVE} > keeps the unit until the last registrant releases it")
async def test_keeps_unit_until_last_release(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    first = await registry.register(marks_unit())
    second = await registry.register(marks_unit())
    mark(session, ["kept"])

    await first.dispose()

    # Without last-release semantics, one session ending strips the projection from every
    # other live session, because the first registrant owns the only disposer.
    assert registry.snapshot(session).values["test/marks"] == {"marks": ["kept"]}
    await second.dispose()
    assert registry.snapshot(session).values == {}


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{DRIVE} > refuses to share a key across a stateVersion change")
async def test_refuses_state_version_change(registry: SessionProjectionRegistry) -> None:
    await registry.register(marks_unit())

    # The one incompatibility a runtime comparison can name: the versioned contract says
    # the cached state shape differs, so the two cannot share cells.
    with pytest.raises(
        ValueError,
        match="already registered at stateVersion 1; refusing to share it with stateVersion 9",
    ):
        await registry.register(replace(marks_unit(), state_version=9))


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > rejects a non-integer or negative stateVersion at register time"
)
async def test_rejects_invalid_state_version(registry: SessionProjectionRegistry) -> None:
    with pytest.raises(ValueError, match="stateVersion"):
        await registry.register(replace(marks_unit(), state_version=-1))
    with pytest.raises(ValueError, match="stateVersion"):
        await registry.register(replace(marks_unit(), state_version=1.5))  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > register() disposer removes the key (with its cells) and frees it for "
    "re-registration"
)
async def test_disposer_frees_key(registry: SessionProjectionRegistry, store: SessionStore) -> None:
    session = store.create()
    dispose = await registry.register(marks_unit())
    mark(session, ["cached"])
    await dispose.dispose()
    assert registry.snapshot(session).values == {}
    await registry.register(marks_unit())
    # Fresh registration rebuilds from the log, not from a stale cell.
    assert registry.snapshot(session).values["test/marks"] == {"marks": ["cached"]}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > removes registrations and change listeners when their owning fiber "
    "unloads (HMR safety)"
)
async def test_owner_unload_removes_registrations(
    ctx: Context, registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    notifications: list[str] = []

    async def plugin(inner: Context, config: object) -> None:
        await projections_of(inner).register(marks_unit())
        projections_of(inner).on_changed(
            lambda _session, key, _value, _seq: notifications.append(key)
        )

    fiber = await ctx.plugin(SimpleNamespace(inject=["sessionProjections"], apply=plugin))
    mark(session, ["live"])
    assert notifications == ["test/marks"]
    await fiber.dispose()
    mark(session, ["after-dispose"])
    assert notifications == ["test/marks"]
    assert registry.snapshot(session).values == {}


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::{DRIVE} > retains readable state during owner teardown")
async def test_retains_state_during_teardown(
    ctx: Context, registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    observed: list[MarksState] = []

    async def plugin(inner: Context, config: object) -> None:
        handle = await projections_of(inner).register(marks_unit())

        def setup() -> Callable[[], None]:
            return lambda: observed.append(handle.state_of(session))

        inner.effect(setup)

    fiber = await ctx.plugin(SimpleNamespace(inject=["sessionProjections"], apply=plugin))
    mark(session, ["live"])
    await fiber.dispose()
    assert observed == [{"marks": ["live"]}]
    assert "test/marks" not in registry.snapshot(session).values


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > snapshot serves client views and excludes host-only state"
)
async def test_snapshot_excludes_host_only(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    await registry.register(marks_unit())
    count = await registry.register(count_unit())
    mark(session, ["a", "b"])
    values = registry.snapshot(session).values
    assert values["test/marks"] == {"marks": ["a", "b"]}
    assert "test/count" not in values
    assert count.state_of(session) == 1
    assert "test/unregistered" not in values


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > checkpoints every persisted unit with its stateVersion and per-cell "
    "watermark"
)
async def test_checkpoints_every_unit(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    await registry.register(marks_unit())
    await registry.register(replace(count_unit(), state_version=7))
    marked = mark(session, ["a"])
    rows = registry.checkpoint(session)
    assert rows["test/marks"] == {"ver": 1, "seq": marked["seq"], "val": {"marks": ["a"]}}
    assert rows["test/count"] == {"ver": 7, "seq": marked["seq"], "val": 1}
    # Empty log: init-derived state at watermark -1.
    fresh = store.create()
    assert registry.checkpoint(fresh)["test/marks"] == {"ver": 1, "seq": -1, "val": None}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > checkpoint states are detached clones — mutating them cannot corrupt "
    "the watermark cache"
)
async def test_checkpoint_states_are_detached(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    await registry.register(marks_unit())
    mark(session, ["a"])
    rows = registry.checkpoint(session)
    # A hostile (or merely careless) consumer mutates the handed-out state.
    rows["test/marks"]["val"]["marks"].append("INJECTED")
    # The registry's authoritative cell is untouched.
    assert registry.snapshot(session).values["test/marks"] == {"marks": ["a"]}
    assert registry.checkpoint(session)["test/marks"]["val"] == {"marks": ["a"]}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > restoreFloor anchors one below the lowest usable watermark and at 0 for "
    "missing or mismatched rows"
)
async def test_restore_floor(registry: SessionProjectionRegistry) -> None:
    assert registry.restore_floor({}) is None  # no unit registered
    await registry.register(marks_unit())
    await registry.register(count_unit())
    assert registry.restore_floor({}) == 0
    # Lowest usable watermark is count's 5: the anchored tail starts AT 5 (one below the
    # first needed seq 6), so the read proves seq 5 still exists.
    assert (
        registry.restore_floor(
            {
                "test/marks": {"ver": 1, "seq": SessionSeq(10), "val": {"marks": []}},
                "test/count": {"ver": 1, "seq": SessionSeq(5), "val": 6},
            }
        )
        == 5
    )
    # A version-mismatched row forces that key back to a full refold.
    assert (
        registry.restore_floor(
            {
                "test/marks": {"ver": 2, "seq": SessionSeq(10), "val": {"marks": []}},
                "test/count": {"ver": 1, "seq": SessionSeq(5), "val": 6},
            }
        )
        == 0
    )
    # A fresh (-1) row still needs the whole tail from 0.
    assert (
        registry.restore_floor(
            {
                "test/marks": {"ver": 1, "seq": -1, "val": None},
                "test/count": {"ver": 1, "seq": -1, "val": 0},
            }
        )
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > restore folds the tail past each usable row and refolds from init on "
    "version mismatch"
)
async def test_restore_folds_tail(registry: SessionProjectionRegistry) -> None:
    await registry.register(marks_unit())
    await registry.register(count_unit())
    tail = [event("test/mark", 3, 3, {"marks": ["new"]}), turn_end(4, 4, 1)]
    # marks row usable (watermark 2, tail starts at 3); count row mismatched: a mismatch
    # with base_seq > 0 cannot silently refold, it raises for a re-read.
    with pytest.raises(ValueError, match="re-read from seq 0"):
        registry.restore(
            {
                "test/marks": {"ver": 1, "seq": SessionSeq(2), "val": {"marks": ["old"]}},
                "test/count": {"ver": 99, "seq": SessionSeq(2), "val": 3},
            },
            tail,
            SessionLogOffset(3),
            RESTORE_HEADER,
            SessionLogOffset(0),
        )
    # The full-log re-read (base_seq 0) refolds the mismatched key from init.
    full = [
        event("turn/start", 0, 0, {"turn": 1}),
        event("test/mark", 1, 1, {"marks": ["old"]}),
        event("test/mark", 2, 2, {"marks": ["old", "2"]}),
        *tail,
    ]
    restored = registry.restore(
        {
            "test/marks": {"ver": 1, "seq": SessionSeq(2), "val": {"marks": ["old", "2"]}},
            "test/count": {"ver": 99, "seq": SessionSeq(2), "val": 3},
        },
        full,
        SessionLogOffset(0),
        RESTORE_HEADER,
        SessionLogOffset(0),
    )
    assert restored.snapshot.as_of_seq == 4
    assert restored.snapshot.values["test/marks"] == {"marks": ["new"]}
    assert "test/count" not in restored.snapshot.values
    # The refreshed rows sit at the served cut, ready for a durable write-back.
    assert restored.checkpoint["test/marks"] == {"ver": 1, "seq": 4, "val": {"marks": ["new"]}}
    assert restored.checkpoint["test/count"] == {"ver": 1, "seq": 4, "val": 5}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > restore over a suffix folds only past each row watermark and serves an "
    "exact empty-tail cut"
)
async def test_restore_over_suffix(registry: SessionProjectionRegistry) -> None:
    await registry.register(marks_unit())
    await registry.register(count_unit())
    rows: ProjectionCheckpoint = {
        "test/marks": {"ver": 1, "seq": SessionSeq(4), "val": {"marks": ["done"]}},
        "test/count": {"ver": 1, "seq": SessionSeq(2), "val": 3},
    }
    tail = [event("turn/start", 3, 3, {"turn": 2}), turn_end(4, 4, 2)]
    restored = registry.restore(
        rows, tail, SessionLogOffset(3), RESTORE_HEADER, SessionLogOffset(0)
    )
    assert restored.snapshot.as_of_seq == 4
    # marks already covers the tail (watermark 4): nothing re-applied.
    assert restored.snapshot.values["test/marks"] == {"marks": ["done"]}
    # count folds exactly seqs 3 and 4 on top of its checkpoint, but remains host-only.
    assert restored.checkpoint["test/count"] == {"ver": 1, "seq": 4, "val": 5}
    assert "test/count" not in restored.snapshot.values

    # Empty tail (checkpoint is current): the cut sits at base_seq - 1.
    current = registry.restore(
        {
            "test/marks": {"ver": 1, "seq": SessionSeq(4), "val": {"marks": ["done"]}},
            "test/count": {"ver": 1, "seq": SessionSeq(4), "val": 5},
        },
        [],
        SessionLogOffset(5),
        RESTORE_HEADER,
        SessionLogOffset(0),
    )
    assert current.snapshot.as_of_seq == 4
    assert "test/count" not in current.snapshot.values
    assert current.checkpoint["test/count"] == {"ver": 1, "seq": 4, "val": 5}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > viewCheckpoint serves version-matching rows without any log and skips "
    "mismatched keys"
)
async def test_view_checkpoint_matching_rows(registry: SessionProjectionRegistry) -> None:
    await registry.register(marks_unit())
    await registry.register(count_unit())
    values = registry.view_checkpoint(
        {
            "test/marks": {"ver": 1, "seq": SessionSeq(4), "val": {"marks": ["stored"]}},
            "test/count": {"ver": 99, "seq": SessionSeq(4), "val": 5},  # mismatched: absent
        }
    )
    assert values["test/marks"] == {"marks": ["stored"]}
    assert "test/count" not in values
    assert registry.view_checkpoint({}) == {}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > viewCheckpoint and restore exclude host-only state while retaining its "
    "checkpoint"
)
async def test_view_checkpoint_and_restore_exclude_host_only(
    registry: SessionProjectionRegistry,
) -> None:
    await registry.register(marks_unit())
    await registry.register(count_unit())
    rows: ProjectionCheckpoint = {
        "test/marks": {"ver": 1, "seq": SessionSeq(4), "val": {"marks": ["stored"]}},
        "test/count": {"ver": 1, "seq": SessionSeq(4), "val": 5},
    }
    assert registry.view_checkpoint(rows) == {"test/marks": {"marks": ["stored"]}}

    restored = registry.restore(rows, [], SessionLogOffset(5), RESTORE_HEADER, SessionLogOffset(0))
    assert restored.snapshot.values == {"test/marks": {"marks": ["stored"]}}
    assert restored.checkpoint["test/count"] == rows["test/count"]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > rejects version-matching rows whose state no longer matches the "
    "registered schema"
)
async def test_rejects_drifted_rows(registry: SessionProjectionRegistry) -> None:
    await registry.register(marks_unit())
    drifted: ProjectionCheckpoint = {
        "test/marks": {"ver": 1, "seq": SessionSeq(2), "val": {"marks": "not-an-array"}},
    }

    assert registry.view_checkpoint(drifted) == {}
    with pytest.raises(ValueError):
        registry.restore(drifted, [], SessionLogOffset(3), RESTORE_HEADER, SessionLogOffset(0))


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > restore rejects a row claiming events past the supplied log end "
    "(shrunk log ⇒ re-read)"
)
async def test_restore_rejects_overreaching_row(registry: SessionProjectionRegistry) -> None:
    await registry.register(count_unit())
    rows: ProjectionCheckpoint = {"test/count": {"ver": 1, "seq": SessionSeq(9), "val": 10}}
    # The anchored floor sits ON the watermark, so the tail read must return at least
    # seq 9 from an intact log...
    floor = registry.restore_floor(rows)
    assert floor == 9
    # ...an intact log serves the anchor event and the checkpoint stands as is.
    anchored = registry.restore(
        rows, [turn_end(9, 9, 2)], SessionLogOffset(9), RESTORE_HEADER, SessionLogOffset(0)
    )
    assert anchored.snapshot.values == {}
    assert anchored.checkpoint["test/count"] == {"ver": 1, "seq": 9, "val": 10}
    # ...while a log crash-repaired down to fewer events returns an empty tail: the row
    # overreaches the proven end and a tail read cannot fix this key.
    with pytest.raises(ValueError, match="re-read from seq 0"):
        registry.restore(rows, [], SessionLogOffset(9), RESTORE_HEADER, SessionLogOffset(0))
    # The full re-read discards the overreaching row and refolds from init.
    events = [event("turn/start", 0, 0, {"turn": 1}), turn_end(1, 1, 1)]
    restored = registry.restore(
        rows, events, SessionLogOffset(0), RESTORE_HEADER, SessionLogOffset(0)
    )
    assert restored.snapshot.as_of_seq == 1
    assert restored.snapshot.values == {}
    assert restored.checkpoint["test/count"] == {"ver": 1, "seq": 1, "val": 2}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{DRIVE} > fails loud when a unit view violates its own schema (async unit output "
    "is unrepresentable)"
)
async def test_fails_loud_on_view_schema_violation(
    registry: SessionProjectionRegistry, store: SessionStore
) -> None:
    session = store.create()
    # An awaitable (what an accidentally async view would return) is not the declared
    # shape: the boundary validation rejects it before it leaves.
    pending: asyncio.Future[MarksView] = asyncio.get_running_loop().create_future()
    pending.set_result({"marks": []})
    await registry.register(
        ProjectionDefinition(
            key="test/marks",
            state_schema=MARKS_STATE,
            init=lambda _header, _inherited: None,
            apply=lambda state, _event: state,
            wire=ProjectionWire(view_schema=MARKS_VIEW, view=lambda _state: pending),
            state_version=1,
        )
    )
    with pytest.raises(ValueError):
        registry.snapshot(session)
