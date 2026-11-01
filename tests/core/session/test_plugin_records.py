"""Ported from packages/core/session/tests/plugin-records.spec.ts."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core.session import (
    PluginRecord,
    Session,
    SessionEvent,
    SessionStore,
    append_plugin_record,
    build_fork_seed,
    plugin_record_of,
    session_id,
    session_log_offset,
    session_seq,
)
from deskagents.llm import create_user_message
from deskagents.util.values import FrozenJsonObject

SPEC = "packages/core/session/tests/plugin-records.spec.ts"
GROUP = "plugin records"
SESSION_LOGGER = "deskagents.core.session.session"


def open_turn(session: Session) -> None:
    """One user turn opening, so a record lands between model-visible events."""
    session.append("turn/start", {"turn": 1})
    session.append(
        "user/message",
        create_user_message(content=[{"type": "text", "text": "hello"}], source={"kind": "user"}),
        surface_op="append",
    )


def sessions(ctx: Context) -> SessionStore:
    store = ctx.get("sessions")
    assert isinstance(store, SessionStore)
    return store


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > commits an ignorable record outside the model-visible surface"
)
def test_commits_ignorable_record_outside_surface() -> None:
    session = Session.create(session_id("record"))
    open_turn(session)
    messages = session.derive_messages()
    nodes = session.surface.nodes
    data = {"count": 1}

    seq = append_plugin_record(session, "plugin:test/state", data)
    data["count"] = 2

    assert seq == session_seq(2)
    record = session.event_at(seq)
    assert record is not None
    assert record == {
        "type": "plugin:test/state",
        "seq": 2,
        "time": record["time"],
        "data": {"count": 1},
        "ignorable": True,
    }
    assert isinstance(record, FrozenJsonObject)
    assert isinstance(record["data"], FrozenJsonObject)
    assert session.derive_messages() == messages
    assert session.surface.nodes == nodes


@pytest.mark.upstream(f"{SPEC}::{GROUP} > reads records back and leaves every other event unread")
def test_reads_records_back_and_leaves_other_events_unread() -> None:
    session = Session.create(session_id("reader"))
    open_turn(session)
    append_plugin_record(session, "plugin:test.bridge/entry_1", ["a", 2])

    events = session.snapshot_events()
    assert [plugin_record_of(event) for event in events] == [
        None,
        None,
        PluginRecord(
            type="plugin:test.bridge/entry_1",
            seq=session_seq(2),
            time=events[2]["time"],
            data=["a", 2],
        ),
    ]


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > snapshots an interface-typed request with readonly messages"
)
def test_snapshots_interface_typed_request() -> None:
    session = Session.create(session_id("typed-request"))
    request: dict[str, Any] = {
        "config": {"provider": "fixture", "model": "fixture"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Translate this fragment."}]}
        ],
    }
    seq = append_plugin_record(session, "plugin:test/request", request)
    request["config"]["model"] = "changed"

    event = session.event_at(seq)
    assert event is not None
    assert event["data"] == {
        "config": {"provider": "fixture", "model": "fixture"},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Translate this fragment."}]}
        ],
    }
    assert isinstance(event["data"], FrozenJsonObject)
    assert session.derive_messages() == []


@pytest.mark.upstream(f"{SPEC}::{GROUP} > does not read a known event, even one marked ignorable")
def test_does_not_read_known_event_even_when_ignorable() -> None:
    events: list[SessionEvent] = [
        {"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}},
        {
            "type": "turn/start",
            "seq": session_seq(1),
            "time": 2,
            "data": {"turn": 2},
            "ignorable": True,
        },
    ]

    assert [plugin_record_of(event) for event in events] == [None, None]


@pytest.mark.upstream(f"{SPEC}::{GROUP} > rejects the record type %j without changing the log")
@pytest.mark.parametrize(
    "type_",
    [
        "plugin:",
        "plugin:Test/state",
        "plugin:/state",
        "plugin:test/",
        "plugin:test//state",
        "plugin:-test",
        "plugin:test state",
        "turn/start",
    ],
)
def test_rejects_record_type_without_changing_log(type_: str) -> None:
    session = Session.create(session_id("grammar"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            f'plugin record type "{type_}" must be "plugin:" followed by lowercase '
            "slash-separated segments"
        ),
    ):
        append_plugin_record(session, type_, {})
    assert session.seq == 0


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > rejects data that is not losslessly JSON-serializable without changing "
    "the log"
)
def test_rejects_lossy_data_without_changing_log() -> None:
    session = Session.create(session_id("lossy"))

    with pytest.raises(
        ValueError,
        match=re.escape('plugin record "plugin:test/number" carries non-JSON-serializable data'),
    ):
        append_plugin_record(session, "plugin:test/number", math.nan)
    assert session.seq == 0


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > publishes a record to session/event observers and contains a reentrant "
    "record"
)
async def test_publishes_record_and_contains_reentrant_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = Context()
    await ctx.plugin(SessionStore)
    caplog.set_level(logging.WARNING, logger=SESSION_LOGGER)
    session = sessions(ctx).create(session_id("observed-record"))
    heard: list[SessionEvent] = []
    ctx.on(
        "session/event",
        lambda observed, _event: append_plugin_record(observed, "plugin:test/echo", {}),
    )
    ctx.on("session/event", lambda _observed, event: heard.append(event))

    append_plugin_record(session, "plugin:test/state", {"count": 1})

    assert [event["type"] for event in heard] == ["plugin:test/state"]
    assert session.seq == 1
    assert [record.getMessage() for record in caplog.records if record.name == SESSION_LOGGER] == [
        'session "observed-record": session/event listener threw: session append cannot '
        "reenter while another append is being published"
    ]
    await ctx.fiber.dispose()


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > carries records through a snapshot restore, a storage reopen, and a fork"
)
def test_carries_records_through_restore_reopen_and_fork() -> None:
    session = Session.create(session_id("source"))
    open_turn(session)
    append_plugin_record(session, "plugin:test/state", {"count": 1})
    events = session.snapshot_events()

    restored = Session.create(session_id("restored"), events)
    reopened = Session.from_restore(
        session_id("source"), events, session.header, session_log_offset(0), "shared-frozen"
    )
    child = Session.create(session_id("child"), build_fork_seed(events, session_seq(2)))

    for copy in [restored, reopened, child]:
        assert plugin_record_of(copy.snapshot_events()[2]) == PluginRecord(
            type="plugin:test/state",
            seq=session_seq(2),
            time=events[2]["time"],
            data={"count": 1},
        )
        assert copy.derive_messages() == session.derive_messages()
