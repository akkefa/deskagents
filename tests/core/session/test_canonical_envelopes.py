"""Ported from packages/core/session/tests/canonical-envelopes.spec.ts."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from deepseek_cordis import Context

from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    CreateSessionOptions,
    RestoredSessionOptions,
    Session,
    SessionEvent,
    SessionSeedEventState,
    SessionStore,
    adopt_session_event,
    session_id,
    session_log_offset,
    session_seq,
    snapshot_session_event,
)
from deskagents.llm import (
    ToolCallId,
    create_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
)
from deskagents.util.values import FrozenJsonArray, FrozenJsonObject, deep_freeze

SPEC = "packages/core/session/tests/canonical-envelopes.spec.ts"
PAYLOAD = "canonical event payload acceptance"
SURFACE = "canonical event-local surface metadata"

Event = dict[str, Any]
Accept = Callable[[Any], object]

ID = session_id("canonical-envelopes")
HEADER: dict[str, Any] = {
    "version": SESSION_FORMAT_VERSION,
    "id": ID,
    "createdAt": 1,
    "isSeeded": False,
}
CONFIG: dict[str, Any] = {"provider": "mock", "model": "mock", "maxTokens": 1}
FAILURE: dict[str, Any] = {"name": "ToolError", "code": "FAILED"}


def request_event(data: object) -> Event:
    return {"type": "request/header", "seq": session_seq(0), "time": 1, "data": data}


def tool_event(data: object) -> Event:
    return {
        "type": "tool/result",
        "seq": session_seq(0),
        "time": 1,
        "data": data,
        "surfaceOp": "append",
    }


def tool_data(is_error: bool) -> Event:
    return {
        "turn": 1,
        "step": 1,
        "message": create_tool_result_message(
            call_id=ToolCallId("tool"), content=[], is_error=is_error
        ),
    }


def append_event(session: Session, event: Mapping[str, Any]) -> SessionEvent:
    return session.append(
        event["type"],
        event["data"],
        surface_op=event.get("surfaceOp"),
        source_event_seqs=event.get("sourceEventSeqs"),
    )


ENTRY_PATHS: dict[str, Accept] = {
    "append": lambda event: append_event(Session.create(ID), event),
    "seed": lambda event: Session.create(ID, [event]),
    "restore": lambda event: Session.from_restore(
        ID, [event], {**HEADER}, session_log_offset(0), "detached"
    ),
    "adopt": adopt_session_event,
    "snapshot": snapshot_session_event,
}


def _invalid_data() -> list[tuple[str, Event, str]]:
    cases: list[tuple[str, Event, str]] = []
    data: object
    for data in [None, [], 1, "invalid"]:
        cases += [
            ("request data is not an object", request_event(data), "data must be an object"),
            (
                "request header is not an object",
                request_event({"header": data, "reason": "initial"}),
                "header must be an object",
            ),
            ("tool data is not an object", tool_event(data), "data must be an object"),
        ]
    cases.append(
        ("missing request header", request_event({"reason": "initial"}), "header must be an object")
    )
    optionals: list[dict[str, Any]] = [{"tools": []}, {"adapterDefaults": {}}]
    for optional in optionals:
        cases.append(
            (
                "empty optional request field",
                request_event({"header": {"config": CONFIG, **optional}, "reason": "initial"}),
                "must omit empty",
            )
        )
    system: object
    for system in ["", " ", "system", None, [], {}]:
        cases.append(
            (
                "obsolete request header system",
                request_event(
                    {"header": {"config": CONFIG, "system": system}, "reason": "initial"}
                ),
                "must omit header.system",
            )
        )
    cases.append(
        (
            "successful tool has failure metadata",
            tool_event({**tool_data(False), "error": FAILURE}),
            "error requires",
        )
    )
    absent = object()
    for message in [absent, None, [], {}, {"isError": "true"}, {"content": [{"isError": True}]}]:
        fields: Event = {"turn": 1, "step": 1}
        if message is not absent:
            fields["message"] = message
        cases.append(
            (
                "malformed tool message with failure metadata",
                tool_event({**fields, "error": FAILURE}),
                "error requires",
            )
        )
    return cases


INVALID_DATA = _invalid_data()


def user_event() -> Event:
    return {
        "type": "user/message",
        "seq": session_seq(0),
        "time": 1,
        "data": create_user_message(content=[], source={"kind": "user"}),
        "surfaceOp": "append",
    }


def model_assistant_message(text: str | None = None) -> Any:
    return create_message(
        role="assistant",
        content=[] if text is None else [{"type": "text", "text": text}],
        source={"kind": "model", "provider": "mock", "model": "mock"},
    )


def without(event: Mapping[str, Any], key: str) -> Event:
    return {name: value for name, value in event.items() if name != key}


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> None:
        self.calls.append(args)


def sessions(ctx: Context) -> SessionStore:
    store = ctx.get("sessions")
    assert isinstance(store, SessionStore)
    return store


# --- canonical event payload acceptance --------------------------------------------


@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > restores system prompts only with system-prompt producer attribution"
)
def test_restores_system_prompts_only_with_system_prompt_source() -> None:
    message = create_system_message("prompt")
    event: Event = {
        "type": "system/message",
        "seq": session_seq(0),
        "time": 1,
        "surfaceOp": "append",
        "data": {"turn": 1, "step": 1, "message": message},
    }
    for path in ["seed", "restore", "adopt", "snapshot"]:
        accept = ENTRY_PATHS[path]
        accept(copy.deepcopy(event))
        for source in [
            {"kind": "runtime-context"},
            {"kind": "plugin", "plugin": "@deepseek-ai/dsh-system-prompt"},
            {"kind": "user"},
        ]:
            invalid = {**event, "data": {**event["data"], "message": {**message, "source": source}}}
            with pytest.raises(ValueError, match="system-prompt source"):
                accept(invalid)


@pytest.mark.upstream(f"{SPEC}::{PAYLOAD} > rejects $name with a located error")
@pytest.mark.parametrize("path", list(ENTRY_PATHS))
@pytest.mark.parametrize(
    ("name", "event", "rule"),
    INVALID_DATA,
    ids=[f"{index}-{name}" for index, (name, _, _) in enumerate(INVALID_DATA)],
)
def test_rejects_invalid_data_with_a_located_error(
    path: str, name: str, event: Event, rule: str
) -> None:
    accept = ENTRY_PATHS[path]
    given = copy.deepcopy(event)
    caught: BaseException | None = None
    try:
        accept(given)
    except Exception as error:
        caught = error
    assert isinstance(caught, Exception)
    assert not isinstance(caught, TypeError)
    assert re.search(rule, str(caught))
    assert re.search(r"(?:seq|index) 0", str(caught))
    assert given == event


@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > accepts absent optional fields, nonempty fields, and optional "
    "failure identity"
)
@pytest.mark.parametrize("path", list(ENTRY_PATHS))
def test_accepts_absent_optional_fields_and_failure_identity(path: str) -> None:
    accept = ENTRY_PATHS[path]
    events = [
        request_event({"header": {"config": CONFIG}, "reason": "initial"}),
        request_event(
            {
                "header": {
                    "config": CONFIG,
                    "extension": {"nested": True},
                    "tools": [{"name": "tool", "description": "", "parameters": {}}],
                    "adapterDefaults": {"maxTokens": True},
                },
                "reason": "initial",
            }
        ),
        tool_event(tool_data(False)),
        tool_event(tool_data(True)),
        tool_event({**tool_data(True), "error": FAILURE}),
    ]
    for event in events:
        accept(copy.deepcopy(event))


@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > preserves nested header, source, and data extras without "
    "normalizing accepted events"
)
def test_preserves_nested_extras_without_normalizing() -> None:
    message = create_system_message("prompt")
    events: list[Event] = [
        request_event(
            {
                "header": {"config": CONFIG, "extra": {"nested": [True, None]}},
                "reason": "initial",
                "extra": ["retained"],
            }
        ),
        {
            "type": "system/message",
            "seq": session_seq(0),
            "time": 1,
            "surfaceOp": "append",
            "data": {
                "turn": 1,
                "step": 1,
                "message": {
                    **message,
                    "source": {**message["source"], "extra": {"nested": True}},
                },
                "extra": ["retained"],
            },
        },
    ]
    for event in events:
        for path, accept in ENTRY_PATHS.items():
            given = copy.deepcopy(event)
            accept(given)
            assert given == event
            if path in ("adopt", "snapshot"):
                assert accept(given) == event
        assert Session.create(ID, [event]).event_at(session_seq(0)) == event


@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > rejects any own header.system even when its value is undefined "
    "at adoption"
)
def test_rejects_own_header_system_even_when_none() -> None:
    event = request_event({"header": {"config": CONFIG, "system": None}, "reason": "initial"})
    with pytest.raises(ValueError, match="must omit header.system"):
        adopt_session_event(event)


@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > adopts an assistant message and freezes its identified message"
)
def test_adopts_assistant_message_and_freezes_it() -> None:
    event: Event = {
        "type": "assistant/message",
        "seq": session_seq(0),
        "time": 1,
        "surfaceOp": "append",
        "data": {"turn": 1, "step": 1, "stream": [], "message": model_assistant_message("answer")},
    }
    # Python adopts a plain event as a frozen copy; an already-frozen event is reused.
    owned = deep_freeze(copy.deepcopy(event))
    assert adopt_session_event(owned) is owned
    adopted = adopt_session_event(copy.deepcopy(event))
    assert adopted == event
    assert isinstance(adopted["data"]["message"], FrozenJsonObject)
    assert isinstance(adopted["data"]["message"]["content"], FrozenJsonArray)
    assert snapshot_session_event(event) == event


@pytest.mark.upstream(f"{SPEC}::{PAYLOAD} > rejects a non-object restored request config (%j)")
@pytest.mark.parametrize("config", [None, "invalid"])
def test_rejects_non_object_restored_request_config(config: object) -> None:
    event = request_event({"header": {"config": config}, "reason": "initial"})
    with pytest.raises(ValueError, match="lacks provider/model"):
        Session.create(ID, [event])
    with pytest.raises(ValueError, match="lacks provider/model"):
        Session.from_restore(ID, [event], {**HEADER}, session_log_offset(0), "detached")


@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > updates the frozen request-header cache only for accepted snapshots"
)
def test_updates_request_header_cache_only_for_accepted_snapshots() -> None:
    session = Session.create(ID)
    tools = [{"name": "tool", "description": "", "parameters": {}}]
    session.append(
        "request/header", {"header": {"config": CONFIG, "tools": tools}, "reason": "initial"}
    )
    first = session.request_header()
    assert first == {"config": CONFIG, "tools": tools}
    assert isinstance(first, FrozenJsonObject)
    assert session.request_header() is first
    with pytest.raises(ValueError, match="must omit header.system"):
        append_event(
            session, request_event({"header": {"config": CONFIG, "system": ""}, "reason": "change"})
        )
    assert session.request_header() is first
    session.append("request/header", {"header": {"config": CONFIG}, "reason": "change"})
    assert session.request_header() == {"config": CONFIG}
    assert first == {"config": CONFIG, "tools": tools}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > prepares owned restored events without publication (%s)"
)
@pytest.mark.parametrize("event_state", ["detached", "shared-frozen"])
async def test_prepares_owned_restored_events_without_publication(
    event_state: SessionSeedEventState,
) -> None:
    ctx = Context()
    fiber = ctx.plugin(SessionStore)
    await fiber
    try:
        store = sessions(ctx)
        # Python adopts plain storage values as frozen copies; frozen ones are reused.
        event = deep_freeze(request_event({"header": {"config": CONFIG}, "reason": "initial"}))
        created = Recorder()
        ctx.on("session/created", created)
        prepared = store.prepare(
            ID,
            RestoredSessionOptions(
                seed=[event],
                meta={**HEADER},
                inherited_event_count=session_log_offset(0),
                event_state=event_state,
            ),
        )
        assert prepared.event_at(session_seq(0)) is event
        assert prepared.request_header() == {"config": CONFIG}
        assert store.get(ID) is None
        assert created.calls == []
        invalid = request_event({"header": {"config": CONFIG, "tools": []}, "reason": "initial"})
        with pytest.raises(ValueError, match="must omit empty tools"):
            store.prepare(
                ID,
                RestoredSessionOptions(
                    seed=[invalid],
                    meta={**HEADER},
                    inherited_event_count=session_log_offset(0),
                    event_state=event_state,
                ),
            )
        assert store.get(ID) is None
        assert created.calls == []
    finally:
        await fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > does not publish rejected appends or seeds or change derived state"
)
async def test_does_not_publish_rejected_appends_or_seeds() -> None:
    ctx = Context()
    fiber = ctx.plugin(SessionStore)
    await fiber
    try:
        store = sessions(ctx)
        session = store.create(ID)
        created = Recorder()
        published = Recorder()
        ctx.on("session/created", created)
        ctx.on("session/event", published)
        events = session.snapshot_events()
        messages = session.derive_messages()
        for _, event, _ in INVALID_DATA:
            with pytest.raises(Exception):  # noqa: B017 -- upstream asserts any throw
                append_event(session, event)
            with pytest.raises(Exception):  # noqa: B017 -- upstream asserts any throw
                store.create(session_id("invalid-seed"), CreateSessionOptions(seed=[event]))
            assert store.get(session_id("invalid-seed")) is None
            assert session.seq == 0
            assert session.snapshot_events() is events
            assert list(session.surface.nodes) == []
            assert session.surface.replace_generation == 0
            assert session.derive_messages() == messages
            assert session.request_header() is None
        assert created.calls == []
        assert published.calls == []
    finally:
        await fiber.dispose()


@pytest.mark.upstream(
    f"{SPEC}::{PAYLOAD} > reports malformed restored event envelopes without TypeError (%j)"
)
@pytest.mark.parametrize("value", [None, [], 1, "invalid"])
def test_reports_malformed_restored_envelopes_without_type_error(value: object) -> None:
    with pytest.raises(ValueError, match="seed event at index 0 has an invalid event envelope"):
        Session.from_restore(ID, [value], {**HEADER}, session_log_offset(0), "detached")  # type: ignore[list-item]


# --- canonical event-local surface metadata ----------------------------------------

INVALID_OPS: list[dict[str, Any]] = [
    {"op": "replace", "start": 0, "end": 0},
    {"op": "replace", "start": 0, "endSeq": 0},
    {"op": "replace", "startSeq": 0, "end": 0},
    {"op": "replace", "startSeq": 0, "endSeq": 0, "start": 0},
    {"op": "replace", "startSeq": 0, "endSeq": 0, "end": 0},
    {"op": "replace", "startSeq": 0, "endSeq": 0, "extra": True},
    {"op": "replace", "startSeq": -1, "endSeq": 0},
]


@pytest.mark.upstream(f"{SPEC}::{SURFACE} >  rejects noncanonical replace keys (%j)")
@pytest.mark.parametrize("path", list(ENTRY_PATHS))
@pytest.mark.parametrize("surface_op", INVALID_OPS)
def test_rejects_noncanonical_replace_keys(path: str, surface_op: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="invalid replace surfaceOp"):
        ENTRY_PATHS[path]({**user_event(), "surfaceOp": surface_op})


@pytest.mark.upstream(
    f"{SPEC}::{SURFACE} >  requires markers and forbids non-surface and assistant "
    "source-event references"
)
@pytest.mark.parametrize("path", list(ENTRY_PATHS))
def test_requires_markers_and_forbids_source_event_references(path: str) -> None:
    accept = ENTRY_PATHS[path]
    markerless = without(user_event(), "surfaceOp")
    with pytest.raises(ValueError, match="requires a surfaceOp marker"):
        accept(markerless)
    turn_start: Event = {
        "type": "turn/start",
        "seq": session_seq(0),
        "time": 1,
        "data": {"turn": 1},
    }
    with pytest.raises(ValueError, match="not surface-eligible"):
        accept({**turn_start, "surfaceOp": "append"})
    with pytest.raises(ValueError, match="not surface-eligible"):
        accept({**turn_start, "sourceEventSeqs": [0]})
    assistant = {
        **user_event(),
        "type": "assistant/message",
        "sourceEventSeqs": [0],
        "data": {"turn": 1, "step": 1, "stream": [], "message": model_assistant_message()},
    }
    with pytest.raises(ValueError, match="cannot carry sourceEventSeqs"):
        accept(assistant)


@pytest.mark.upstream(
    f"{SPEC}::{SURFACE} > retains opaque ignorable %s metadata without deriving messages"
)
@pytest.mark.parametrize(
    "type_", ["extension/event", "tool/code-dispatch", "tool/code-dispatch-start"]
)
def test_retains_opaque_ignorable_metadata(type_: str) -> None:
    event: Event = {
        "type": type_,
        "seq": session_seq(0),
        "time": 1,
        "data": {"nested": {"retained": True}},
        "ignorable": True,
        "surfaceOp": {"opaque": ["not", "placement"]},
        "sourceEventSeqs": {"opaque": [None, True]},
    }
    for path, accept in ENTRY_PATHS.items():
        if path == "append":
            continue
        accept(copy.deepcopy(event))
    session = Session.create(ID, [event])
    assert session.event_at(session_seq(0)) == event
    assert list(session.surface.nodes) == []
    assert session.derive_messages() == []


@pytest.mark.upstream(f"{SPEC}::{SURFACE} > rejects known log-only %s metadata even when ignorable")
@pytest.mark.parametrize(
    "type_",
    ["turn/start", "assistant/attempt", "request/context", "session/title", "tool/ptc-dispatch"],
)
def test_rejects_known_log_only_metadata_even_when_ignorable(type_: str) -> None:
    metadatas: list[dict[str, Any]] = [{"surfaceOp": "append"}, {"sourceEventSeqs": [0]}]
    for metadata in metadatas:
        event: Event = {
            "type": type_,
            "seq": session_seq(0),
            "time": 1,
            "data": {"turn": 1, "step": 1, "stream": []},
            "ignorable": True,
            **metadata,
        }
        for accept in ENTRY_PATHS.values():
            with pytest.raises(ValueError, match="not surface-eligible"):
                accept(event)


@pytest.mark.upstream(
    f"{SPEC}::{SURFACE} > requires system placement and preserves system data and "
    "source-event references on head replacements"
)
def test_requires_system_placement_and_preserves_head_replacements() -> None:
    session = Session.create(ID)
    data: Event = {
        "turn": 1,
        "step": 1,
        "message": create_system_message("head"),
        "extra": {"nested": True},
    }
    head = session.append("system/message", data, surface_op="append")
    following = session.append(
        "system/message",
        {**data, "message": create_system_message("next")},
        surface_op={"op": "replace", "startSeq": head["seq"], "endSeq": head["seq"]},
        source_event_seqs=[head["seq"]],
    )
    assert following["data"] == {**data, "message": following["data"]["message"]}
    assert list(session.surface.nodes) == [following["seq"]]
    assert session.derive_messages() == [following["data"]["message"]]
    markerless = without(head, "surfaceOp")
    for accept in ENTRY_PATHS.values():
        with pytest.raises(ValueError, match="requires a surfaceOp marker"):
            accept(markerless)


@pytest.mark.upstream(
    f"{SPEC}::{SURFACE} > adoption rejects current or future replacement endpoints (%s)"
)
@pytest.mark.parametrize("endpoint", [1, 2])
def test_adoption_rejects_current_or_future_replacement_endpoints(endpoint: int) -> None:
    event: Event = {
        **user_event(),
        "seq": session_seq(1),
        "surfaceOp": {
            "op": "replace",
            "startSeq": session_seq(endpoint),
            "endSeq": session_seq(0),
        },
    }
    with pytest.raises(ValueError, match="must reference earlier events"):
        adopt_session_event(event)
    with pytest.raises(ValueError, match="must reference earlier events"):
        snapshot_session_event(event)


@pytest.mark.upstream(
    f"{SPEC}::{SURFACE} > adoption validates only local metadata, not referenced history or "
    "endpoint order"
)
def test_adoption_validates_only_local_metadata() -> None:
    event: Event = {
        **user_event(),
        "seq": session_seq(12),
        "surfaceOp": {"op": "replace", "startSeq": session_seq(11), "endSeq": session_seq(4)},
        "sourceEventSeqs": [session_seq(11), session_seq(4)],
    }
    # Python adopts a plain event as a frozen copy; an already-frozen event is reused.
    frozen = deep_freeze(copy.deepcopy(event))
    assert adopt_session_event(frozen) is frozen
    assert adopt_session_event(event) == event
    assert snapshot_session_event(event) == event
