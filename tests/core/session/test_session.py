"""Ported from packages/core/session/tests/session.spec.ts."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, cast, overload

import pytest
from deepseek_cordis import Context

import deskagents.core.session.surface as surface_module
from deskagents.core.session import (
    SESSION_FORMAT_VERSION,
    CreateSessionMeta,
    CreateSessionOptions,
    Session,
    SessionEvent,
    SessionStore,
    SurfaceManager,
    adopt_session_event,
    session_id,
    session_log_offset,
    session_seq,
    snapshot_session_event,
)
from deskagents.llm import (
    Message,
    MessageId,
    ReasoningEffortId,
    ToolCallId,
    create_developer_message,
    create_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
)
from deskagents.util.values import MAX_SAFE_INTEGER, FrozenJsonArray, FrozenJsonObject, thaw_json

SPEC = "packages/core/session/tests/session.spec.ts"
LOGGER = "deskagents.core.session.session"

BIG = 2**64
"""Python's analog of a JavaScript ``bigint``: an int that is not lossless JSON."""


class Opaque:
    """A class instance: Python's analog of JavaScript's Symbol, function or Map values."""


def text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def user(value: str, kind: str = "user") -> Message:
    return create_user_message(content=[text(value)], source={"kind": kind})


def assistant(content: Sequence[dict[str, Any]]) -> Message:
    return create_message(
        role="assistant",
        content=content,
        source={"kind": "model", "provider": "mock", "model": "mock"},
    )


def tool_result(call_id: str, value: str) -> Message:
    return create_tool_result_message(
        call_id=ToolCallId(call_id), content=[text(value)], is_error=False
    )


def last_of(events: Sequence[SessionEvent], kind: str) -> SessionEvent:
    return next(event for event in reversed(events) if event["type"] == kind)


def is_frozen(value: object) -> bool:
    return isinstance(value, FrozenJsonObject | FrozenJsonArray)


async def mount_store() -> tuple[Context, SessionStore]:
    ctx = Context()
    await ctx.plugin(SessionStore)
    return ctx, store_of(ctx)


def store_of(ctx: Context) -> SessionStore:
    return cast(SessionStore, ctx.get("sessions"))


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


async def settle() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


class CountingSeed(Sequence[SessionEvent]):
    """A one-entry seed whose element read is counted, like an upstream accessor index."""

    def __init__(self, read: Callable[[int], SessionEvent]) -> None:
        self.read = read
        self.reads = 0

    def __len__(self) -> int:
        return 1

    @overload
    def __getitem__(self, index: int) -> SessionEvent: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[SessionEvent]: ...
    def __getitem__(self, index: int | slice) -> SessionEvent | Sequence[SessionEvent]:
        if isinstance(index, slice) or index != 0:
            raise IndexError(index)
        self.reads += 1
        return self.read(self.reads)


# --- Session -----------------------------------------------------------------------


@pytest.mark.upstream(f"{SPEC}::Session > exposes one stable readonly surface view")
def test_exposes_one_stable_surface_view() -> None:
    session = Session.create(session_id("surface-view"))
    surface = session.surface

    assert isinstance(surface, SurfaceManager)
    assert surface is session.surface


@pytest.mark.upstream(f"{SPEC}::Session > derives message history from the event log")
def test_derives_message_history() -> None:
    session = Session.create(session_id("s1"))
    session.append("turn/start", {"turn": 1})
    session.append("user/message", user("hello"), surface_op="append")
    session.append(
        "assistant/attempt",
        {
            "turn": 1,
            "step": 1,
            "stream": [{"type": "text-chunks", "time0": 1, "index": 0, "dt": [], "texts": ["hi"]}],
        },
    )
    session.append(
        "assistant/message",
        {
            "stream": [],
            "turn": 1,
            "step": 1,
            "message": assistant(
                [
                    text("let me check"),
                    {
                        "type": "tool-call",
                        "id": ToolCallId("c1"),
                        "name": "echo",
                        "arguments": "{}",
                    },
                ]
            ),
        },
        surface_op="append",
    )
    session.append(
        "tool/result",
        {"turn": 1, "step": 1, "message": tool_result("c1", "ok")},
        surface_op="append",
    )
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    messages = session.derive_messages()
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    # raw chunks must NOT appear in derived history
    assert len(messages[1]["content"]) == 2
    assert messages[2]["role"] == "tool"
    assert messages[2]["content"] == [text("ok")]


@pytest.mark.upstream(f"{SPEC}::Session > accepts and round-trips a max-tokens turn/end reason")
def test_round_trips_max_tokens_reason() -> None:
    session = Session.create(session_id("s1"))
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "max-tokens"}})

    turn_end = last_of(session.snapshot_events(), "turn/end")
    assert turn_end["data"]["reason"] == {"kind": "max-tokens"}
    # survives a deep copy (the persistence-serialization boundary)
    assert thaw_json(turn_end["data"]["reason"]) == {"kind": "max-tokens"}


@pytest.mark.upstream(f"{SPEC}::Session > round-trips an aborted turn with its cancellation cause")
def test_round_trips_aborted_turn() -> None:
    session = Session.create(session_id("aborted"))
    session.append("turn/start", {"turn": 1})
    session.append(
        "turn/end", {"turn": 1, "reason": {"kind": "aborted", "reason": {"kind": "user"}}}
    )
    replayed = Session.create(
        session_id("aborted-replay"), thaw_json(list(session.snapshot_events()))
    )
    assert replayed.snapshot_events()[:-1] == session.snapshot_events()
    turn_end = last_of(replayed.snapshot_events(), "turn/end")
    assert turn_end["data"]["reason"] == {"kind": "aborted", "reason": {"kind": "user"}}


@pytest.mark.upstream(
    f"{SPEC}::Session > renders injected-context and user messages as plain user content"
)
def test_renders_context_and_user_messages() -> None:
    session = Session.create(session_id("s2"))
    session.append("user/message", user("file changed: a.ts", "watcher"), surface_op="append")
    session.append("user/message", user("focus on tests"), surface_op="append")

    context_message, steering_message = session.derive_messages()
    assert context_message["role"] == "user"
    assert context_message["content"] == [text("file changed: a.ts")]
    assert steering_message["role"] == "user"
    assert steering_message["content"] == [text("focus on tests")]


@pytest.mark.upstream(
    f"{SPEC}::Session > keeps the exact identified context message in durable history and "
    "projection"
)
def test_keeps_exact_identified_context_message() -> None:
    session = Session.create(session_id("s2-raw"))
    message = create_user_message(
        content=[
            text("<system-reminder>Additional instructions from: pkg/AGENTS.md</system-reminder>")
        ],
        source={"kind": "agent-instructions", "form": "instructions", "changes": []},
    )
    session.append("user/message", message, surface_op="append")

    assert session.derive_messages() == [message]
    event = session.snapshot_events()[0]
    assert event["type"] == "user/message"
    assert event["data"]["source"] == {
        "kind": "agent-instructions",
        "form": "instructions",
        "changes": [],
    }


@pytest.mark.upstream(f"{SPEC}::Session > replays identically from a seeded event log")
def test_replays_identically_from_seed() -> None:
    original = Session.create(session_id("s3"))
    original.append("turn/start", {"turn": 1})
    original.append("user/message", user("q"), surface_op="append")
    original.append(
        "assistant/message",
        {"stream": [], "turn": 1, "step": 1, "message": assistant([text("a")])},
        surface_op="append",
    )
    original.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    replayed = Session.create(session_id("s3-replay"), original.snapshot_events())
    assert replayed.derive_messages() == original.derive_messages()
    # The seed verbatim, plus the end-seed event the constructor appends.
    assert replayed.snapshot_events()[: original.seq] == original.snapshot_events()
    assert replayed.seq == original.seq + 1
    assert replayed.first_live_seq == original.seq


@pytest.mark.upstream(
    f"{SPEC}::Session > marks an explicitly empty seed without marking a fresh session"
)
def test_marks_explicitly_empty_seed() -> None:
    fresh = Session.create(session_id("fresh-empty"))
    assert fresh.snapshot_events() == ()

    resumed = Session.create(session_id("resumed-empty"), [])
    assert resumed.first_live_seq == 0
    events = resumed.snapshot_events()
    assert len(events) == 1
    assert (events[0]["type"], events[0]["seq"], events[0]["data"]) == ("session/end-seed", 0, {})

    reopened = Session.create(session_id("reopened-empty"), resumed.snapshot_events())
    assert reopened.first_live_seq == 1
    assert reopened.snapshot_events() == resumed.snapshot_events()


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects pre-provider request headers and assistant messages on seed/load"
)
def test_rejects_pre_provider_seed_shapes() -> None:
    request_header = {
        "type": "request/header",
        "seq": 0,
        "time": 1,
        "data": {"header": {"config": {"model": "old-model"}}, "reason": "initial"},
    }
    with pytest.raises(ValueError, match="seed request/header at index 0 lacks provider/model"):
        Session.create(session_id("old-header"), [request_header])

    assistant_message = {
        "type": "assistant/message",
        "seq": 0,
        "time": 1,
        "data": {"turn": 1, "step": 1, "content": [text("old")]},
        "surfaceOp": "append",
    }
    with pytest.raises(
        ValueError, match="seed assistant/message at index 0 lacks an identified message"
    ):
        Session.create(session_id("old-assistant"), [assistant_message])

    malformed_header = {
        "type": "request/header",
        "seq": 0,
        "time": 1,
        "data": {"header": "old-header"},
    }
    with pytest.raises(ValueError, match="seed request/header at index 0 header must be an object"):
        Session.create(session_id("malformed-header"), [malformed_header])

    unrelated_primitive_data = {"type": "plugin/event", "seq": 0, "time": 1, "data": None}
    created = Session.create(session_id("primitive-plugin-data"), [unrelated_primitive_data])
    assert created.snapshot_events()[:1] == (unrelated_primitive_data,)


@pytest.mark.upstream(
    f"{SPEC}::Session > validates Assistant settlement fields without replaying embedded streams"
)
def test_validates_assistant_settlement_fields() -> None:
    sid = session_id("invalid-restored-assistant-stream")
    header = {
        "version": SESSION_FORMAT_VERSION,
        "id": sid,
        "createdAt": 1,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    for data in [
        None,
        {"turn": "1", "step": 1, "stream": []},
        {"turn": -1, "step": 1, "stream": []},
        {"turn": -0.0, "step": 1, "stream": []},
        {"turn": 1.5, "step": 1, "stream": []},
        {"turn": 1, "step": "1", "stream": []},
        {"turn": 1, "step": -1, "stream": []},
        {"turn": 1, "step": -0.0, "stream": []},
        {"turn": 1, "step": 1.5, "stream": []},
        {"turn": 1, "step": 1, "stream": None},
    ]:
        invalid_attempt = {"type": "assistant/attempt", "seq": 0, "time": 1, "data": data}
        with pytest.raises(ValueError, match="invalid settlement fields"):
            Session.from_restore(sid, [invalid_attempt], header, session_log_offset(0), "detached")

    mismatched_message = {
        "type": "assistant/message",
        "seq": 0,
        "time": 1,
        "data": {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "message",
                "role": "assistant",
                "content": [text("different")],
                "source": {"kind": "model", "provider": "mock", "model": "mock"},
            },
            "stream": [
                {"type": "text-chunks", "time0": 1, "index": 0, "dt": [], "texts": ["streamed"]}
            ],
        },
        "surfaceOp": "append",
    }
    restored = Session.from_restore(
        sid, [mismatched_message], header, session_log_offset(0), "detached"
    )
    # Python freezes a copy instead of the adopted object.
    assert restored.event_at(session_seq(0)) == mismatched_message
    assert type(mismatched_message) is dict


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects historical or malformed request-header lifecycle markers on "
    "seed/load"
)
def test_rejects_malformed_request_header_markers() -> None:
    base: dict[str, Any] = {
        "type": "request/header",
        "seq": session_seq(0),
        "time": 1,
        "data": {"header": {"config": {"provider": "mock", "model": "model"}}, "reason": "initial"},
    }
    for reason in ["fallback", "unknown", None]:
        invalid = copy.deepcopy(base)
        invalid["data"]["reason"] = reason
        with pytest.raises(
            ValueError, match="seed request/header at index 0 has an invalid reason"
        ):
            Session.create(session_id("invalid-header-reason"), [invalid])
    for starts_series in [False, 1, "true"]:
        invalid = copy.deepcopy(base)
        invalid["data"]["startsSeries"] = starts_series
        with pytest.raises(
            ValueError,
            match="seed request/header at index 0 has an invalid startsSeries marker",
        ):
            Session.create(session_id("invalid-series-marker"), [invalid])


_USER = {"id": "user", "role": "user", "content": [text("content")], "source": {"kind": "user"}}
_ASSISTANT = {
    "id": "assistant",
    "role": "assistant",
    "content": [text("content")],
    "source": {"kind": "model", "provider": "mock", "model": "mock"},
}
_TOOL = {
    "id": "tool",
    "role": "tool",
    "toolCallId": "call",
    "content": [text("result")],
    "source": {"kind": "tool", "callId": "call"},
}


def _message_event(kind: str, data: object) -> dict[str, object]:
    return {"type": kind, "seq": 0, "time": 1, "surfaceOp": "append", "data": data}


def _settled(message: object) -> dict[str, object]:
    return {"turn": 1, "step": 1, "message": message}


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects event-specific malformed message shapes on seed/load"
)
def test_rejects_malformed_message_shapes() -> None:
    invalid: list[tuple[str, dict[str, object], str]] = [
        ("message record", _message_event("user/message", None), "lacks an identified message"),
        (
            "user role",
            _message_event("user/message", {**_USER, "role": "assistant"}),
            'message must have role "user"',
        ),
        (
            "source",
            _message_event("user/message", {**_USER, "source": None}),
            "message has invalid source",
        ),
        (
            "content shape",
            _message_event("user/message", {**_USER, "content": "not-an-array"}),
            "message has invalid content",
        ),
        (
            "assistant source",
            _message_event(
                "assistant/message", _settled({**_ASSISTANT, "source": {"kind": "user"}})
            ),
            "message must have model source",
        ),
        (
            "tool source",
            _message_event("tool/result", _settled({**_TOOL, "source": {"kind": "user"}})),
            "message must have tool source",
        ),
        (
            "system role",
            _message_event(
                "system/message",
                _settled({**_USER, "id": "system", "source": {"kind": "system-prompt"}}),
            ),
            'message must have role "system"',
        ),
        (
            "system source",
            _message_event(
                "system/message",
                _settled(
                    {
                        **_USER,
                        "id": "system",
                        "role": "system",
                        "source": {"kind": "runtime-context"},
                    }
                ),
            ),
            "message must have system-prompt source",
        ),
        (
            "tool role",
            _message_event("tool/result", _settled({**_TOOL, "role": "user"})),
            'message must have role "tool"',
        ),
        (
            "tool correlation",
            _message_event(
                "tool/result",
                _settled({**_TOOL, "source": {"kind": "tool", "callId": "other-call"}}),
            ),
            "message has mismatched tool call ids",
        ),
    ]

    for name, event, message in invalid:
        with pytest.raises(ValueError) as raised:
            Session.create(session_id(f"invalid-{name}"), [event])
        assert message in str(raised.value), name


@pytest.mark.upstream(
    f"{SPEC}::Session > snapshots message events without validating plugin-owned block details"
)
def test_snapshots_message_events_without_plugin_block_validation() -> None:
    boundary = snapshot_session_event(
        {"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}}
    )
    assert boundary == {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}

    extended = snapshot_session_event(
        {
            "type": "user/message",
            "seq": session_seq(0),
            "time": 1,
            "surfaceOp": "append",
            "data": {
                "id": "extended-message",
                "role": "user",
                "content": [{"type": "plugin-block", "value": 1}],
                "source": {"kind": "plugin-source", "value": 1},
            },
        }
    )
    assert extended["type"] == "user/message"
    assert extended["data"]["content"] == [{"type": "plugin-block", "value": 1}]


@pytest.mark.upstream(
    f"{SPEC}::Session > adopts exclusively owned messages in place and keeps snapshots detached"
)
def test_adopts_owned_messages_and_detaches_snapshots() -> None:
    owned = {
        "type": "user/message",
        "seq": 0,
        "time": 1,
        "surfaceOp": "append",
        "data": {
            "id": "owned-message",
            "role": "user",
            "content": [text("owned")],
            "source": {"kind": "user"},
        },
    }
    # Python cannot freeze in place: adoption returns a frozen copy, reused once frozen.
    adopted = adopt_session_event(owned)
    assert adopted == owned
    assert adopt_session_event(adopted) is adopted
    assert is_frozen(adopted["data"])
    assert is_frozen(adopted["data"]["content"])

    source = copy.deepcopy(owned)
    snapshot = snapshot_session_event(source)
    assert snapshot is not source
    assert snapshot["data"] is not source["data"]
    assert snapshot["data"]["content"] is not source["data"]["content"]  # type: ignore[index]


@pytest.mark.upstream(
    f"{SPEC}::Session > adopts developer messages in place and freezes their nested tool changes"
)
def test_adopts_developer_messages_and_freezes_tool_changes() -> None:
    event = {
        "type": "developer/message",
        "seq": session_seq(1),
        "time": 1,
        "surfaceOp": "append",
        "data": {
            "turn": 1,
            "step": 1,
            "headerSeq": session_seq(0),
            "message": thaw_json(
                create_developer_message(
                    content=[{"type": "tool-addition", "toolName": "search"}],
                    source={"kind": "test"},
                )
            ),
        },
    }
    assert not is_frozen(event["data"]["message"])  # type: ignore[index]
    adopted = adopt_session_event(event)
    assert adopted == event
    assert adopt_session_event(adopted) is adopted
    assert is_frozen(adopted["data"]["message"])
    assert is_frozen(adopted["data"]["message"]["content"][0])


@pytest.mark.upstream(f"{SPEC}::Session > adopts a system/message by freezing its message")
def test_adopts_system_message_by_freezing() -> None:
    event = {
        "type": "system/message",
        "seq": session_seq(0),
        "time": 1,
        "surfaceOp": "append",
        "data": {"turn": 1, "step": 1, "message": create_system_message("You are terse.")},
    }
    adopted = adopt_session_event(event)
    assert adopted["type"] == "system/message"
    assert is_frozen(adopted["data"]["message"])
    assert is_frozen(adopted["data"]["message"]["content"][0])


@pytest.mark.upstream(f"{SPEC}::Session > validates message shape before adopting ownership")
def test_validates_message_shape_before_adopting() -> None:
    malformed = {
        "type": "user/message",
        "surfaceOp": "append",
        "seq": 0,
        "time": 1,
        "data": {
            "id": "wrong-role",
            "role": "assistant",
            "content": [],
            "source": {"kind": "user"},
        },
    }
    with pytest.raises(ValueError, match='message must have role "user"'):
        adopt_session_event(malformed)


@pytest.mark.upstream(
    f"{SPEC}::Session > round-trips a non-empty reasoning effort and rejects invalid durable values"
)
def test_round_trips_reasoning_effort() -> None:
    valid: dict[str, Any] = {
        "type": "request/header",
        "seq": session_seq(0),
        "time": 1,
        "data": {
            "header": {
                "config": {
                    "provider": "mock",
                    "model": "model",
                    "reasoningEffort": ReasoningEffortId("adapter-owned"),
                }
            },
            "reason": "initial",
        },
    }
    assert Session.create(session_id("reasoning-effort"), [valid]).snapshot_events()[0] == valid

    for reasoning_effort in ["", 1]:
        invalid = copy.deepcopy(valid)
        invalid["data"]["header"]["config"]["reasoningEffort"] = reasoning_effort
        with pytest.raises(
            ValueError, match="seed request/header at index 0 has an invalid reasoningEffort"
        ):
            Session.create(session_id("invalid-reasoning-effort"), [invalid])


@pytest.mark.parametrize(
    "adapter_defaults",
    [
        pytest.param(
            None,
            id="null",
        ),
        pytest.param([], id="array"),
        pytest.param({"unknown": True}, id="unknown-key"),
        pytest.param({"maxTokens": False}, id="false-marker"),
        pytest.param({"reasoningEffort": True}, id="unconfigured-marker"),
    ],
)
@pytest.mark.upstream(
    f"{SPEC}::Session > round-trips adapter-default markers and rejects invalid durable values"
)
def test_round_trips_adapter_default_markers(adapter_defaults: object) -> None:
    valid: dict[str, Any] = {
        "type": "request/header",
        "seq": session_seq(0),
        "time": 1,
        "data": {
            "header": {
                "config": {"provider": "mock", "model": "model", "maxTokens": 256_000},
                "adapterDefaults": {"maxTokens": True},
            },
            "reason": "initial",
        },
    }
    assert Session.create(session_id("adapter-defaults"), [valid]).snapshot_events()[0] == valid

    invalid = copy.deepcopy(valid)
    invalid["data"]["header"]["adapterDefaults"] = adapter_defaults
    with pytest.raises(
        ValueError, match="seed request/header at index 0 has invalid adapterDefaults"
    ):
        Session.create(session_id("invalid-adapter-defaults"), [invalid])


@pytest.mark.upstream(
    f"{SPEC}::Session > isolates the log from mutation through a derived message (append-only "
    "contract)"
)
def test_isolates_log_from_derived_message_mutation() -> None:
    session = Session.create(session_id("s4"))
    session.append("user/message", user("original"), surface_op="append")
    session.append(
        "tool/result",
        {"turn": 1, "step": 1, "message": tool_result("c1", "tool out")},
        surface_op="append",
    )
    before = thaw_json(list(session.snapshot_events()))

    # A misbehaving consumer tries to mutate the messages it was handed.
    messages = session.derive_messages()
    user_block = messages[0]["content"][0]
    with pytest.raises(TypeError):
        user_block["text"] = "HACKED"
    tool_message = messages[1]
    with pytest.raises(TypeError):
        tool_message["content"].append(text("injected"))
    with pytest.raises(TypeError):
        messages[0]["content"].append(text("extra"))
    # The returned LIST is the caller's own snapshot, though: reordering it
    # is the caller's business and never reaches the cache or the log.
    messages.reverse()

    # The log is unchanged: deep-equal to the snapshot taken before mutation.
    assert list(session.snapshot_events()) == before
    # And a fresh derivation still reflects the original content and order.
    assert session.derive_messages()[0]["content"] == [text("original")]


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects non-JSON-serializable event data at the source (incl. sparse "
    "arrays)"
)
def test_rejects_non_json_event_data() -> None:
    session = Session.create(session_id("s5"))

    def bad(extra: object) -> None:
        data = {"content": [text("x")], "source": {"kind": "user"}, "extra": extra}
        with pytest.raises(ValueError, match="non-JSON-serializable"):
            session.append("user/message", data, surface_op="append")

    bad(BIG)
    bad(lambda: 0)
    bad(Opaque())
    bad({1, 2})
    bad(float("inf"))
    bad(float("nan"))
    # A list carrying a non-serializable element is rejected too.
    bad([1, BIG, 3])
    # A nested non-serializable value (inside a plain dict) is rejected.
    bad({"nested": {"deep": lambda: 0}})
    # A circular reference is rejected (the seen-set guard, not a stack blow-up).
    cyclic: dict[str, object] = {"a": 1}
    cyclic["self"] = cyclic
    bad(cyclic)
    # The rejected appends never entered the log.
    assert len(session.snapshot_events()) == 0


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects a surface-eligible append with no surfaceOp marker (runtime guard "
    "for the union-widening loophole)"
)
def test_rejects_surface_eligible_append_without_marker() -> None:
    session = Session.create(session_id("s5b"))
    session.append("turn/start", {"turn": 1})
    with pytest.raises(ValueError, match="surface-eligible and requires a surfaceOp marker"):
        session.append("user/message", user("hi"))
    # The rejected append never entered the log (only turn/start is present).
    assert len(session.snapshot_events()) == 1


@pytest.mark.upstream(f"{SPEC}::Session > accepts dense arrays and nested plain objects")
def test_accepts_dense_arrays_and_nested_objects() -> None:
    session = Session.create(session_id("s6"))
    session.append(
        "user/message",
        {
            "content": [text("x")],
            "source": {"kind": "user"},
            "extra": [1, 2, [3, {"a": None, "b": True}]],
        },
        surface_op="append",
    )
    assert len(session.snapshot_events()) == 1


@pytest.mark.upstream(
    f"{SPEC}::Session > validates seed events: rejects a non-JSON-serializable seed"
)
def test_rejects_non_json_seed() -> None:
    bad_seed = [
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": {"content": [text("x")], "source": {"kind": "user"}, "bad": BIG},
        }
    ]
    with pytest.raises(ValueError, match="losslessly JSON-serializable"):
        Session.create(session_id("seed-bad"), bad_seed)


@pytest.mark.upstream(f"{SPEC}::Session > validates seed events: rejects a non-contiguous seq")
def test_rejects_non_contiguous_seed() -> None:
    gap_seed = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        # gap: expected seq 1
        {
            "type": "turn/end",
            "seq": 5,
            "time": 2,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    with pytest.raises(ValueError, match="contiguous|seq"):
        Session.create(session_id("seed-gap"), gap_seed)


@pytest.mark.upstream(
    f"{SPEC}::Session > validates seed events: rejects a surface-eligible event missing its "
    "surfaceOp marker"
)
def test_rejects_seed_event_missing_surface_marker() -> None:
    markerless_seed = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "user/message", "seq": 1, "time": 2, "data": user("hi")},
        {
            "type": "turn/end",
            "seq": 2,
            "time": 3,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    with pytest.raises(ValueError, match="requires a surfaceOp marker"):
        Session.create(session_id("seed-no-marker"), markerless_seed)


@pytest.mark.upstream(f"{SPEC}::Session > accepts a well-formed contiguous serializable seed")
def test_accepts_well_formed_seed() -> None:
    good_seed = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {"type": "user/message", "seq": 1, "time": 2, "data": user("hi"), "surfaceOp": "append"},
        {
            "type": "turn/end",
            "seq": 2,
            "time": 3,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    session = Session.create(session_id("seed-ok"), good_seed)
    assert list(session.snapshot_events()[:3]) == good_seed
    assert session.first_live_seq == 3


@pytest.mark.upstream(
    f"{SPEC}::Session > reads each seed array entry once so validation and storage use the same "
    "event"
)
def test_reads_each_seed_entry_once() -> None:
    accepted = {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
    drifted = {**accepted, "seq": 99, "data": {"invalid": BIG}}
    seed = CountingSeed(lambda reads: accepted if reads == 1 else drifted)

    session = Session.create(session_id("seed-entry-snapshot"), seed)

    assert seed.reads == 1
    assert session.snapshot_events()[:1] == (accepted,)


@pytest.mark.upstream(f"{SPEC}::Session > rejects non-JSON surface metadata in a seed event")
def test_rejects_non_json_seed_surface_metadata() -> None:
    seed = [
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": user("hello"),
            "surfaceOp": {"op": "replace", "startSeq": BIG, "endSeq": 2},
        }
    ]
    with pytest.raises(ValueError, match="losslessly JSON-serializable"):
        Session.create(session_id("seed-bad-metadata"), seed)


class _ReplaceOp:
    def __init__(self) -> None:
        self.op = "replace"
        self.startSeq = session_seq(0)
        self.endSeq = session_seq(0)


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects exotic seed metadata before cloning can erase its prototype"
)
def test_rejects_exotic_seed_metadata() -> None:
    seed = [
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": user("hello"),
            "surfaceOp": _ReplaceOp(),
        }
    ]
    with pytest.raises(ValueError, match="losslessly JSON-serializable"):
        Session.create(session_id("seed-exotic-metadata"), seed)


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects an exotic seed event shell before spreading erases its prototype"
)
def test_rejects_exotic_seed_event_shell() -> None:
    class SeedEvent:
        def __init__(self) -> None:
            self.type = "turn/start"
            self.seq = session_seq(0)
            self.time = 1
            self.data = {"turn": 1}

    seed = cast(list[SessionEvent], [SeedEvent()])
    with pytest.raises(ValueError, match="not losslessly JSON-serializable"):
        Session.create(session_id("seed-exotic-shell"), seed)


@pytest.mark.parametrize(
    ("name", "failure", "expected"),
    [("an Error", ValueError("validator failed"), "validator failed")],
)
@pytest.mark.upstream(f"{SPEC}::Session > adds seed context when surface validation throws %s")
def test_adds_seed_context_when_surface_validation_throws(
    monkeypatch: pytest.MonkeyPatch, name: str, failure: Exception, expected: str
) -> None:
    original = surface_module._is_replace_op

    def is_replace_op(op: Any) -> bool:
        if op.get("op") == "replace":
            raise failure
        return original(op)

    monkeypatch.setattr(surface_module, "_is_replace_op", is_replace_op)
    seed = [
        {
            "type": "user/message",
            "seq": 0,
            "time": 1,
            "data": user("source"),
            "surfaceOp": "append",
        },
        {
            "type": "user/message",
            "seq": 1,
            "time": 2,
            "data": user("hello"),
            "surfaceOp": {"op": "replace", "startSeq": 0, "endSeq": 0},
            "sourceEventSeqs": [0],
        },
    ]

    with pytest.raises(ValueError) as raised:
        Session.create(session_id("seed-non-error-metadata-failure"), seed)
    assert f"invalid seed event at index 1: {expected}" in str(raised.value)


@pytest.mark.upstream(
    f"{SPEC}::Session > snapshots the seed: mutating the original after construction does not "
    "affect session.snapshotEvents()"
)
def test_snapshots_the_seed() -> None:
    seed: list[dict[str, Any]] = [
        {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
        {
            "type": "user/message",
            "seq": 1,
            "time": 2,
            "data": {
                "id": MessageId("seed-input"),
                "role": "user",
                "content": [text("original")],
                "source": {"kind": "user"},
            },
            "surfaceOp": "append",
        },
        {
            "type": "turn/end",
            "seq": 2,
            "time": 3,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    session = Session.create(session_id("seed-snapshot"), seed)
    # Mutate the ORIGINAL seed objects after construction: a shared reference
    # would let this rewrite the forked log (or reintroduce non-serializable
    # data past validation). The snapshot must shield session.snapshot_events().
    um = seed[1]
    um["data"]["content"][0]["text"] = "HACKED"
    um["data"]["injected"] = BIG  # would have failed validation
    logged = session.snapshot_events()[1]
    assert logged["type"] == "user/message"
    assert logged["data"]["content"][0]["text"] == "original"
    assert "injected" not in logged["data"]


@pytest.mark.upstream(
    f"{SPEC}::Session > snapshots append data: mutating the passed object after append does not "
    "affect session.snapshotEvents()"
)
def test_snapshots_append_data() -> None:
    session = Session.create(session_id("append-snapshot"))
    data: dict[str, Any] = {
        "id": MessageId("append-input"),
        "role": "user",
        "content": [text("original")],
        "source": {"kind": "user"},
    }
    event = session.append("user/message", data, surface_op="append")
    # Mutate the caller's object after append returns. A shared reference would
    # make session.snapshot_events() diverge from the value that passed validation.
    data["content"][0]["text"] = "HACKED"
    data["injected"] = BIG
    logged = session.snapshot_events()[0]
    assert logged["type"] == "user/message"
    assert logged["data"]["content"][0]["text"] == "original"
    assert "injected" not in logged["data"]
    # The returned event carries the same snapshot, not the caller's input.
    assert event["data"]["content"][0]["text"] == "original"


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects non-JSON surface metadata before appending the event"
)
def test_rejects_non_json_append_surface_metadata() -> None:
    session = Session.create(session_id("append-bad-metadata"))

    with pytest.raises(ValueError, match="non-JSON-serializable surface metadata"):
        session.append(
            "user/message",
            user("hello"),
            surface_op={"op": "replace", "startSeq": BIG, "endSeq": 2},
        )
    assert session.snapshot_events() == ()


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects exotic surface metadata before cloning can erase its prototype"
)
def test_rejects_exotic_append_surface_metadata() -> None:
    session = Session.create(session_id("append-exotic-metadata"))

    with pytest.raises(ValueError, match="non-JSON-serializable surface metadata"):
        session.append("user/message", user("hello"), surface_op=_ReplaceOp())
    assert session.snapshot_events() == ()


@pytest.mark.upstream(f"{SPEC}::Session > rejects invalid plain surface metadata shapes at append")
def test_rejects_invalid_append_surface_shapes() -> None:
    session = Session.create(session_id("append-invalid-surface-shape"))
    data = {"content": [text("hello")], "source": {"kind": "user"}}

    with pytest.raises(ValueError, match="invalid surfaceOp"):
        session.append("user/message", data, surface_op="invalid")
    with pytest.raises(ValueError, match="invalid replace surfaceOp"):
        session.append(
            "user/message", data, surface_op={"op": "replace", "startSeq": -1, "endSeq": 0}
        )
    with pytest.raises(ValueError, match="non-negative safe integers"):
        session.append(
            "user/message",
            data,
            surface_op="append",
            source_event_seqs=[session_seq(0), cast(Any, -1)],
        )
    assert session.snapshot_events() == ()


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects surface metadata on non-surface append and seed events"
)
def test_rejects_surface_metadata_on_non_surface_events() -> None:
    session = Session.create(session_id("non-surface-metadata"))

    with pytest.raises(ValueError, match="not surface-eligible and cannot carry surfaceOp"):
        session.append("turn/start", {"turn": 1}, surface_op="append")
    with pytest.raises(ValueError, match="invalid seed event.*not surface-eligible"):
        Session.create(
            session_id("non-surface-metadata-seed"),
            [
                {
                    "type": "turn/start",
                    "seq": session_seq(0),
                    "time": 1,
                    "data": {"turn": 1},
                    "surfaceOp": "append",
                }
            ],
        )
    assert session.snapshot_events() == ()


@pytest.mark.upstream(f"{SPEC}::Session > deep-freezes seeded and appended event snapshots")
def test_deep_freezes_seeded_and_appended_events() -> None:
    seeded = Session.create(
        session_id("seed-frozen"),
        [{"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}}],
    )
    seeded_event = seeded.snapshot_events()[0]
    assert seeded_event["type"] == "turn/start"
    assert is_frozen(seeded_event)
    assert is_frozen(seeded_event["data"])
    with pytest.raises(TypeError):
        seeded_event["data"]["turn"] = 99

    appended = Session.create(session_id("append-frozen"))
    appended_event = appended.append("user/message", user("first"), surface_op="append")
    assert is_frozen(appended_event)
    assert is_frozen(appended_event["data"])
    assert is_frozen(appended_event["data"]["content"])
    assert is_frozen(appended_event["data"]["content"][0])
    with pytest.raises(TypeError):
        appended_event["data"]["content"][0]["text"] = "mutated"


@pytest.mark.upstream(
    f"{SPEC}::Session > returns cached frozen event-array snapshots that do not grow after append"
)
def test_returns_cached_frozen_event_snapshots() -> None:
    session = Session.create(session_id("events-snapshot"))
    session.append("turn/start", {"turn": 1})
    before = session.snapshot_events()
    before_event = before[0]
    assert before_event["type"] == "turn/start"

    assert session.snapshot_events() is before
    # A tuple is Python's frozen array: it has no mutating methods at all.
    assert isinstance(before, tuple)
    assert not hasattr(before, "append")
    with pytest.raises(TypeError):
        before_event["data"]["turn"] = 99

    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    after = session.snapshot_events()
    assert after is not before
    assert len(before) == 1
    assert len(after) == 2
    assert session.snapshot_events() is after


@pytest.mark.upstream(
    f"{SPEC}::Session > reads one event without materializing an array and snapshots half-open "
    "ranges"
)
def test_reads_one_event_and_snapshots_ranges() -> None:
    session = Session.create(session_id("event-reads"))
    start = session.append("turn/start", {"turn": 1})
    end = session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    assert session.event_at(session_seq(0)) is start
    assert session.event_at(session_seq(1)) is end
    assert session.event_at(session_seq(2)) is None

    events_range = session.snapshot_events(session_log_offset(1), session_log_offset(2))
    assert events_range == (end,)
    assert isinstance(events_range, tuple)

    session.append("turn/start", {"turn": 2})
    assert events_range == (end,)


@pytest.mark.upstream(
    f"{SPEC}::Session > detaches and freezes an explicitly supplied session header"
)
def test_detaches_and_freezes_supplied_header() -> None:
    header_input: dict[str, Any] = {
        "version": SESSION_FORMAT_VERSION,
        "id": session_id("header-owned"),
        "createdAt": 123,
        "cwd": "/accepted",
        "parentSession": session_id("parent"),
        "isSeeded": True,
    }

    session = Session.create(session_id("header-owned"), [], header_input, session_log_offset(0))
    header_input["cwd"] = "/caller-mutated"

    assert session.header == {
        "version": SESSION_FORMAT_VERSION,
        "id": "header-owned",
        "createdAt": 123,
        "cwd": "/accepted",
        "parentSession": "parent",
        "isSeeded": True,
    }
    assert session.header is not header_input
    assert is_frozen(session.header)
    with pytest.raises(TypeError):
        cast(dict[str, Any], session.header)["cwd"] = "/published-mutated"
    assert session.id == "header-owned"
    assert session.header["cwd"] == "/accepted"


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects an exotic, non-JSON, or mismatched supplied header"
)
def test_rejects_exotic_non_json_or_mismatched_header() -> None:
    class ExoticHeader:
        def __init__(self) -> None:
            self.version = SESSION_FORMAT_VERSION
            self.id = session_id("header-invalid")
            self.createdAt = 123
            self.isSeeded = False

    exotic = cast(dict[str, Any], ExoticHeader())
    with pytest.raises(ValueError, match="not losslessly JSON-serializable"):
        Session.create(session_id("header-invalid"), None, exotic)
    with pytest.raises(ValueError, match="not a plain JSON record"):
        Session.from_restore(
            session_id("header-invalid"), [], exotic, session_log_offset(0), "detached"
        )
    invalid_headers: list[object] = [None, 1, []]
    for header in invalid_headers:
        with pytest.raises(ValueError, match="not a plain JSON record"):
            Session.from_restore(
                session_id("header-invalid"),
                [],
                cast(dict[str, Any], header),
                session_log_offset(0),
                "detached",
            )
    with pytest.raises(ValueError, match="not losslessly JSON-serializable"):
        Session.create(
            session_id("header-invalid"),
            None,
            {
                "version": SESSION_FORMAT_VERSION,
                "id": session_id("header-invalid"),
                "createdAt": 123,
                "isSeeded": False,
                "parentSession": BIG,
            },
        )
    with pytest.raises(ValueError, match="does not match session id"):
        Session.create(
            session_id("header-invalid"),
            None,
            {
                "version": SESSION_FORMAT_VERSION,
                "id": session_id("other"),
                "createdAt": 123,
                "isSeeded": False,
            },
        )


@pytest.mark.upstream(
    f"{SPEC}::Session > rejects invalid scalar fields in an explicitly supplied header"
)
def test_rejects_invalid_scalar_header_fields() -> None:
    base = {
        "version": SESSION_FORMAT_VERSION,
        "id": session_id("header-shape"),
        "createdAt": 123,
        "isSeeded": False,
    }
    cases: list[tuple[object, str]] = [
        # Upstream's `null` header case is omitted: Python's None means "no header supplied".
        (1, "not a plain JSON record"),
        ({**base, "version": SESSION_FORMAT_VERSION + 1}, "header version"),
        ({**base, "createdAt": "123"}, "createdAt must be a non-negative safe integer"),
        ({**base, "cwd": 1}, "header cwd must be a string"),
        ({**base, "cwd": "relative"}, "header cwd must be an absolute path"),
        ({**base, "parentSession": 1}, "header parentSession must be a string"),
        ({**base, "isSeeded": "yes"}, "isSeeded must be a boolean"),
        ({**base, "seedLength": 1}, 'invalid field "seedLength"'),
    ]

    for header, error in cases:
        with pytest.raises(ValueError, match=error):
            Session.create(session_id("header-shape"), None, cast(dict[str, Any], header))


@pytest.mark.upstream(
    f"{SPEC}::Session > retains opaque logical header metadata except the physical v0 seed field"
)
def test_retains_opaque_header_metadata() -> None:
    sid = session_id("header-extension")
    session = Session.create(
        sid,
        None,
        {
            "version": SESSION_FORMAT_VERSION,
            "id": sid,
            "createdAt": 123,
            "isSeeded": False,
            "extension": {"value": "kept"},
        },
    )

    assert session.header["extension"] == {"value": "kept"}


@pytest.mark.upstream(f"{SPEC}::Session > rejects seed records with invalid fixed-envelope fields")
def test_rejects_invalid_seed_envelopes() -> None:
    base: dict[str, Any] = {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}}
    cases: list[dict[str, Any]] = [
        {**base, "extra": True},
        {**base, "type": 1},
        {**base, "seq": "0"},
        {**base, "seq": 0.5},
        {**base, "seq": -1},
        {**base, "time": "1"},
        {**base, "time": 0.5},
        {"type": base["type"], "seq": base["seq"], "time": base["time"]},
        {**base, "ignorable": False},
        {**base, "ignorable": "yes"},
    ]

    for index, event in enumerate(cases):
        with pytest.raises(ValueError, match="invalid event envelope"):
            Session.create(session_id(f"bad-envelope-{index}"), [event])

    # `ignorable: true` is the one accepted marker value (unknown-type skip contract).
    marked = Session.create(session_id("ignorable-envelope"), [{**base, "ignorable": True}])
    assert marked.snapshot_events()[0].get("ignorable") is True


# --- SessionStore ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > creates sessions, emits session/created and session/event"
)
async def test_creates_sessions_and_emits_events() -> None:
    ctx, sessions = await mount_store()

    created: list[Session] = []
    events: list[tuple[Session, SessionEvent]] = []
    ctx.on("session/created", created.append)
    ctx.on("session/event", lambda session, event: events.append((session, event)))

    session = sessions.create()
    assert created == [session]

    # The store-owned append publication hooks are module-private. A caller may
    # create an unrelated attribute with the old implementation's name, but cannot
    # suppress the durable event feed.
    session.onAppend = None  # type: ignore[attr-defined]
    session.append("turn/start", {"turn": 1})
    session.append("user/message", user("x"), surface_op="append")
    assert len(events) == 2
    assert events[1][0] is session
    assert events[1][1]["type"] == "user/message"

    assert sessions.get(session.id) is session
    assert sessions.list() == [session]


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::SessionStore > rejects duplicate ids and supports seeding")
async def test_rejects_duplicate_ids_and_supports_seeding() -> None:
    _, sessions = await mount_store()
    a = sessions.create(session_id("fixed"))
    with pytest.raises(ValueError, match="already exists"):
        sessions.create(session_id("fixed"))

    a.append("turn/start", {"turn": 1})
    a.append("user/message", user("q"), surface_op="append")
    forked = sessions.create(session_id("fork"), CreateSessionOptions(seed=a.snapshot_events()))
    assert forked.derive_messages() == a.derive_messages()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > enter() rejects a stale prepared session whose id is already live "
    "(no overwrite)"
)
async def test_enter_rejects_stale_prepared_session() -> None:
    _, sessions = await mount_store()
    stale = sessions.prepare(session_id("racy"))
    live = sessions.create(session_id("racy"))
    with pytest.raises(ValueError, match="already exists"):
        sessions.enter(stale)
    # The live session is intact and still the store entry.
    assert sessions.get(session_id("racy")) is live


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > prepare() + enter() + announce() register a session and emit "
    "session/created"
)
async def test_prepare_enter_announce_lifecycle() -> None:
    ctx, sessions = await mount_store()
    created: list[Session] = []
    ctx.on("session/created", created.append)

    session = sessions.prepare(session_id("lifecycle"))
    # prepare alone does NOT enter the store.
    assert sessions.get(session_id("lifecycle")) is None
    detach = sessions.enter(session)
    assert sessions.get(session_id("lifecycle")) is session
    # enter does NOT announce.
    assert created == []
    sessions.announce(session)
    assert created == [session]
    # The detach disposer removes the entry + stops notification.
    detach()
    detach()  # idempotent: cannot disturb a later same-id lifecycle
    assert sessions.get(session_id("lifecycle")) is None


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > prevents simultaneous attachment of one session object to two stores"
)
async def test_prevents_attachment_to_two_stores() -> None:
    _, first = await mount_store()
    _, second = await mount_store()
    session = Session.create(session_id("owned-key"))
    detach_first = first.enter(session)

    with pytest.raises(ValueError, match="already attached to a store"):
        second.enter(session)
    assert first.get(session_id("owned-key")) is session

    detach_first()
    assert first.get(session_id("owned-key")) is None
    detach_second = second.enter(session)
    assert second.get(session_id("owned-key")) is session
    detach_second()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > rejects direct and reentrant repeat announcements to preserve one "
    "lifecycle pair"
)
async def test_rejects_repeat_announcements() -> None:
    ctx, sessions = await mount_store()
    counts = {"created": 0, "disposed": 0}
    reentrant_error: list[str] = []

    def on_created(session: Session) -> None:
        counts["created"] += 1
        try:
            sessions.announce(session)
        except ValueError as error:
            reentrant_error.append(str(error))

    def on_disposed(session: Session) -> None:
        counts["disposed"] += 1

    ctx.on("session/created", on_created)
    ctx.on("session/disposed", on_disposed)

    session = sessions.prepare(session_id("once"))
    detach = sessions.enter(session)
    sessions.announce(session)
    assert len(reentrant_error) == 1
    assert "already announced" in reentrant_error[0]
    with pytest.raises(ValueError, match="already announced"):
        sessions.announce(session)
    detach()
    assert counts == {"created": 1, "disposed": 1}


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > defers a reentrant detach until the creation dispatch unwinds"
)
async def test_defers_reentrant_detach_during_creation() -> None:
    ctx, sessions = await mount_store()
    order: list[str] = []
    live: list[bool] = []
    session = sessions.prepare(session_id("reentrant-detach"))
    detach = sessions.enter(session)

    def first(created: Session) -> None:
        order.append("created:first")
        detach()
        live.append(sessions.get(created.id) is created)

    def second(created: Session) -> None:
        order.append("created:second")
        live.append(sessions.get(created.id) is created)

    def disposed(gone: Session) -> None:
        order.append("disposed")
        live.append(sessions.get(gone.id) is None)

    ctx.on("session/created", first)
    ctx.on("session/created", second)
    ctx.on("session/disposed", disposed)

    sessions.announce(session)

    assert order == ["created:first", "created:second", "disposed"]
    assert live == [True, True, True]
    assert sessions.get(session.id) is None
    detach()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > rolls back create when its owner unloads from session/created"
)
async def test_rolls_back_create_when_owner_unloads() -> None:
    ctx, sessions = await mount_store()
    owner_ctx: list[Context] = []
    owner = await ctx.inject(["sessions"], lambda inner, _config: owner_ctx.append(inner))
    sid = session_id("create-unload-race")

    def on_created(session: Session) -> None:
        if session.id == sid:
            owner.dispose()

    ctx.on("session/created", on_created)

    store_of(owner_ctx[0]).create(sid, ctx=owner_ctx[0])
    await owner.dispose()
    assert sessions.get(sid) is None


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > synthesizes a minimal current-version header for a bare-created "
    "session"
)
async def test_synthesizes_minimal_header() -> None:
    _, sessions = await mount_store()
    session = sessions.create(session_id("plain"))
    assert session.header["version"] == SESSION_FORMAT_VERSION
    assert session.header["id"] == "plain"
    assert session.header["isSeeded"] is False
    created_at = session.header["createdAt"]
    assert isinstance(created_at, int) and 0 <= created_at <= MAX_SAFE_INTEGER
    assert "cwd" not in session.header
    assert "parentSession" not in session.header


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > attaches cwd and parentSession from meta to the header"
)
async def test_attaches_cwd_and_parent_session() -> None:
    _, sessions = await mount_store()
    session = sessions.create(
        session_id("child"),
        CreateSessionOptions(
            meta=CreateSessionMeta(cwd="/work/project", parent_session=session_id("parent"))
        ),
    )
    assert {
        key: session.header[key] for key in ("version", "id", "cwd", "parentSession", "isSeeded")
    } == {
        "version": SESSION_FORMAT_VERSION,
        "id": "child",
        "cwd": "/work/project",
        "parentSession": "parent",
        "isSeeded": False,
    }


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > attaches subagent origin and delegationDepth from meta to the header"
)
async def test_attaches_origin_and_delegation_depth() -> None:
    _, sessions = await mount_store()
    session = sessions.create(
        session_id("delegated-child"),
        CreateSessionOptions(
            meta=CreateSessionMeta(
                parent_session=session_id("parent"), origin="subagent", delegation_depth=2
            )
        ),
    )
    assert {
        key: session.header[key] for key in ("id", "parentSession", "origin", "delegationDepth")
    } == {
        "id": "delegated-child",
        "parentSession": "parent",
        "origin": "subagent",
        "delegationDepth": 2,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("meta", "error"),
    [
        ({"parent_session": BIG}, "header is not losslessly JSON-serializable"),
        ({"cwd": 1}, "header cwd must be a string"),
        ({"parent_session": 1}, "header parentSession must be a string"),
        ({"created_at": "123"}, "header createdAt must be a non-negative safe integer"),
        ({"created_at": 1.5}, "header createdAt must be a non-negative safe integer"),
        ({"created_at": -1}, "header createdAt must be a non-negative safe integer"),
        # Number.MAX_SAFE_INTEGER + 1 is a double in JavaScript.
        (
            {"created_at": float(MAX_SAFE_INTEGER + 1)},
            "header createdAt must be a non-negative safe integer",
        ),
        pytest.param(
            {"is_seeded": "yes"},
            "isSeeded must be a boolean",
        ),
        ({"origin": "fork"}, 'origin must be "subagent"'),
        ({"delegation_depth": "1"}, "delegationDepth must be a non-negative safe integer"),
        ({"delegation_depth": 0.5}, "delegationDepth must be a non-negative safe integer"),
        ({"delegation_depth": -1}, "delegationDepth must be a non-negative safe integer"),
        ({"agent_preset": 1}, "agentPreset must be a string"),
    ],
)
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > rejects non-JSON and invalid scalar session metadata"
)
async def test_rejects_invalid_session_metadata(meta: dict[str, Any], error: str) -> None:
    _, sessions = await mount_store()
    with pytest.raises(ValueError, match=error):
        sessions.prepare(
            session_id(f"bad-meta-{next(iter(meta))}"),
            CreateSessionOptions(meta=CreateSessionMeta(**meta)),
        )


@pytest.mark.asyncio
@pytest.mark.upstream(f"{SPEC}::SessionStore > rejects a non-absolute meta.cwd")
async def test_rejects_relative_meta_cwd() -> None:
    _, sessions = await mount_store()
    with pytest.raises(ValueError, match="cwd must be an absolute path"):
        sessions.create(
            session_id("rel"), CreateSessionOptions(meta=CreateSessionMeta(cwd="relative/path"))
        )
    # the rejected session was not registered
    assert sessions.get(session_id("rel")) is None


@pytest.mark.upstream(
    f"{SPEC}::SessionStore > a bare Session() constructed without the store still exposes a "
    "current-version header"
)
def test_bare_session_exposes_current_header() -> None:
    session = Session.create(session_id("bare"))
    assert session.header["version"] == SESSION_FORMAT_VERSION
    assert session.header["id"] == "bare"
    assert session.header["isSeeded"] is False
    assert isinstance(session.header["createdAt"], int)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > detaches sessions when the creating fiber is disposed (HMR safety)"
)
async def test_detaches_sessions_when_fiber_disposed() -> None:
    ctx, sessions = await mount_store()

    scoped: list[Session] = []
    fiber = await ctx.inject(
        ["sessions"],
        lambda inner, _config: scoped.append(store_of(inner).create(session_id("scoped"))),
    )
    session = scoped[0]
    assert sessions.get(session_id("scoped")) is session

    observed: list[object] = []
    ctx.on("session/event", lambda *args: observed.append(args))

    await fiber.dispose()
    assert sessions.get(session_id("scoped")) is None
    with pytest.raises(ValueError, match="cannot append outside a SessionStore"):
        session.append("user/message", user("late"), surface_op="append")
    assert observed == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > pairs a partial session/created announcement with disposal during "
    "rollback"
)
async def test_pairs_partial_announcement_with_disposal() -> None:
    ctx, sessions = await mount_store()

    threw: list[bool] = []
    disposed: list[Session] = []
    ctx.on("session/disposed", disposed.append)

    def on_created(session: Session) -> None:
        if not threw:
            threw.append(True)
            raise ValueError("boom created listener")

    ctx.on("session/created", on_created)

    # The throwing emit must roll the store entry back, not leak it.
    with pytest.raises(ValueError, match="boom created listener"):
        sessions.create(session_id("fixed"))
    assert sessions.get(session_id("fixed")) is None  # rolled back, not leaked
    assert [session.id for session in disposed] == ["fixed"]

    # A subsequent create of the SAME id succeeds (the already-exists check is
    # not wedged) and its store-owned publication hooks are correctly wired.
    events: list[SessionEvent] = []
    ctx.on("session/event", lambda _session, event: events.append(event))
    session = sessions.create(session_id("fixed"))
    assert sessions.get(session_id("fixed")) is session
    session.append("turn/start", {"turn": 1})
    session.append("user/message", user("hi"), surface_op="append")
    assert events[-1]["type"] == "user/message"


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > contains session/event observer failures after the append commit point"
)
async def test_contains_session_event_observer_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    ctx, sessions = await mount_store()
    session = sessions.create(session_id("contained-event"))
    heard: list[SessionEvent] = []
    committed_before_notify: list[bool] = []

    def sync_observer(observed_session: Session, event: SessionEvent) -> None:
        committed_before_notify.append(observed_session.snapshot_events()[-1] is event)
        raise ValueError("sync event observer")

    async def async_observer(_session: Session, _event: SessionEvent) -> None:
        raise ValueError("async event observer")

    ctx.on("session/event", sync_observer)
    ctx.on("session/event", async_observer)
    ctx.on("session/event", lambda _session, event: heard.append(event))

    appended = session.append("turn/start", {"turn": 1})
    assert committed_before_notify == [True]
    assert session.snapshot_events() == (appended,)
    assert heard == [appended]
    await settle()

    assert warnings_of(caplog) == [
        'session "contained-event": session/event listener threw: sync event observer',
        'session "contained-event": session/event listener rejected: async event observer',
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > runs internal dispatch validation on one frozen candidate before "
    "commit and resets after a veto"
)
async def test_internal_dispatch_validates_frozen_candidate() -> None:
    ctx, sessions = await mount_store()
    session = sessions.create(session_id("dispatch-veto"))
    validations: list[tuple[SessionEvent, int, bool]] = []
    observed: list[SessionEvent] = []
    reject = [True]

    def instrument(_mode: str, name: str, args: tuple[object, ...], _filter: object) -> None:
        if name != "session/event":
            return
        observed_session, event = cast(tuple[Session, SessionEvent], args)
        validations.append(
            (
                event,
                len(observed_session.snapshot_events()),
                is_frozen(event) and is_frozen(event["data"]),
            )
        )
        if reject[0]:
            reject[0] = False
            raise ValueError("reject first candidate")

    ctx.on("internal/dispatch", instrument)
    ctx.on("session/event", lambda _session, event: observed.append(event))

    with pytest.raises(ValueError, match="reject first candidate"):
        session.append("turn/start", {"turn": 1})
    assert session.snapshot_events() == ()
    assert observed == []

    appended = session.append("turn/start", {"turn": 1})
    assert [(length, frozen) for _, length, frozen in validations] == [(0, True), (0, True)]
    assert [event["seq"] for event, _, _ in validations] == [0, 0]
    assert validations[1][0] is appended
    assert session.snapshot_events() == (appended,)
    assert observed == [appended]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > does not publish a surface transition rejected by internal dispatch"
)
async def test_rejected_surface_transition_not_published() -> None:
    ctx, sessions = await mount_store()
    session = sessions.create(session_id("surface-dispatch-veto"))
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("user/message", user("source"), surface_op="append")
    surface = session.surface
    reject = [True]

    def instrument(_mode: str, name: str, _args: object, _filter: object) -> None:
        if name == "session/event" and reject[0]:
            reject[0] = False
            raise ValueError("reject surface candidate")

    ctx.on("internal/dispatch", instrument)

    with pytest.raises(ValueError, match="reject surface candidate"):
        session.append(
            "user/message",
            user("replacement", "test"),
            surface_op={"op": "replace", "startSeq": session_seq(2), "endSeq": session_seq(2)},
            source_event_seqs=[session_seq(2)],
        )

    assert len(session.snapshot_events()) == 3
    assert list(surface.nodes) == [2]
    assert surface.replace_generation == 0

    session.append("user/message", user("next"), surface_op="append")
    assert list(surface.nodes) == [2, 3]
    assert surface.replace_generation == 0


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > resolves session/event dispatch before commit so instrumentation "
    "failure cannot hide a logged event"
)
async def test_dispatch_resolution_failure_blocks_commit() -> None:
    ctx, sessions = await mount_store()
    session = sessions.create(session_id("dispatch-check"))
    observed: list[SessionEvent] = []

    def instrument(_mode: str, name: str, _args: object, _filter: object) -> None:
        if name == "session/event":
            raise ValueError("dispatch instrumentation rejected the carrier")

    ctx.on("internal/dispatch", instrument)
    ctx.on("session/event", lambda _session, event: observed.append(event))

    with pytest.raises(ValueError, match="dispatch instrumentation rejected the carrier"):
        session.append("turn/start", {"turn": 1})
    assert session.snapshot_events() == ()
    assert observed == []


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > contains a reentrant observer append without reordering later "
    "observers"
)
async def test_contains_reentrant_observer_append(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    ctx, sessions = await mount_store()
    session = sessions.create(session_id("reentrant-observer"))
    heard: list[SessionEvent] = []

    def reenter(observed_session: Session, _event: SessionEvent) -> None:
        observed_session.append("request/context", {"provider": "mock", "model": "mock"})

    ctx.on("session/event", reenter)
    ctx.on("session/event", lambda _session, event: heard.append(event))

    appended = session.append("turn/start", {"turn": 1})
    assert session.snapshot_events() == (appended,)
    assert heard == [appended]
    assert warnings_of(caplog) == [
        'session "reentrant-observer": session/event listener threw: session append cannot '
        "reenter while another append is being published",
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > defers detach through dispatch resolution, commit, and observer "
    "publication"
)
async def test_defers_detach_through_append_publication() -> None:
    ctx, sessions = await mount_store()
    order: list[str] = []
    session = sessions.prepare(session_id("detach-during-append"))
    detach = sessions.enter(session)

    def state(target: Session) -> str:
        return "live" if sessions.get(target.id) is target else "detached"

    def instrument(_mode: str, name: str, args: tuple[object, ...], _filter: object) -> None:
        if name != "session/event":
            return
        order.append(f"resolve:{state(cast(Session, args[0]))}")
        detach()

    ctx.on("internal/dispatch", instrument)
    ctx.on("session/event", lambda target, _event: order.append(f"observe:{state(target)}"))
    ctx.on("session/disposed", lambda target: order.append(f"dispose:{state(target)}"))
    sessions.announce(session)

    appended = session.append("turn/start", {"turn": 1})

    assert session.snapshot_events() == (appended,)
    assert order == ["resolve:live", "observe:live", "dispose:detached"]
    assert sessions.get(session.id) is None


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > observes async session/created rejection without rolling back or "
    "starving peers"
)
async def test_observes_async_created_rejection(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    ctx, sessions = await mount_store()
    heard: list[str] = []

    async def reject(_session: Session) -> None:
        raise ValueError("late creation failure")

    ctx.on("session/created", reject)
    ctx.on("session/created", lambda session: heard.append(session.id))

    session = sessions.create(session_id("async-created"))
    await settle()

    assert sessions.get(session.id) is session
    assert heard == ["async-created"]
    assert warnings_of(caplog) == [
        'session "async-created": session/created listener rejected: late creation failure',
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > contains synchronous and async session/disposed listener failures "
    "per observer"
)
async def test_contains_disposed_listener_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    ctx, sessions = await mount_store()
    heard: list[str] = []

    def sync_failure(_session: Session) -> None:
        raise ValueError("sync disposed")

    async def async_failure(_session: Session) -> None:
        raise ValueError("async disposed")

    ctx.on("session/disposed", sync_failure)
    ctx.on("session/disposed", async_failure)
    ctx.on("session/disposed", lambda session: heard.append(session.id))

    unannounced = sessions.prepare(session_id("never-announced"))
    detach_unannounced = sessions.enter(unannounced)
    detach_unannounced()
    assert heard == []
    await settle()
    assert warnings_of(caplog) == []

    announced = sessions.prepare(session_id("contained-disposal"))
    detach = sessions.enter(announced)
    sessions.announce(announced)
    detach()
    await settle()

    assert heard == ["contained-disposal"]
    assert warnings_of(caplog) == [
        'session "contained-disposal": session/disposed listener threw: sync disposed',
        'session "contained-disposal": session/disposed listener rejected: async disposed',
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > contains internal dispatch failure after session detachment"
)
async def test_contains_internal_dispatch_failure_after_detach(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    ctx, sessions = await mount_store()
    heard: list[Session] = []

    def instrument(_mode: str, name: str, _args: object, _filter: object) -> None:
        if name == "session/disposed":
            raise ValueError("disposed dispatch instrumentation")

    ctx.on("internal/dispatch", instrument)
    ctx.on("session/disposed", heard.append)
    session = sessions.prepare(session_id("disposed-dispatch"))
    detach = sessions.enter(session)
    sessions.announce(session)

    detach()
    assert sessions.get(session.id) is None
    assert heard == []
    assert warnings_of(caplog) == [
        'session "disposed-dispatch": session/disposed dispatch threw: disposed dispatch '
        "instrumentation",
    ]


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{SPEC}::SessionStore > does not let internal dispatch replace the disposed callback tuple"
)
async def test_internal_dispatch_cannot_replace_disposed_tuple() -> None:
    ctx, sessions = await mount_store()
    replacement = Session.create(session_id("replacement-disposed"))
    heard: list[Session] = []
    attempted: list[bool] = []

    def instrument(_mode: str, name: str, args: object, _filter: object) -> None:
        if name != "session/disposed":
            return
        # The arguments are an immutable tuple; the replacement attempt is refused.
        try:
            cast(list[object], args)[0] = replacement
        except TypeError:
            attempted.append(True)

    ctx.on("internal/dispatch", instrument)
    ctx.on("session/disposed", heard.append)
    session = sessions.prepare(session_id("fixed-disposed-tuple"))
    detach = sessions.enter(session)
    sessions.announce(session)

    detach()

    assert attempted == [True]
    assert heard == [session]
