"""Ported from packages/session/session-format-v3-to-v4/tests/relationships.spec.ts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from deskagents.core.session import Session, SessionEvent, SessionHeader
from deskagents.llm import Message
from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEvent,
    SessionFormatRestoreOptions,
)
from deskagents.session.format_catalog import session_format_catalog
from deskagents.util.values import parse_json, stringify_json

SPEC = "packages/session/session-format-v3-to-v4/tests/relationships.spec.ts"
LIFECYCLE = "mandatory V4 lifecycle restoration"
COMPACTION = "mandatory V4 compaction restoration"
DEPENDENT = "mandatory V4 dependent event restoration"
SYSTEM = "V4 protected system surface and malformed relationship inputs"

Row = dict[str, Any]
STEP: dict[str, Any] = {"turn": 1, "step": 1}
CALL: dict[str, Any] = {"type": "tool-call", "id": "call", "name": "read", "arguments": "{}"}


def row(type_: str, data: dict[str, Any]) -> Row:
    return {"type": type_, "data": data}


def user(kind: str = "user") -> dict[str, Any]:
    return {
        "id": "input",
        "role": "user",
        "source": {"kind": kind},
        "content": [{"type": "text", "text": "input"}],
    }


def input_(kind: str = "user") -> Row:
    return {**row("user/message", user(kind)), "surfaceOp": "append"}


def assistant(content: list[Any] | None = None) -> Row:
    message = {
        "id": "assistant",
        "role": "assistant",
        "source": {"kind": "model", "provider": "mock", "model": "mock"},
        "content": [CALL] if content is None else content,
    }
    return {
        **row("assistant/message", {**STEP, "stream": [], "message": message}),
        "surfaceOp": "append",
    }


def result(id_: str = "call", extra: dict[str, Any] | None = None) -> Row:
    message = {
        "id": "result",
        "role": "tool",
        "source": {"kind": "tool", "callId": id_},
        "toolCallId": id_,
        "content": [{"type": "text", "text": "done"}],
        **(extra or {}),
    }
    return {**row("tool/result", {**STEP, "message": message}), "surfaceOp": "append"}


def begin() -> list[Row]:
    return [row("turn/start", {"turn": 1}), row("step/start", STEP)]


def end() -> list[Row]:
    return [row("step/end", STEP), row("turn/end", {"turn": 1, "reason": {"kind": "completed"}})]


def tool_call() -> Row:
    return row("tool/call", {**STEP, "callId": "call", "name": "read", "arguments": "{}"})


def request() -> Row:
    return row(
        "request/header",
        {"reason": "initial", "header": {"config": {"provider": "mock", "model": "mock"}}},
    )


def reopen(
    rows: Sequence[Row], inherited: int = 0, seeded: bool = False
) -> tuple[SessionFormatArtifact, list[Message]]:
    """Exercise physical JSON, catalog relationship admission, Session adoption and derivation."""
    header: dict[str, Any] = {
        "type": "session",
        "version": 4,
        "id": "native-relations",
        "createdAt": 1,
        "delegationDepth": 0,
        "isSeeded": seeded,
        **({"parentSession": "parent"} if seeded else {}),
    }
    restore = session_format_catalog.create_restore(
        header, SessionFormatRestoreOptions(recovery="strict", validation="current")
    )
    for seq, candidate in enumerate(rows):
        event = cast(SessionFormatEvent, {**candidate, "seq": seq, "time": seq + 1})
        restore.decode_row(
            parse_json(stringify_json(session_format_catalog.encode_current_event(event)))
        )
    artifact = restore.finish()
    assert artifact.inherited_event_count == inherited
    session = Session.from_restore(
        cast(str, artifact.header["id"]),
        cast(Sequence[SessionEvent], artifact.events),
        cast(SessionHeader, artifact.header),
        inherited,
        "detached",
    )
    return artifact, session.derive_messages()


def compact() -> list[Row]:
    return [
        row("turn/start", {"turn": 1}),
        input_(),
        row("compaction/start", {"compactionId": "c", "turn": 1, "sourceCommandId": "cmd"}),
        row(
            "compaction/summary",
            {
                "compactionId": "c",
                "sourceCommandId": "cmd",
                "summary": [{"type": "text", "text": "summary"}],
                "shadowedRange": {"start": 1, "end": 1},
                "shadowedSeqs": [1],
                "shadowedTokenCount": 1,
                "provider": "mock",
                "model": "mock",
            },
        ),
        {
            **input_("compact-checkpoint"),
            "data": {
                **user("compact-checkpoint"),
                "id": "checkpoint",
                "source": {
                    "kind": "compact-checkpoint",
                    "compactionId": "c",
                    "sourceCommandId": "cmd",
                },
            },
            "surfaceOp": {"op": "replace", "startSeq": 1, "endSeq": 1},
            "sourceEventSeqs": [1, 2, 3],
        },
        row("compaction/end", {"compactionId": "c", "sourceCommandId": "cmd", "turn": 1}),
        row("turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    ]


def change(rows: list[Row], index: int, data: dict[str, Any]) -> list[Row]:
    return [
        {**candidate, "data": {**candidate["data"], **data}} if i == index else candidate
        for i, candidate in enumerate(rows)
    ]


def retry(**changes: Any) -> Row:
    return row(
        "llm/retry",
        {
            **STEP,
            "retryId": "retry",
            "retry": 1,
            "provider": "mock",
            "policyKey": "policy",
            **changes,
        },
    )


def retry_start(**changes: Any) -> Row:
    return row("llm/retry-started", {**STEP, "retryId": "retry", "retry": 1, **changes})


def dispatch(type_: str, **changes: Any) -> Row:
    return row(
        "tool/ptc-dispatch-start" if type_ == "start" else "tool/ptc-dispatch",
        {
            "rootCallId": "root",
            "parentCallId": "root",
            "subCallId": "child",
            "name": "read",
            "arguments": {"a": 1, "b": 2},
            **changes,
        },
    )


@pytest.mark.upstream(
    f"{SPEC}::{LIFECYCLE} > retains valid ordinary tool exchanges and open interrupted tails"
)
def test_retains_tool_exchanges_and_open_tails() -> None:
    rows = [*begin(), input_(), request(), assistant(), tool_call(), result(), *end()]
    assert [message["role"] for message in reopen(rows)[1]] == ["user", "assistant", "tool"]
    for length in range(len(rows)):
        reopen(rows[:length])


LIFECYCLE_REJECTIONS: list[tuple[str, list[Row]]] = [
    ("turn order", [row("turn/start", {"turn": 2})]),
    ("duplicate turn", [row("turn/start", {"turn": 1}), row("turn/start", {"turn": 1})]),
    ("unopened turn end", [row("turn/end", {"turn": 1})]),
    ("open step at turn end", [*begin(), row("turn/end", {"turn": 1})]),
    ("unopened step", [row("step/start", STEP)]),
    ("step sequence", [row("turn/start", {"turn": 1}), row("step/start", {"turn": 1, "step": 2})]),
    ("mismatched step end", [*begin(), row("step/end", {"turn": 2, "step": 1})]),
    ("assistant outside step", [assistant([])]),
    ("assistant attempt outside step", [row("assistant/attempt", {**STEP, "stream": []})]),
    ("tool without advertisement", [*begin(), tool_call()]),
    ("result without advertisement", [*begin(), result()]),
    ("repeated advertisement", [*begin(), assistant([CALL, CALL])]),
    (
        "changed tool name",
        [
            *begin(),
            assistant(),
            row("tool/call", {**STEP, "callId": "call", "name": "other", "arguments": "{}"}),
        ],
    ),
    (
        "changed tool arguments",
        [
            *begin(),
            assistant(),
            row("tool/call", {**STEP, "callId": "call", "name": "read", "arguments": "[]"}),
        ],
    ),
    ("repeated start", [*begin(), assistant(), tool_call(), tool_call()]),
    ("unstarted success", [*begin(), assistant(), result()]),
    ("repeated result", [*begin(), assistant(), tool_call(), result(), result()]),
    ("unresolved advertised call", [*begin(), assistant(), *end()]),
    ("unresolved started call", [*begin(), assistant(), tool_call(), *end()]),
    ("header outside turn", [request()]),
    ("context outside turn", [row("request/context", {"provider": "mock", "model": "mock"})]),
]


@pytest.mark.upstream(f"{SPEC}::{LIFECYCLE} > rejects %s")
@pytest.mark.parametrize(
    ("name", "rows"), LIFECYCLE_REJECTIONS, ids=[name for name, _ in LIFECYCLE_REJECTIONS]
)
def test_rejects_lifecycle(name: str, rows: list[Row]) -> None:
    with pytest.raises(SessionFormatError):
        reopen(rows)


@pytest.mark.upstream(
    f"{SPEC}::{LIFECYCLE} > accepts exact not-started repairs with preserved historical "
    "identity suffixes"
)
def test_accepts_exact_not_started_repairs() -> None:
    repaired = result(
        "call",
        {
            "id": "interrupted-tool-result-call-200",
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": "The tool call was interrupted before the Harness recorded it as "
                    "started. Retry it if it is still needed.",
                }
            ],
        },
    )
    repaired["data"]["error"] = {"name": "ToolNotStartedError", "code": "TOOL_NOT_STARTED"}
    reopen([*begin(), assistant(), repaired, *end()])
    extras: list[dict[str, Any]] = [
        {"id": "made-up"},
        {"id": "interrupted-tool-result-call-01"},
        {"id": "interrupted-tool-result-call-9007199254740992"},
        {"content": []},
        {"content": [{"type": "text", "text": "invented repair"}]},
    ]
    for extra in extras:
        invalid = {
            **repaired,
            "data": {**repaired["data"], "message": {**repaired["data"]["message"], **extra}},
        }
        with pytest.raises(SessionFormatError):
            reopen([*begin(), assistant(), invalid])
    with pytest.raises(SessionFormatError):
        reopen([*begin(), assistant(), {**repaired, "sourceEventSeqs": [2]}])
    with pytest.raises(SessionFormatError):
        reopen(
            [
                *begin(),
                assistant(),
                {
                    **repaired,
                    "data": {
                        **repaired["data"],
                        "error": {"name": "OtherError", "code": "TOOL_NOT_STARTED"},
                    },
                },
            ]
        )


@pytest.mark.upstream(
    f"{SPEC}::{LIFECYCLE} > preserves ignorable payloads and unknown attribution metadata"
)
def test_preserves_ignorable_payloads_and_attribution_metadata() -> None:
    message = {
        **user("external"),
        "source": parse_json('{"kind":"external","__proto__":{"x":1},"meta":{"a":2}}'),
    }
    opaque: Row = {"type": "external/event", "data": {"arbitrary": True}, "ignorable": True}
    rows = [opaque, {**input_(), "data": message}]
    artifact, messages = reopen(rows)
    assert artifact.events[0]["data"] == opaque["data"]
    assert messages[0]["source"] == message["source"]


@pytest.mark.upstream(
    f"{SPEC}::{COMPACTION} > retains matching checkpoint sources and inherited unfinished "
    "compactions"
)
def test_retains_checkpoint_sources_and_inherited_compactions() -> None:
    assert reopen(compact())[1][0]["source"] == {
        "kind": "compact-checkpoint",
        "compactionId": "c",
        "sourceCommandId": "cmd",
    }
    inherited = [
        row("compaction/start", {"compactionId": "old", "turn": None}),
        *begin(),
        *end(),
        row("session/end-seed", {"inherited": True}),
    ]
    reopen(inherited, 5, True)
    reopen(
        [
            row("compaction/start", {"compactionId": "c", "turn": None}),
            row("compaction/end", {"compactionId": "c", "turn": None, "error": "cancelled"}),
        ]
    )
    reopen(compact()[:5])


COMPACTION_REJECTIONS: list[tuple[str, list[Row]]] = [
    (
        "checkpoint identity",
        change(
            compact(),
            4,
            {
                "source": {
                    "kind": "compact-checkpoint",
                    "compactionId": "WRONG",
                    "sourceCommandId": "cmd",
                }
            },
        ),
    ),
    (
        "checkpoint command",
        change(
            compact(),
            4,
            {
                "source": {
                    "kind": "compact-checkpoint",
                    "compactionId": "c",
                    "sourceCommandId": "WRONG",
                }
            },
        ),
    ),
    ("summary identity", change(compact(), 3, {"compactionId": "WRONG"})),
    ("end identity", change(compact(), 5, {"compactionId": "WRONG"})),
    ("end turn", change(compact(), 5, {"turn": None})),
    ("start turn", change(compact(), 2, {"turn": None})),
    ("wrong span", change(compact(), 3, {"shadowedSeqs": []})),
    (
        "missing span",
        change(compact(), 3, {"shadowedRange": {"start": 0, "end": 0}, "shadowedSeqs": [0]}),
    ),
    (
        "overlap",
        [*compact()[:3], row("compaction/start", {"compactionId": "second", "turn": 1})],
    ),
    ("repeated summary", [*compact()[:4], compact()[3]]),
    ("success without summary", [*compact()[:3], compact()[5]]),
    ("turn closes transaction", [*compact()[:3], compact()[6]]),
    ("unowned end", [row("compaction/end", {"compactionId": "c", "turn": None})]),
]


@pytest.mark.upstream(f"{SPEC}::{COMPACTION} > rejects %s")
@pytest.mark.parametrize(
    ("name", "rows"), COMPACTION_REJECTIONS, ids=[name for name, _ in COMPACTION_REJECTIONS]
)
def test_rejects_compaction(name: str, rows: list[Row]) -> None:
    with pytest.raises(SessionFormatError):
        reopen(rows)


@pytest.mark.upstream(
    f"{SPEC}::{DEPENDENT} > accepts PTC nesting and settlements with equal argument objects"
)
def test_accepts_ptc_nesting_and_equal_arguments() -> None:
    reopen(
        [
            *begin(),
            dispatch("start"),
            dispatch("start", subCallId="grandchild", parentCallId="child"),
            dispatch("end", subCallId="grandchild", parentCallId="child"),
            dispatch("end", arguments={"b": 2, "a": 1}),
            *end(),
        ]
    )


PTC_REJECTIONS: list[tuple[str, list[Row]]] = [
    ("outside turn", [dispatch("start")]),
    ("without start", [*begin(), dispatch("end")]),
    ("duplicate start", [*begin(), dispatch("start"), dispatch("start")]),
    ("duplicate settlement", [*begin(), dispatch("start"), dispatch("end"), dispatch("end")]),
    ("unknown parent", [*begin(), dispatch("start", parentCallId="unknown")]),
    ("root changes", [*begin(), dispatch("start"), dispatch("end", rootCallId="different")]),
    ("arguments change", [*begin(), dispatch("start"), dispatch("end", arguments={})]),
    ("name changes", [*begin(), dispatch("start"), dispatch("end", name="different")]),
]


@pytest.mark.upstream(f"{SPEC}::{DEPENDENT} > rejects PTC %s")
@pytest.mark.parametrize(("name", "rows"), PTC_REJECTIONS, ids=[name for name, _ in PTC_REJECTIONS])
def test_rejects_ptc(name: str, rows: list[Row]) -> None:
    with pytest.raises(SessionFormatError):
        reopen(rows)


@pytest.mark.upstream(
    f"{SPEC}::{DEPENDENT} > accepts retry chains, including retries scheduled after the step ends"
)
def test_accepts_retry_chains() -> None:
    reopen(
        [
            *begin(),
            request(),
            retry(),
            retry_start(),
            retry(retry=2),
            retry_start(retry=2),
            row("step/end", STEP),
            retry(retry=3),
            row("turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        ]
    )


RETRY_REJECTIONS: list[tuple[str, list[Row]]] = [
    ("outside turn", [retry()]),
    ("wrong step", [*begin(), request(), retry(step=2)]),
    ("wrong provider", [*begin(), request(), retry(provider="other")]),
    ("skip attempt", [*begin(), request(), retry(retry=2)]),
    ("changed identity", [*begin(), request(), retry(), retry(retry=2, retryId="other")]),
    ("reused identity", [*begin(), request(), retry(), retry(policyKey="other")]),
    ("unpaired start", [retry_start()]),
    ("changed start coordinates", [*begin(), request(), retry(), retry_start(step=2)]),
    ("duplicate start", [*begin(), request(), retry(), retry_start(), retry_start()]),
]


@pytest.mark.upstream(f"{SPEC}::{DEPENDENT} > rejects retry %s")
@pytest.mark.parametrize(
    ("name", "rows"), RETRY_REJECTIONS, ids=[name for name, _ in RETRY_REJECTIONS]
)
def test_rejects_retry(name: str, rows: list[Row]) -> None:
    with pytest.raises(SessionFormatError):
        reopen(rows)


@pytest.mark.upstream(
    f"{SPEC}::{DEPENDENT} > requires command completion to cite its run and an earlier "
    "non-command success event"
)
def test_requires_command_completion_citations() -> None:
    run = row("command/run", {"commandId": "cmd", "name": "test", "source": {"kind": "user"}})
    done = row("command/done", {"commandId": "cmd", "kind": "success", "sourceEventSeq": 1})
    reopen([run, input_(), done, row("command/done", {"commandId": "cmd", "kind": "error"})])
    for rows in [
        [done],
        [run, run],
        [run, done],
        [run, input_(), {**done, "data": {**done["data"], "kind": "error"}}],
        [run, input_(), {**done, "data": {**done["data"], "sourceEventSeq": 10}}],
    ]:
        with pytest.raises(SessionFormatError):
            reopen(rows)


@pytest.mark.upstream(
    f"{SPEC}::{DEPENDENT} > requires titles and durable title requests to cite earlier human "
    "messages"
)
def test_requires_title_citations() -> None:
    title = row(
        "session/title",
        {"title": "Example", "source": {"kind": "generated"}, "messageSeqs": [0]},
    )
    request_title = row(
        "session/title-llm-request",
        {
            "messageSeqs": [0],
            "messages": [
                {
                    **user("dsh-session-title-llm"),
                    "content": [
                        {"type": "text", "text": "historical frame with original coordinates"}
                    ],
                }
            ],
        },
    )
    reopen(
        [
            input_(),
            title,
            request_title,
            row(
                "session/title", {"title": "Manual", "source": {"kind": "user"}, "messageSeqs": []}
            ),
        ]
    )
    for rows in [
        [input_("external"), title],
        [input_(), {**title, "data": {**title["data"], "source": {"kind": "user"}}}],
        [input_(), {**title, "data": {**title["data"], "messageSeqs": [0, 0]}}],
        [input_(), {**title, "data": {**title["data"], "messageSeqs": [1]}}],
        [input_(), {**request_title, "data": {**request_title["data"], "messages": [user()]}}],
        [input_(), {**request_title, "data": {**request_title["data"], "messageSeqs": []}}],
    ]:
        with pytest.raises(SessionFormatError):
            reopen(rows)


def system() -> Row:
    message = {
        "id": "system",
        "role": "system",
        "source": {"kind": "system-prompt"},
        "content": [{"type": "text", "text": "system"}],
    }
    return {**row("system/message", {**STEP, "message": message}), "surfaceOp": "append"}


@pytest.mark.upstream(
    f"{SPEC}::{SYSTEM} > keeps the first system head protected across exact replacements"
)
def test_keeps_first_system_head_protected() -> None:
    replacement = {
        **system(),
        "surfaceOp": {"op": "replace", "startSeq": 2, "endSeq": 2},
        "sourceEventSeqs": [2],
    }
    reopen([*begin(), system(), input_(), replacement, system(), *end()])
    prune = row(
        "compaction/prune",
        {"shadowedRange": {"start": 3, "end": 3}, "shadowedSeqs": [3], "shadowedTokenCount": 1},
    )
    reopen([*begin(), system(), input_(), prune, *end()])
    for rows in [
        [*begin(), input_(), system()],
        [*begin(), input_(), {**replacement, "sourceEventSeqs": [2]}],
        [
            *begin(),
            system(),
            {
                **input_(),
                "surfaceOp": {"op": "replace", "startSeq": 2, "endSeq": 2},
                "sourceEventSeqs": [2],
            },
        ],
        [
            *begin(),
            system(),
            input_(),
            {
                **replacement,
                "surfaceOp": {"op": "replace", "startSeq": 2, "endSeq": 3},
                "sourceEventSeqs": [2, 3],
            },
        ],
        [
            *begin(),
            system(),
            input_(),
            {**replacement, "surfaceOp": {"op": "replace", "startSeq": 0, "endSeq": 0}},
        ],
        [
            *begin(),
            system(),
            input_(),
            {**replacement, "surfaceOp": {"op": "replace", "startSeq": 3, "endSeq": 2}},
        ],
        [*begin(), system(), input_(), {**replacement, "sourceEventSeqs": [0]}],
        [
            *begin(),
            system(),
            row("compaction/prune", {"shadowedRange": {"start": 2, "end": 2}, "shadowedSeqs": [2]}),
        ],
    ]:
        with pytest.raises(SessionFormatError):
            reopen(rows)


@pytest.mark.upstream(
    f"{SPEC}::{SYSTEM} > rejects malformed relationship values and mismatched inherited "
    "transaction ownership"
)
def test_rejects_malformed_relationship_values() -> None:
    with pytest.raises(SessionFormatError):
        reopen([row("command/run", {"commandId": ""})])
    with pytest.raises(SessionFormatError):
        reopen([row("session/title", {"messageSeqs": None})])
    inherited = [
        row("compaction/start", {"compactionId": "old", "turn": None}),
        row("turn/start", {"turn": 1}),
        input_(),
        row(
            "compaction/summary",
            {"compactionId": "old", "shadowedRange": {"start": 2, "end": 2}, "shadowedSeqs": [2]},
        ),
        row("turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
        row("session/end-seed", {"inherited": True}),
    ]
    with pytest.raises(SessionFormatError, match=r"does not match the open turn"):
        reopen(inherited, 5, True)
