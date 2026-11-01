"""Ported from packages/core/session/tests/surface.spec.ts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from deskagents.core.session import (
    Session,
    SessionEvent,
    SurfaceFoldReplacement,
    SurfaceManager,
    fold_surface,
    is_append_surface_event,
    is_replacement_surface_event,
    is_surface_eligible_type,
    is_surface_event,
    session_id,
    session_log_offset,
    session_seq,
    snapshot_session_event,
)
from deskagents.llm import (
    Message,
    MessageId,
    ToolCallId,
    create_developer_message,
    create_message,
    create_system_message,
    create_tool_result_message,
    create_user_message,
    freeze_message,
)

SPEC = "packages/core/session/tests/surface.spec.ts"
REFERENCES = "foldSurface source-event references"
REWRITES = "foldSurface tool-result rewrites"
MANAGER = "SurfaceManager"
DERIVE = "deriveMessages with surface"
OPTS = "Session.append surface opts"
GUARDS = "surface type guards"
GENERATION = "SurfaceManager.replaceGeneration"
SYSTEM = "system/message surface node"
DEVELOPER = "developer message history"

_UNSET: Any = object()


def replace_op(start: int, end: int) -> dict[str, Any]:
    return {"op": "replace", "startSeq": session_seq(start), "endSeq": session_seq(end)}


def checkpoint_source(compaction_id: str) -> dict[str, str]:
    """Build a checkpoint source for a surface replacement fixture."""
    return {"kind": "compact-checkpoint", "compactionId": compaction_id}


def text(value: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": value}]


def user_message(value: str) -> Message:
    return create_user_message(content=text(value), source={"kind": "user"})


def replacement_message(value: str) -> Message:
    return create_user_message(content=text(value), source={"kind": "test"})


def assistant_data(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stream": [],
        "turn": 1,
        "step": 1,
        "message": create_message(
            role="assistant",
            content=content,
            source={"kind": "model", "provider": "mock", "model": "mock"},
        ),
    }


def tool_result_data(call_id: str, value: str) -> dict[str, Any]:
    return {
        "turn": 1,
        "step": 1,
        "message": create_tool_result_message(
            call_id=ToolCallId(call_id), content=text(value), is_error=False
        ),
    }


def matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """``toMatchObject`` over one flat mapping."""
    return all(key in actual and actual[key] == value for key, value in expected.items())


def surface_session() -> Session:
    """Build a minimal session with turn boundaries and a single user message."""
    s = Session.create(session_id("ss"))
    s.append("turn/start", {"turn": 1})
    s.append("user/message", user_message("hello"), surface_op="append")
    s.append("assistant/message", assistant_data(text("hi")), surface_op="append")
    s.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    return s


def source_event_reference_event(seq: int, source_event_seqs: Any = _UNSET) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "user/message",
        "seq": session_seq(seq),
        "time": seq,
        "data": create_user_message(content=[], source={"kind": "user"}),
        "surfaceOp": "append",
    }
    if source_event_seqs is not _UNSET:
        event["sourceEventSeqs"] = source_event_seqs
    return event


def tool_result_event(
    seq: int,
    call_id: str,
    op: Any = "append",
    source_event_seqs: list[int] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "tool/result",
        "seq": session_seq(seq),
        "time": seq,
        "data": tool_result_data(call_id, f"result {seq}"),
        "surfaceOp": op,
    }
    if source_event_seqs is not None:
        event["sourceEventSeqs"] = [session_seq(value) for value in source_event_seqs]
    return event


def markerless_user_event(seq: int = 0, time: int = 0) -> SessionEvent:
    return {
        "type": "user/message",
        "seq": session_seq(seq),
        "time": time,
        "data": user_message("hi"),
    }


# --- foldSurface source-event references ---------------------------------------------


@pytest.mark.upstream(
    f"{SPEC}::{REFERENCES} > accepts absent or valid source-event references and complete "
    "replacement coverage"
)
def test_accepts_valid_source_event_references() -> None:
    events = [
        source_event_reference_event(0),
        source_event_reference_event(1),
        {**source_event_reference_event(2, [0, 1]), "surfaceOp": replace_op(0, 1)},
    ]
    fold_surface(events)


@pytest.mark.upstream(
    f"{SPEC}::{REFERENCES} > rejects source-event references on a non-surface event"
)
def test_rejects_source_event_references_on_non_surface_event() -> None:
    event = {
        "type": "turn/start",
        "seq": session_seq(0),
        "time": 1,
        "data": {"turn": 1},
        "sourceEventSeqs": [session_seq(0)],
    }
    with pytest.raises(ValueError, match="cannot carry sourceEventSeqs"):
        fold_surface([event])


@pytest.mark.upstream(
    f"{SPEC}::{REFERENCES} > rejects obsolete Assistant chunk references on an assistant message"
)
def test_rejects_assistant_chunk_references() -> None:
    event = {
        "type": "assistant/message",
        "seq": session_seq(0),
        "time": 0,
        "data": assistant_data([]),
        "surfaceOp": "append",
        "sourceEventSeqs": [],
    }
    with pytest.raises(ValueError, match="embeds its source stream"):
        fold_surface([event])


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [{**source_event_reference_event(0), "sourceEventSeqs": "invalid"}],
            "must be an array",
            id="a non-array",
        ),
        pytest.param(
            [source_event_reference_event(0, [])], "must not be empty", id="an empty array"
        ),
        pytest.param(
            [source_event_reference_event(0), source_event_reference_event(1, [0, 0])],
            "must not contain duplicates",
            id="duplicates",
        ),
        # Python lists cannot be sparse; a hole serializes to JSON null.
        pytest.param(
            [source_event_reference_event(0, [None])], "densely contain", id="a sparse array"
        ),
        pytest.param(
            [{**source_event_reference_event(0), "sourceEventSeqs": ["0"]}],
            "non-negative safe integers",
            id="a non-number",
        ),
        pytest.param(
            [source_event_reference_event(0, [0.5])],
            "non-negative safe integers",
            id="a fractional number",
        ),
        pytest.param(
            [source_event_reference_event(0, [-1])],
            "non-negative safe integers",
            id="a negative number",
        ),
        pytest.param(
            [source_event_reference_event(0, [0])],
            "must reference earlier events",
            id="a self reference",
        ),
        pytest.param(
            [source_event_reference_event(0), source_event_reference_event(2, [1])],
            "seq 2 is not contiguous; expected 1",
            id="a non-contiguous event seq",
        ),
        pytest.param(
            [
                source_event_reference_event(0),
                source_event_reference_event(1),
                {
                    **source_event_reference_event(2, [0]),
                    "surfaceOp": {"op": "replace", "startSeq": 0, "endSeq": 1},
                },
            ],
            "missing 1",
            id="incomplete replacement coverage",
        ),
    ],
)
@pytest.mark.upstream(f"{SPEC}::{REFERENCES} > rejects %s")
def test_rejects_invalid_source_event_references(events: list[SessionEvent], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        fold_surface(events)


# --- foldSurface tool-result rewrites ------------------------------------------------


@pytest.mark.upstream(f"{SPEC}::{REWRITES} > rejects a replacement spanning multiple current nodes")
def test_rejects_rewrite_spanning_multiple_nodes() -> None:
    events = [
        source_event_reference_event(0),
        source_event_reference_event(1),
        tool_result_event(2, "rewrite", replace_op(0, 1), [0, 1]),
    ]
    with pytest.raises(ValueError, match="must rewrite exactly one current node"):
        fold_surface(events)


@pytest.mark.upstream(f"{SPEC}::{REWRITES} > rejects a replacement targeting a non-result node")
def test_rejects_rewrite_targeting_non_result() -> None:
    events = [
        source_event_reference_event(0),
        tool_result_event(1, "rewrite", replace_op(0, 0), [0]),
    ]
    with pytest.raises(ValueError, match=r"must target a current tool/result"):
        fold_surface(events)


@pytest.mark.upstream(f"{SPEC}::{REWRITES} > rejects changes outside tool-result content")
def test_rejects_changes_outside_tool_result_content() -> None:
    events = [
        tool_result_event(0, "original"),
        tool_result_event(1, "changed", replace_op(0, 0), [0]),
    ]
    with pytest.raises(ValueError, match="may change only content"):
        fold_surface(events)


@pytest.mark.parametrize(
    "patch",
    [
        pytest.param({"toolCallId": ToolCallId("changed")}, id="toolCallId"),
        pytest.param({"isError": True}, id="isError"),
    ],
)
@pytest.mark.upstream(
    f"{SPEC}::{REWRITES} > rejects a replacement that changes the result block %s"
)
def test_rejects_rewrite_changing_result_block(patch: dict[str, Any]) -> None:
    original = tool_result_event(0, "original")
    data = original["data"]
    replacement = {
        **original,
        "seq": session_seq(1),
        "time": 1,
        "data": {**data, "message": freeze_message({**data["message"], **patch})},
        "surfaceOp": replace_op(0, 0),
        "sourceEventSeqs": [session_seq(0)],
    }
    with pytest.raises(ValueError, match="may change only content"):
        fold_surface([original, replacement])


@pytest.mark.upstream(
    f"{SPEC}::{REWRITES} > compares array-valued rest fields structurally (meta arrays: equal "
    "accepted, drifted rejected)"
)
def test_compares_array_valued_rest_fields_structurally() -> None:
    def with_meta(
        seq: int, meta: object, op: Any = "append", source_event_seqs: list[int] | None = None
    ) -> SessionEvent:
        event = tool_result_event(seq, "c-meta", op, source_event_seqs)
        data = event["data"]
        message = freeze_message({**data["message"], "id": MessageId("meta-message")})
        return {**event, "data": {**data, "message": message, "meta": meta}}

    # Structurally equal arrays (fresh references) pass the rest-field equality.
    fold_surface(
        [
            with_meta(0, {"tags": ["a", {"n": 1}]}),
            with_meta(1, {"tags": ["a", {"n": 1}]}, replace_op(0, 0), [0]),
        ]
    )
    # Same length, drifted element: the array branch must reject.
    with pytest.raises(ValueError, match="may change only content"):
        fold_surface(
            [with_meta(0, {"tags": ["a"]}), with_meta(1, {"tags": ["b"]}, replace_op(0, 0), [0])]
        )
    # Array vs non-array on one side: the mixed-shape guard rejects.
    with pytest.raises(ValueError, match="may change only content"):
        fold_surface(
            [with_meta(0, {"tags": ["a"]}), with_meta(1, {"tags": "a"}, replace_op(0, 0), [0])]
        )
    # Same key count, different key names: the hasOwn branch rejects.
    with pytest.raises(ValueError, match="may change only content"):
        fold_surface([with_meta(0, {"left": 1}), with_meta(1, {"right": 1}, replace_op(0, 0), [0])])
    # Different key counts: the key-length branch rejects.
    with pytest.raises(ValueError, match="may change only content"):
        fold_surface(
            [
                with_meta(0, {"one": 1}),
                with_meta(1, {"one": 1, "two": 2}, replace_op(0, 0), [0]),
            ]
        )


# --- SurfaceManager ------------------------------------------------------------------


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > folds a contiguous window without materializing earlier event sequences"
)
def test_folds_contiguous_window() -> None:
    base_seq = 400_000
    events = [
        source_event_reference_event(base_seq),
        source_event_reference_event(base_seq + 1),
        {
            **source_event_reference_event(base_seq + 2, [base_seq]),
            "surfaceOp": replace_op(base_seq, base_seq),
        },
    ]
    surface = SurfaceManager(events, session_log_offset(base_seq))
    assert list(surface.nodes) == [base_seq + 2, base_seq + 1]
    assert surface.replace_generation == 1


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > validates tool-result rewrites against a nonzero window offset"
)
def test_validates_tool_result_rewrites_with_window_offset() -> None:
    base_seq = 400_000
    original = tool_result_event(base_seq, "call")
    events = [
        original,
        {
            **original,
            "seq": session_seq(base_seq + 1),
            "time": base_seq + 1,
            "surfaceOp": replace_op(base_seq, base_seq),
            "sourceEventSeqs": [session_seq(base_seq)],
        },
    ]
    assert list(SurfaceManager(events, session_log_offset(base_seq)).nodes) == [base_seq + 1]


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > rejects a replacement that crosses a loaded window head"
)
def test_rejects_replacement_crossing_window_head() -> None:
    base_seq = 400_000
    events = [
        source_event_reference_event(base_seq),
        {
            **source_event_reference_event(base_seq + 1, [base_seq - 1, base_seq]),
            "surfaceOp": replace_op(base_seq - 1, base_seq),
        },
    ]
    surface = SurfaceManager(events, session_log_offset(base_seq))
    with pytest.raises(
        ValueError, match=f"surface replace: start seq {base_seq - 1} not found in surface"
    ):
        _ = surface.nodes


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > shares ordered entries and nested replacement ranges with foldSurface"
)
def test_shares_entries_and_replacement_ranges_with_fold() -> None:
    s = Session.create(session_id("shared-fold"))
    s.append("user/message", user_message("a"), surface_op="append")
    s.append("user/message", user_message("b"), surface_op="append")
    s.append(
        "user/message",
        replacement_message("summary"),
        surface_op=replace_op(0, 0),
        source_event_seqs=[session_seq(0)],
    )
    s.append(
        "user/message",
        replacement_message("summary 2"),
        surface_op=replace_op(2, 1),
        source_event_seqs=[session_seq(2), session_seq(1)],
    )

    folded = fold_surface(s.snapshot_events())
    assert folded.nodes == list(s.surface.nodes)
    assert folded.replacements == [
        SurfaceFoldReplacement(session_seq(2), session_seq(0), session_seq(0), [session_seq(0)]),
        SurfaceFoldReplacement(
            session_seq(3), session_seq(2), session_seq(1), [session_seq(2), session_seq(1)]
        ),
    ]
    folded.nodes[0] = session_seq(99)
    folded.replacements[0].shadowed_seqs.append(session_seq(99))
    assert list(s.surface.nodes) == [3]
    assert fold_surface(s.snapshot_events()).nodes == [3]
    assert fold_surface(s.snapshot_events()).replacements[0].shadowed_seqs == [0]


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > does not retain fold-only replacement history in incremental state"
)
def test_does_not_retain_replacement_history_incrementally() -> None:
    s = Session.create(session_id("incremental-state"))
    s.append("user/message", user_message("a"), surface_op="append")
    s.append(
        "user/message",
        replacement_message("b"),
        surface_op=replace_op(0, 0),
        source_event_seqs=[session_seq(0)],
    )

    assert list(s.surface.nodes) == [1]
    assert not hasattr(s.surface._state, "replacements")
    assert fold_surface(s.snapshot_events()).replacements == [
        SurfaceFoldReplacement(session_seq(1), session_seq(0), session_seq(0), [session_seq(0)])
    ]


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > foldSurface reports the same invalid replacement failures as the "
    "incremental manager"
)
def test_fold_reports_same_failures_as_manager() -> None:
    events = [
        source_event_reference_event(0),
        {"type": "turn/start", "seq": session_seq(1), "time": 1, "data": {"turn": 1}},
        {
            **source_event_reference_event(2, [0]),
            "surfaceOp": {"op": "replace", "startSeq": 1, "endSeq": 0},
        },
    ]
    with pytest.raises(ValueError, match="start seq 1 not found"):
        fold_surface(events)
    with pytest.raises(ValueError, match="start seq 1 not found"):
        Session.create(session_id("shared-fold-invalid"), events)


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > rejects negative-zero replacement event sequences")
def test_rejects_negative_zero_replacement_sequences() -> None:
    # Python ints have no negative zero; the float -0.0 is the JSON value that differs from 0.
    event = {
        **source_event_reference_event(0, [0]),
        "surfaceOp": {"op": "replace", "startSeq": -0.0, "endSeq": -0.0},
    }
    with pytest.raises(ValueError, match="invalid replace surfaceOp"):
        fold_surface([event])


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > leaves incremental state unchanged when candidate validation fails"
)
def test_leaves_state_unchanged_on_failed_validation() -> None:
    s = Session.create(session_id("atomic-validation"))
    s.append("user/message", user_message("a"), surface_op="append")
    surface = s.surface
    nodes = surface.nodes
    # Python's nodes is an immutable tuple snapshot; the live list is the fold state's.
    state = surface._state
    live_nodes = state.nodes

    assert list(nodes) == fold_surface(s.snapshot_events()).nodes
    assert surface.replace_generation == 0

    with pytest.raises(ValueError, match="missing 0"):
        s.append("assistant/message", assistant_data(text("invalid")), surface_op=replace_op(0, 0))

    assert len(s.snapshot_events()) == 1
    assert s.surface is surface
    assert list(surface.nodes) == [0]
    assert surface.replace_generation == 0
    assert list(surface.nodes) == fold_surface(s.snapshot_events()).nodes

    s.append("user/message", user_message("b"), surface_op="append")
    assert surface._state is state
    assert surface._state.nodes is live_nodes
    assert list(surface.nodes) == [0, 1]
    assert surface.replace_generation == 0
    assert list(surface.nodes) == fold_surface(s.snapshot_events()).nodes


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > foldSurface rejects a surface-eligible event without its mandatory marker"
)
def test_fold_rejects_markerless_surface_event() -> None:
    malformed: SessionEvent = {
        "type": "user/message",
        "seq": session_seq(0),
        "time": 1,
        "data": user_message("hidden"),
    }
    with pytest.raises(ValueError, match="surface-eligible and requires a surfaceOp marker"):
        fold_surface([malformed])


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > foldSurface rejects surfaceOp on a non-surface event")
def test_fold_rejects_surface_op_on_non_surface_event() -> None:
    malformed: SessionEvent = {
        "type": "turn/start",
        "seq": session_seq(0),
        "time": 1,
        "data": {"turn": 1},
        "surfaceOp": "append",
    }
    with pytest.raises(ValueError, match="not surface-eligible and cannot carry surfaceOp"):
        fold_surface([malformed])


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > folds an ordered sequence list from surfaceOp: append markers"
)
def test_folds_ordered_sequence_from_append_markers() -> None:
    s = surface_session()
    # Only the user/message and assistant/message carry surfaceOp: "append".
    # The turn boundaries do not have surface markers.
    assert list(s.surface.nodes) == [1, 2]


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > empty surface yields empty nodes")
def test_empty_surface_yields_empty_nodes() -> None:
    s = Session.create(session_id("empty"))
    s.append("turn/start", {"turn": 1})
    s.append("step/start", {"turn": 1, "step": 1})
    s.append("step/end", {"turn": 1, "step": 1})
    s.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert len(s.surface.nodes) == 0
    assert s.derive_messages() == []


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > picks up new events incrementally (delta processing)")
def test_picks_up_new_events_incrementally() -> None:
    s = surface_session()
    assert len(s.surface.nodes) == 2
    s.append("tool/result", tool_result_data("c1", "ok"), surface_op="append")
    assert len(s.surface.nodes) == 3
    assert s.surface.nodes[2] == 4  # seq 4: after turn/end at seq 3


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > replays identically from a seeded log with surface markers"
)
def test_replays_identically_from_seeded_log() -> None:
    original = surface_session()
    original.append("tool/result", tool_result_data("c1", "ok"), surface_op="append")
    replayed = Session.create(session_id("replay"), original.snapshot_events())
    assert list(replayed.surface.nodes) == [1, 2, 4]
    assert replayed.derive_messages() == original.derive_messages()


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > rebuild with replace operation splices out shadowed nodes"
)
def test_replace_splices_out_shadowed_nodes() -> None:
    s = surface_session()
    s.append(
        "user/message",
        replacement_message("summary"),
        surface_op=replace_op(1, 2),
        source_event_seqs=[session_seq(1), session_seq(2)],
    )
    assert list(s.surface.nodes) == [4]


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > replace with both ends at real nodes splices only the range"
)
def test_replace_splices_only_the_range() -> None:
    s = Session.create(session_id("range"))
    s.append("user/message", user_message("a"), surface_op="append")  # seq 0
    s.append("user/message", user_message("b"), surface_op="append")  # seq 1
    s.append("user/message", user_message("c"), surface_op="append")  # seq 2
    # Replace seq 0 through 1 inclusive: shadow a and b, keep c.
    s.append(
        "user/message",
        replacement_message("summary"),
        surface_op=replace_op(0, 1),
        source_event_seqs=[session_seq(0), session_seq(1)],
    )  # seq 3
    assert list(s.surface.nodes) == [3, 2]


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > single-node replacement (start === end)")
def test_single_node_replacement() -> None:
    s = Session.create(session_id("single"))
    s.append("user/message", user_message("a"), surface_op="append")  # seq 0
    s.append("user/message", user_message("b"), surface_op="append")  # seq 1
    # Replace only seq 1 (single node).
    s.append(
        "user/message",
        replacement_message("x"),
        surface_op=replace_op(1, 1),
        source_event_seqs=[session_seq(1)],
    )  # seq 2
    assert list(s.surface.nodes) == [0, 2]


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > throws when replace start is not found")
def test_throws_when_replace_start_not_found() -> None:
    s = Session.create(session_id("bad-start"))
    s.append("user/message", user_message("a"), surface_op="append")  # seq 0
    s.append("turn/start", {"turn": 1})
    with pytest.raises(ValueError, match="surface replace: start seq 1 not found"):
        s.append(
            "user/message",
            replacement_message("y"),
            surface_op=replace_op(1, 0),
            source_event_seqs=[session_seq(0)],
        )


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > throws when replace end is not found")
def test_throws_when_replace_end_not_found() -> None:
    s = Session.create(session_id("bad-end"))
    s.append("user/message", user_message("a"), surface_op="append")  # seq 0
    s.append("turn/start", {"turn": 1})
    with pytest.raises(ValueError, match="surface replace: end seq 1 not found"):
        s.append(
            "user/message",
            replacement_message("y"),
            surface_op=replace_op(0, 1),
            source_event_seqs=[session_seq(0)],
        )


@pytest.mark.upstream(f"{SPEC}::{MANAGER} > throws when start is after end")
def test_throws_when_start_after_end() -> None:
    s = Session.create(session_id("reversed"))
    s.append("user/message", user_message("a"), surface_op="append")  # seq 0
    s.append("user/message", user_message("b"), surface_op="append")  # seq 1
    # start=1, end=0 would be reversed order.
    with pytest.raises(ValueError, match="start seq 1.*after end seq 0"):
        s.append(
            "user/message",
            replacement_message("y"),
            surface_op=replace_op(1, 0),
            source_event_seqs=[session_seq(1), session_seq(0)],
        )


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > sourceEventSeqs is snapshot so caller mutation does not affect "
    "logged event"
)
def test_source_event_seqs_are_snapshot() -> None:
    s = Session.create(session_id("immutable"))
    s.append("user/message", user_message("source"), surface_op="append")
    sources = [session_seq(0)]
    s.append(
        "user/message", replacement_message("h"), surface_op="append", source_event_seqs=sources
    )
    # Mutate caller's list after append.
    sources.append(session_seq(1))
    sources[0] = session_seq(99)
    logged = s.snapshot_events()[1]
    assert logged["sourceEventSeqs"] == [0]


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > replace starting at non-head position preserves surrounding order"
)
def test_non_head_replace_preserves_order() -> None:
    s = Session.create(session_id("mid-replace"))
    s.append("user/message", user_message("a"), surface_op="append")  # seq 0
    s.append("user/message", user_message("b"), surface_op="append")  # seq 1
    s.append("user/message", user_message("c"), surface_op="append")  # seq 2
    # Replace the middle node (seq 1) only, keeping seq 0 and seq 2.
    s.append(
        "user/message",
        replacement_message("x"),
        surface_op=replace_op(1, 1),
        source_event_seqs=[session_seq(1)],
    )  # seq 3
    assert list(s.surface.nodes) == [0, 3, 2]


@pytest.mark.upstream(
    f"{SPEC}::{MANAGER} > surfaceOp replace object is snapshot so caller mutation is isolated"
)
def test_surface_op_replace_object_is_snapshot() -> None:
    s = Session.create(session_id("immutable-op"))
    s.append("user/message", user_message("a"), surface_op="append")
    op = replace_op(0, 0)
    s.append(
        "user/message", replacement_message("s"), surface_op=op, source_event_seqs=[session_seq(0)]
    )
    # Mutate caller's mapping after append.
    op["startSeq"] = session_seq(99)
    logged = s.snapshot_events()[1]
    assert logged["surfaceOp"] == {"op": "replace", "startSeq": 0, "endSeq": 0}


# --- deriveMessages with surface -----------------------------------------------------


@pytest.mark.upstream(f"{SPEC}::{DERIVE} > uses the surface path when surface markers are present")
def test_uses_surface_path_with_markers() -> None:
    s = surface_session()
    messages = s.derive_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert matches(messages[0]["content"][0], {"type": "text", "text": "hello"})
    assert messages[1]["role"] == "assistant"
    assert matches(messages[1]["content"][0], {"type": "text", "text": "hi"})


@pytest.mark.upstream(
    f"{SPEC}::{DERIVE} > surface path skips non-surface events (attempts, boundaries)"
)
def test_surface_path_skips_non_surface_events() -> None:
    s = Session.create(session_id("filter"))
    s.append("turn/start", {"turn": 1})
    s.append(
        "assistant/attempt",
        {
            "turn": 1,
            "step": 1,
            "stream": [
                {"type": "text-chunks", "time0": 1, "index": 0, "dt": [1], "texts": ["h", "i"]}
            ],
        },
    )
    s.append("user/message", user_message("hello"), surface_op="append")
    s.append("assistant/message", assistant_data(text("hi")), surface_op="append")
    s.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    # Attempts and boundaries are NOT in the surface, so only 2 messages.
    assert len(s.derive_messages()) == 2


@pytest.mark.upstream(
    f"{SPEC}::{DERIVE} > deriveMessages via surface respects replace (shadowed nodes are excluded)"
)
def test_derive_messages_respects_replace() -> None:
    s = Session.create(session_id("compacted"))
    s.append("user/message", user_message("original"), surface_op="append")
    s.append(
        "user/message",
        replacement_message("compacted"),
        surface_op=replace_op(0, 0),
        source_event_seqs=[session_seq(0)],
    )
    # Only the compaction node is visible.
    messages = s.derive_messages()
    assert len(messages) == 1
    assert matches(messages[0]["content"][0], {"type": "text", "text": "compacted"})


@pytest.mark.upstream(f"{SPEC}::{DERIVE} > injected-context and user messages appear on surface")
def test_injected_context_and_user_messages_appear() -> None:
    s = Session.create(session_id("ctx"))
    s.append(
        "user/message",
        create_user_message(content=text("file changed"), source={"kind": "watcher"}),
        surface_op="append",
    )
    s.append("user/message", user_message("focus"), surface_op="append")
    messages = s.derive_messages()
    assert len(messages) == 2
    assert messages[0]["content"] == text("file changed")
    assert messages[1]["content"] == text("focus")


# --- Session.append surface opts -----------------------------------------------------


@pytest.mark.upstream(
    f"{SPEC}::{OPTS} > records sourceEventSeqs and surfaceOp on a source-derived event"
)
def test_records_source_event_seqs_and_surface_op() -> None:
    s = Session.create(session_id("opts"))
    s.append("turn/start", {"turn": 1})
    s.append("step/start", {"turn": 1, "step": 1})
    event = s.append(
        "user/message",
        create_user_message(content=text("h"), source={"kind": "test"}),
        surface_op="append",
        source_event_seqs=[session_seq(0), session_seq(1)],
    )
    assert event["sourceEventSeqs"] == [0, 1]
    assert event["surfaceOp"] == "append"
    # The logged event matches the returned event.
    assert s.snapshot_events()[2]["sourceEventSeqs"] == [0, 1]
    assert s.snapshot_events()[2]["surfaceOp"] == "append"


@pytest.mark.upstream(
    f"{SPEC}::{OPTS} > deriveMessages skips a surface node that derives to null (empty "
    "assistant/message)"
)
def test_derive_messages_skips_null_surface_node() -> None:
    # An empty-content assistant/message is surface-eligible (it can host usage) but
    # projects to None, so the node is on the surface yet produces no message.
    seed: list[SessionEvent] = [
        {"type": "turn/start", "seq": session_seq(0), "time": 1, "data": {"turn": 1}},
        {"type": "step/start", "seq": session_seq(1), "time": 2, "data": {"turn": 1, "step": 1}},
        {
            "type": "assistant/message",
            "seq": session_seq(2),
            "time": 3,
            "data": assistant_data([]),
            "surfaceOp": "append",
        },
        {"type": "step/end", "seq": session_seq(3), "time": 4, "data": {"turn": 1, "step": 1}},
        {
            "type": "turn/end",
            "seq": session_seq(4),
            "time": 5,
            "data": {"turn": 1, "reason": {"kind": "completed"}},
        },
    ]
    s = Session.create(session_id("nomessage"), seed)
    assert len(s.derive_messages()) == 0


@pytest.mark.upstream(f"{SPEC}::{OPTS} > a non-surface event carries no surface fields")
def test_non_surface_event_carries_no_surface_fields() -> None:
    s = Session.create(session_id("noopts"))
    s.append("turn/start", {"turn": 1})
    assert "sourceEventSeqs" not in s.snapshot_events()[0]
    assert "surfaceOp" not in s.snapshot_events()[0]


@pytest.mark.upstream(f"{SPEC}::{OPTS} > surfaceOp primitives are not cloned (they are immutable)")
def test_surface_op_primitives_are_not_cloned() -> None:
    s = Session.create(session_id("prim"))
    event = s.append("assistant/message", assistant_data([]), surface_op="append")
    # The string "append" is immutable; identity-preserving is fine.
    assert event["surfaceOp"] == "append"


@pytest.mark.upstream(
    f"{SPEC}::{OPTS} > isSurfaceEvent rejects a surface-eligible type missing its surfaceOp marker"
)
def test_is_surface_event_rejects_markerless_event_opts() -> None:
    # A raw event (not built via append, which mandates the marker) of a surface-eligible
    # type but with no surfaceOp must not count as a surface event.
    no_marker = markerless_user_event(time=1)
    assert is_surface_event(no_marker) is False
    # A non-surface type is rejected too (the type gate).
    boundary: SessionEvent = {
        "type": "turn/start",
        "seq": session_seq(1),
        "time": 1,
        "data": {"turn": 1},
    }
    assert is_surface_event(boundary) is False
    # A properly-marked surface event is accepted.
    marked = {**no_marker, "surfaceOp": "append"}
    assert is_surface_event(marked) is True


# --- surface type guards -------------------------------------------------------------


@pytest.mark.upstream(
    f"{SPEC}::{GUARDS} > isSurfaceEligibleType is true only for message-producing types"
)
def test_is_surface_eligible_type() -> None:
    assert is_surface_eligible_type("user/message") is True
    assert is_surface_eligible_type("assistant/message") is True
    assert is_surface_eligible_type("tool/result") is True
    assert is_surface_eligible_type("turn/start") is False
    assert is_surface_eligible_type("assistant/attempt") is False


@pytest.mark.upstream(f"{SPEC}::{GUARDS} > isSurfaceEvent narrows a fully-formed surface event")
def test_is_surface_event_accepts_formed_event() -> None:
    s = surface_session()
    user = next(e for e in s.snapshot_events() if e["type"] == "user/message")
    assert is_surface_event(user) is True


@pytest.mark.upstream(f"{SPEC}::{GUARDS} > isSurfaceEvent rejects a non-surface-eligible type")
def test_is_surface_event_rejects_non_eligible_type() -> None:
    s = surface_session()
    turn_start = next(e for e in s.snapshot_events() if e["type"] == "turn/start")
    assert is_surface_event(turn_start) is False


@pytest.mark.upstream(
    f"{SPEC}::{GUARDS} > isSurfaceEvent rejects a surface-eligible type missing its surfaceOp "
    "marker"
)
def test_is_surface_event_rejects_markerless_event_guards() -> None:
    # A seed/load record may lack its required marker before validation.
    markerless = markerless_user_event()
    assert is_surface_eligible_type(markerless["type"]) is True
    assert is_surface_event(markerless) is False


@pytest.mark.upstream(
    f"{SPEC}::{GUARDS} > splits surface events into append-origin and replacement by their marker"
)
def test_splits_append_and_replacement_events() -> None:
    s = surface_session()
    s.append(
        "user/message",
        create_user_message(
            content=text("checkpoint"), source=checkpoint_source("surface-compaction-1")
        ),
        surface_op=replace_op(1, 2),
        source_event_seqs=[session_seq(1), session_seq(2)],
    )
    appended = next(e for e in s.snapshot_events() if e["type"] == "user/message")
    replacement = s.snapshot_events()[-1]

    assert is_append_surface_event(appended) is True
    assert is_replacement_surface_event(appended) is False
    assert is_append_surface_event(replacement) is False
    assert is_replacement_surface_event(replacement) is True


@pytest.mark.upstream(
    f"{SPEC}::{GUARDS} > rejects log-only and markerless events from both marker guards"
)
def test_marker_guards_reject_log_only_and_markerless() -> None:
    s = surface_session()
    turn_start = next(e for e in s.snapshot_events() if e["type"] == "turn/start")
    # A surface-eligible type whose mandatory marker is absent has no origin at all:
    # it never entered the surface.
    markerless = markerless_user_event()

    assert is_append_surface_event(turn_start) is False
    assert is_replacement_surface_event(turn_start) is False
    assert is_append_surface_event(markerless) is False
    assert is_replacement_surface_event(markerless) is False


# --- SurfaceManager.replaceGeneration ------------------------------------------------


@pytest.mark.upstream(
    f"{SPEC}::{GENERATION} > folds the pending log delta on access and counts replaces"
)
def test_replace_generation_folds_pending_delta() -> None:
    s = Session.create(session_id("gen"))
    s.append("turn/start", {"turn": 1})
    s.append("user/message", user_message("one"), surface_op="append")
    s.append("user/message", user_message("two"), surface_op="append")
    # Read the generation first, before nodes, so the getter itself folds the
    # pending delta rather than piggybacking on a nodes read.
    assert s.surface.replace_generation == 0

    nodes = s.surface.nodes
    s.append(
        "user/message",
        create_user_message(
            content=text("summary"), source=checkpoint_source("surface-compaction-2")
        ),
        surface_op={"op": "replace", "startSeq": nodes[0], "endSeq": nodes[1]},
        source_event_seqs=[nodes[0], nodes[1]],
    )
    assert s.surface.replace_generation == 1


# --- system/message surface node -----------------------------------------------------


def system_event(
    seq: int, value: str, op: Any = "append", sources: list[int] | None = None
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "system/message",
        "seq": session_seq(seq),
        "time": seq,
        "data": {"turn": 1, "step": 1, "message": create_system_message(value)},
        "surfaceOp": op,
    }
    if sources is not None:
        event["sourceEventSeqs"] = [session_seq(source) for source in sources]
    return event


def user_event(seq: int, op: Any = "append", sources: list[int] | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "user/message",
        "seq": session_seq(seq),
        "time": seq,
        "data": user_message(f"u{seq}"),
        "surfaceOp": op,
    }
    if sources is not None:
        event["sourceEventSeqs"] = [session_seq(source) for source in sources]
    return event


@pytest.mark.upstream(
    f"{SPEC}::{SYSTEM} > projects a system node as the leading system-role message and an empty "
    "one as no message"
)
def test_projects_system_node_as_leading_message() -> None:
    s = Session.create(session_id("sys"))
    s.append("turn/start", {"turn": 1})
    s.append("step/start", {"turn": 1, "step": 1})
    s.append(
        "system/message",
        {"turn": 1, "step": 1, "message": create_system_message("be brief")},
        surface_op="append",
    )
    s.append("user/message", user_message("hi"), surface_op="append")
    assert [message["role"] for message in s.derive_messages()] == ["system", "user"]
    assert s.derive_messages()[0]["content"] == text("be brief")

    head = s.surface.nodes[0]
    s.append(
        "system/message",
        {"turn": 1, "step": 1, "message": create_system_message("")},
        surface_op={"op": "replace", "startSeq": head, "endSeq": head},
        source_event_seqs=[head],
    )
    assert [message["role"] for message in s.derive_messages()] == ["user"]
    assert len(s.surface.nodes) == 2


@pytest.mark.upstream(
    f"{SPEC}::{SYSTEM} > replaces node 0 with a new system node and rejects other rewrites of "
    "the head"
)
def test_replaces_system_head_and_rejects_other_rewrites() -> None:
    replaced = fold_surface(
        [system_event(0, "v1"), user_event(1), system_event(2, "v2", replace_op(0, 0), [0])]
    )
    assert replaced.nodes == [2, 1]

    with pytest.raises(ValueError, match="node 0 holds the system prompt"):
        fold_surface(
            [system_event(0, "v1"), user_event(1), user_event(2, replace_op(0, 1), [0, 1])]
        )
    with pytest.raises(ValueError, match="node 0 holds the system prompt"):
        fold_surface(
            [
                system_event(0, "v1"),
                user_event(1),
                system_event(2, "v2", replace_op(0, 1), [0, 1]),
            ]
        )


@pytest.mark.upstream(
    f"{SPEC}::{SYSTEM} > leaves later system nodes and a non-system head unprotected"
)
def test_later_system_nodes_and_plain_head_unprotected() -> None:
    later = fold_surface(
        [
            system_event(0, "v1"),
            user_event(1),
            system_event(2, "v2"),
            user_event(3),
            user_event(4, replace_op(1, 3), [1, 2, 3]),
        ]
    )
    assert later.nodes == [0, 4]
    plain_head = fold_surface(
        [user_event(0), user_event(1), user_event(2, replace_op(0, 1), [0, 1])]
    )
    assert plain_head.nodes == [2]


@pytest.mark.upstream(
    f"{SPEC}::{SYSTEM} > rejects a seeded system/message with a non-system role or "
    "non-system-prompt source"
)
def test_rejects_seeded_system_message_with_bad_role_or_source() -> None:
    good = system_event(0, "v1")
    bad_role = {
        **good,
        "data": {**good["data"], "message": {**good["data"]["message"], "role": "user"}},
    }
    with pytest.raises(ValueError, match='role "system"'):
        Session.create(session_id("bad-role"), [bad_role])
    bad_source = {
        **good,
        "data": {
            **good["data"],
            "message": {**good["data"]["message"], "source": {"kind": "user"}},
        },
    }
    with pytest.raises(ValueError, match="system-prompt source"):
        Session.create(session_id("bad-source"), [bad_source])


# --- developer message history -------------------------------------------------------


@pytest.mark.parametrize("type_", ["user/message", "developer/message"])
@pytest.mark.upstream(
    f"{SPEC}::{DEVELOPER} > rejects a mismatched developer role in %s at event admission"
)
def test_rejects_mismatched_developer_role(type_: str) -> None:
    message = create_developer_message(source={"kind": "test"}, content=[])
    data: object = (
        message
        if type_ == "user/message"
        else {"turn": 1, "step": 1, "message": {**message, "role": "user"}}
    )
    event = {"type": type_, "seq": session_seq(0), "time": 0, "surfaceOp": "append", "data": data}
    with pytest.raises(ValueError, match="must occur together"):
        snapshot_session_event(event)


@pytest.mark.upstream(
    f"{SPEC}::{DEVELOPER} > retains an empty developer node while projecting no message"
)
def test_retains_empty_developer_node() -> None:
    session = Session.create(session_id("empty-developer"))
    message = create_developer_message(source={"kind": "test"}, content=[])
    event = session.append(
        "developer/message", {"turn": 1, "step": 1, "message": message}, surface_op="append"
    )
    assert any(logged is event for logged in session.snapshot_events())
    assert session.derive_messages() == []
    restored = Session.from_restore(
        session.id,
        session.snapshot_events(),
        session.header,
        session_log_offset(0),
        "shared-frozen",
    )
    assert restored.derive_messages() == []


@pytest.mark.parametrize("type_", ["tool-addition", "tool-removal"])
@pytest.mark.upstream(f"{SPEC}::{DEVELOPER} > rejects %s in user messages at Session admission")
def test_rejects_tool_change_in_user_messages(type_: str) -> None:
    session = Session.create(session_id("invalid-tool-change"))
    message = create_user_message(
        source={"kind": "test"}, content=[{"type": type_, "toolName": "search"}]
    )
    with pytest.raises(ValueError, match="developer role"):
        session.append("user/message", message, surface_op="append")


@pytest.mark.upstream(
    f"{SPEC}::{DEVELOPER} > retains developer blocks through append, replacement, and restoration"
)
def test_retains_developer_blocks() -> None:
    session = Session.create(session_id("developer"))
    message = create_developer_message(
        source={"kind": "test"}, content=[{"type": "tool-addition", "toolName": "search"}]
    )
    header_seq = session.append(
        "request/header",
        {
            "reason": "initial",
            "header": {
                "config": {"provider": "test", "model": "test"},
                "tools": [{"name": "search", "description": "Search", "parameters": {}}],
            },
        },
    )["seq"]
    first = session.append(
        "developer/message",
        {"turn": 1, "step": 1, "headerSeq": header_seq, "message": message},
        surface_op="append",
    )
    assert session.derive_messages() == [message]
    replacement = create_developer_message(
        source={"kind": "test"}, content=[{"type": "tool-removal", "toolName": "search"}]
    )
    session.append(
        "developer/message",
        {"turn": 1, "step": 1, "message": replacement},
        surface_op={"op": "replace", "startSeq": first["seq"], "endSeq": first["seq"]},
        source_event_seqs=[first["seq"]],
    )
    assert session.derive_messages() == [replacement]
    assert is_surface_eligible_type("developer/message") is True
    restored = Session.from_restore(
        session.id,
        session.snapshot_events(),
        session.header,
        session_log_offset(0),
        "shared-frozen",
    )
    assert restored.derive_messages() == [replacement]
