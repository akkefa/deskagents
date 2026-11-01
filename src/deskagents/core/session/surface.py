"""The ordered surface: the message-producing events that derive model history.

Upstream: packages/core/session/src/surface.ts. The append-only log remains the source
of truth; the surface is an ordered view over it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from deskagents.llm import Message
from deskagents.util.values import deep_equal_json, is_safe_integer

from .known_event_types import KNOWN_SESSION_EVENT_TYPES, MESSAGE_PROJECTION_EVENT_TYPES
from .types import SURFACE_EVENT_TYPES, SessionEvent, SessionLogOffset, SessionSeq, SurfaceOp


@dataclass(frozen=True)
class SessionMessageProjectionContext:
    """Readonly history immediately before a message-projection event."""

    nodes: Sequence[SessionSeq]
    """Current message-producing event sequences in model-visible order."""
    events: Sequence[SessionEvent]
    """Contiguous event window; entries at or beyond the candidate seq are not inputs."""
    base_seq: SessionLogOffset
    """Absolute sequence of the window's first event."""
    messages: Mapping[SessionSeq, Message]
    """Previously projected messages keyed by their original event sequences."""


class SessionMessageProjection(Protocol):
    """Pure interpretation of one plugin-owned event that changes existing message content."""

    @property
    def type(self) -> str:
        """Event type interpreted by this definition."""
        ...

    def project(
        self, event: SessionEvent, context: SessionMessageProjectionContext
    ) -> Mapping[SessionSeq, Message]:
        """Validate the whole decision, then return changed messages keyed by original seq."""
        ...


def is_surface_eligible_type(type_: str) -> bool:
    """Whether an event type can join the model-visible surface."""
    return type_ in SURFACE_EVENT_TYPES


def is_surface_event(event: SessionEvent) -> bool:
    """Whether both the type and the ``surfaceOp`` marker identify a surface event."""
    return event["type"] in SURFACE_EVENT_TYPES and "surfaceOp" in event


def is_append_surface_event(event: SessionEvent) -> bool:
    """An append-origin surface event: entered at its own position, never a replacement copy."""
    return is_surface_event(event) and event["surfaceOp"] == "append"


def is_replacement_surface_event(event: SessionEvent) -> bool:
    """A surface replacement: a node that shadowed an existing surface range."""
    return is_surface_event(event) and event["surfaceOp"] != "append"


def derive_event_message(
    event: SessionEvent, projected_messages: Mapping[SessionSeq, Message] | None = None
) -> Message | None:
    """Project one event into the message it derives to, or ``None`` when it produces none.

    Empty system, developer and assistant messages keep their surface position without
    adding a wire message. Pass the same prefix's ``projected_messages`` from
    :func:`fold_surface` to apply logged message projections.
    """
    if projected_messages is not None:
        projected = projected_messages.get(event["seq"])
        if projected is not None:
            return projected
    kind = event["type"]
    if kind == "user/message":
        return event["data"]  # type: ignore[no-any-return]
    if kind in ("system/message", "developer/message", "assistant/message"):
        message = event["data"]["message"]
        return None if len(message["content"]) == 0 else message
    if kind == "tool/result":
        return event["data"]["message"]  # type: ignore[no-any-return]
    return None


def _is_record(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_event_seq(value: object) -> bool:
    return is_safe_integer(value) and value >= 0 and isinstance(value, int)  # type: ignore[operator]


def validate_session_event_data(event: Mapping[str, Any], subject: str) -> None:
    """Reject noncanonical request headers, developer roles/content and tool failure metadata.

    This does not validate complete payloads or embedded provider streams.
    """
    data = event.get("data")
    kind = event.get("type")
    if kind in SURFACE_EVENT_TYPES and isinstance(data, Mapping):
        message = data if kind == "user/message" else data.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if (kind == "developer/message") != (message.get("role") == "developer"):
                raise ValueError(
                    f"{subject} developer/message and developer role must occur together"
                )
            if (
                message.get("role") != "developer"
                and isinstance(content, list)
                and any(
                    _is_record(block) and block.get("type") in ("tool-addition", "tool-removal")
                    for block in content
                )
            ):
                raise ValueError(f"{subject} tool-change blocks require developer role")
            if kind == "developer/message" and isinstance(content, list):
                has_additions = False
                for block in content:
                    if not _is_record(block) or block.get("type") not in (
                        "tool-addition",
                        "tool-removal",
                    ):
                        continue
                    name = block.get("toolName")
                    if not isinstance(name, str) or not name:
                        raise ValueError(f"{subject} {block['type']} requires a nonempty toolName")
                    if block["type"] == "tool-addition":
                        has_additions = True
                        if "tool" in block:
                            raise ValueError(
                                f"{subject} tool-addition must omit inline tool definitions"
                            )
                if (
                    not _is_event_seq(data.get("headerSeq"))
                    if has_additions
                    else "headerSeq" in data
                ):
                    raise ValueError(
                        f"{subject} requires headerSeq exactly when tool additions are present"
                    )
    if kind == "request/header":
        if not isinstance(data, Mapping):
            raise ValueError(f"{subject} data must be an object")
        header = data.get("header")
        if not isinstance(header, Mapping):
            raise ValueError(f"{subject} header must be an object")
        if "system" in header:
            raise ValueError(f"{subject} must omit header.system; use system/message")
        if isinstance(header.get("tools"), list) and len(header["tools"]) == 0:
            raise ValueError(f"{subject} must omit empty tools")
        defaults = header.get("adapterDefaults")
        if isinstance(defaults, Mapping) and len(defaults) == 0:
            raise ValueError(f"{subject} must omit empty adapterDefaults")
    elif kind == "tool/result":
        if not isinstance(data, Mapping):
            raise ValueError(f"{subject} data must be an object")
        if "error" not in data:
            return
        message = data.get("message")
        if not isinstance(message, Mapping) or message.get("isError") is not True:
            raise ValueError(f"{subject} error requires message.isError === true")


@dataclass(frozen=True)
class SurfaceFoldReplacement:
    """One replacement operation observed while folding a session surface."""

    seq: SessionSeq
    start: SessionSeq
    end: SessionSeq
    shadowed_seqs: list[SessionSeq]


@dataclass(frozen=True)
class SurfaceFoldResult:
    """Complete result of replaying the surface operations in a session log."""

    nodes: list[SessionSeq]
    replacements: list[SurfaceFoldReplacement]
    projected_messages: Mapping[SessionSeq, Message]


@dataclass
class _FoldState:
    nodes: list[SessionSeq] = field(default_factory=list)
    replace_generation: int = 0
    content_generation: int = 0
    projected_messages: dict[SessionSeq, Message] = field(default_factory=dict)
    projections: list[SessionMessageProjection] = field(default_factory=list)


@dataclass(frozen=True)
class _Append:
    seq: SessionSeq


@dataclass(frozen=True)
class _Replace:
    seq: SessionSeq
    start: SessionSeq
    end: SessionSeq
    start_idx: int
    end_idx: int
    shadowed_seqs: list[SessionSeq]


@dataclass(frozen=True)
class _Project:
    projection: SessionMessageProjection
    messages: Mapping[SessionSeq, Message]


_Plan = _Append | _Replace | _Project | None


def _is_replace_op(op: Mapping[str, Any]) -> bool:
    return (
        len(op) == 3
        and set(op) == {"op", "startSeq", "endSeq"}
        and op["op"] == "replace"
        and _is_event_seq(op["startSeq"])
        and _is_event_seq(op["endSeq"])
    )


def _surface_op_of(event: SessionEvent) -> SurfaceOp | None:
    kind = event["type"]
    if not is_surface_eligible_type(kind):
        # Unknown ignorable records retain opaque metadata without affecting history.
        if kind not in KNOWN_SESSION_EVENT_TYPES and event.get("ignorable") is True:
            return None
        if "surfaceOp" in event:
            raise ValueError(
                f'session event "{kind}" is not surface-eligible and cannot carry surfaceOp'
            )
        if "sourceEventSeqs" in event:
            raise ValueError(
                f'session event "{kind}" is not surface-eligible and cannot carry sourceEventSeqs'
            )
        return None
    if "surfaceOp" not in event:
        raise ValueError(
            f'session event "{kind}" is surface-eligible and requires a surfaceOp marker'
        )
    op = event["surfaceOp"]
    if op == "append":
        return op
    if not isinstance(op, Mapping):
        raise ValueError(f'session event "{kind}" carries an invalid surfaceOp')
    if not _is_replace_op(op):
        raise ValueError(f'session event "{kind}" carries an invalid replace surfaceOp')
    return op


def _assert_source_event_references(
    event: SessionEvent, shadowed_seqs: Sequence[SessionSeq]
) -> None:
    present = "sourceEventSeqs" in event
    raw = event.get("sourceEventSeqs")
    if event["type"] == "assistant/message" and present:
        raise ValueError(
            "assistant/message embeds its source stream and cannot carry sourceEventSeqs"
        )
    sources: set[int] = set()
    if present:
        if not isinstance(raw, list):
            raise ValueError(
                f"sourceEventSeqs on event at seq {event['seq']} must be an array when present"
            )
        if len(raw) == 0:
            raise ValueError("sourceEventSeqs must not be empty")
        non_earlier: int | None = None
        for source in raw:
            if not _is_event_seq(source):
                raise ValueError(
                    f'session event "{event["type"]}" sourceEventSeqs must densely contain '
                    "non-negative safe integers"
                )
            sources.add(source)
            if non_earlier is None and source >= event["seq"]:
                non_earlier = source
        if len(sources) != len(raw):
            raise ValueError("sourceEventSeqs must not contain duplicates")
        if non_earlier is not None:
            raise ValueError(
                f"sourceEventSeqs must reference earlier events: {non_earlier} >= current seq "
                f"{event['seq']}"
            )
    missing = [seq for seq in shadowed_seqs if seq not in sources]
    if missing:
        raise ValueError(
            "surface replace: sourceEventSeqs must include every shadowed surface node; "
            f"missing {', '.join(str(seq) for seq in missing)}"
        )


def _assert_developer_header(
    event: SessionEvent, events: Sequence[SessionEvent], base_seq: int
) -> None:
    if event["type"] != "developer/message":
        return
    validate_session_event_data(event, f"developer/message at seq {event['seq']}")
    data = event["data"]
    if "headerSeq" not in data:
        return
    header_seq = data["headerSeq"]
    index = header_seq - base_seq
    header_event = events[index] if 0 <= index < len(events) else None
    if (
        header_seq >= event["seq"]
        or header_event is None
        or header_event["type"] != "request/header"
    ):
        raise ValueError("developer/message headerSeq must reference an earlier request/header")
    tools = header_event["data"]["header"].get("tools") or []
    for block in data["message"]["content"]:
        if block["type"] != "tool-addition":
            continue
        definitions = [tool for tool in tools if tool["name"] == block["toolName"]]
        if len(definitions) != 1:
            raise ValueError(
                f'developer/message tool-addition "{block["toolName"]}" must name exactly one '
                f"tool in headerSeq {header_seq}"
            )
        definition = definitions[0]
        if not isinstance(definition.get("description"), str) or not _is_record(
            definition.get("parameters")
        ):
            raise ValueError(
                f'developer/message tool-addition "{block["toolName"]}" requires a complete '
                f"tool definition in headerSeq {header_seq}"
            )
        if "deferLoading" in definition and definition["deferLoading"] is not True:
            raise ValueError(
                "developer/message referenced tool deferLoading must be true when present"
            )


def validate_surface_metadata(event: SessionEvent) -> SurfaceOp | None:
    """Validate one event's surface metadata without checking membership in a log or surface.

    Returns the validated operation, or ``None`` for a log-only or unknown ignorable event.
    """
    op = _surface_op_of(event)
    if (
        op is not None
        and op != "append"
        and (op["startSeq"] >= event["seq"] or op["endSeq"] >= event["seq"])
    ):
        raise ValueError(
            f"surface replace at seq {event['seq']}: startSeq and endSeq must reference "
            "earlier events"
        )
    if op is not None:
        _assert_source_event_references(event, [])
    return op


def _index_of(nodes: Sequence[SessionSeq], seq: int) -> int:
    try:
        return nodes.index(SessionSeq(seq))
    except ValueError:
        return -1


def _replacement_range(
    state: _FoldState, op: Mapping[str, Any]
) -> tuple[int, int, list[SessionSeq]]:
    start_idx = _index_of(state.nodes, op["startSeq"])
    if start_idx == -1:
        raise ValueError(f"surface replace: start seq {op['startSeq']} not found in surface")
    end_idx = _index_of(state.nodes, op["endSeq"])
    if end_idx == -1:
        raise ValueError(f"surface replace: end seq {op['endSeq']} not found in surface")
    if start_idx > end_idx:
        raise ValueError(
            f"surface replace: start seq {op['startSeq']} (index {start_idx}) is after end seq "
            f"{op['endSeq']} (index {end_idx})"
        )
    return start_idx, end_idx, state.nodes[start_idx : end_idx + 1]


def _assert_tool_result_rewrite(
    event: SessionEvent,
    shadowed_seqs: Sequence[SessionSeq],
    events: Sequence[SessionEvent],
    base_seq: int,
) -> None:
    if event["type"] != "tool/result":
        return
    if len(shadowed_seqs) != 1:
        raise ValueError("tool/result surface replacement must rewrite exactly one current node")
    for original_seq in shadowed_seqs:
        index = original_seq - base_seq
        original = events[index] if 0 <= index < len(events) else None
        if original is None or original["type"] != "tool/result":
            raise ValueError("tool/result surface replacement must target a current tool/result")
        original_rest = {
            **original["data"],
            "message": {**original["data"]["message"], "content": None},
        }
        replacement_rest = {
            **event["data"],
            "message": {**event["data"]["message"], "content": None},
        }
        if not deep_equal_json(original_rest, replacement_rest):
            raise ValueError("tool/result surface replacement may change only content")


def _assert_system_head_rewrite(
    event: SessionEvent,
    state: _FoldState,
    start_idx: int,
    shadowed_seqs: Sequence[SessionSeq],
    events: Sequence[SessionEvent],
    base_seq: int,
) -> None:
    if start_idx != 0:
        return
    index = state.nodes[0] - base_seq
    head = events[index] if 0 <= index < len(events) else None
    if head is None or head["type"] != "system/message":
        return
    if event["type"] != "system/message" or len(shadowed_seqs) != 1:
        raise ValueError(
            "surface replace: node 0 holds the system prompt and may be rewritten only by a "
            "system/message over exactly that node"
        )


def _plan_surface_event(
    state: _FoldState,
    event: SessionEvent,
    expected_seq: int,
    events: Sequence[SessionEvent],
    base_seq: int,
    projections: Sequence[SessionMessageProjection],
) -> _Plan:
    if event["seq"] != expected_seq:
        raise ValueError(
            f"session event seq {event['seq']} is not contiguous; expected {expected_seq}"
        )
    surface_op = validate_surface_metadata(event)
    _assert_developer_header(event, events, base_seq)
    projection = next((item for item in projections if item.type == event["type"]), None)
    if projection is not None:
        context = SessionMessageProjectionContext(
            nodes=tuple(state.nodes),
            events=events,
            base_seq=SessionLogOffset(base_seq),
            messages=dict(state.projected_messages),
        )
        return _Project(projection, projection.project(event, context))
    if event["type"] in MESSAGE_PROJECTION_EVENT_TYPES:
        raise ValueError(
            f'session event "{event["type"]}" requires a message projection; load its owning '
            "plugin or supply its projection definition"
        )
    if surface_op is None:
        return None
    if surface_op == "append":
        return _Append(event["seq"])
    start_idx, end_idx, shadowed = _replacement_range(state, surface_op)
    _assert_source_event_references(event, shadowed)
    _assert_tool_result_rewrite(event, shadowed, events, base_seq)
    _assert_system_head_rewrite(event, state, start_idx, shadowed, events, base_seq)
    return _Replace(
        seq=event["seq"],
        start=surface_op["startSeq"],
        end=surface_op["endSeq"],
        start_idx=start_idx,
        end_idx=end_idx,
        shadowed_seqs=shadowed,
    )


def _apply_surface_plan(state: _FoldState, plan: _Plan) -> SurfaceFoldReplacement | None:
    if isinstance(plan, _Append):
        state.nodes.append(plan.seq)
    elif isinstance(plan, _Replace):
        state.nodes[plan.start_idx : plan.end_idx + 1] = [plan.seq]
        state.replace_generation += 1
        state.content_generation += 1
        return SurfaceFoldReplacement(plan.seq, plan.start, plan.end, plan.shadowed_seqs)
    elif isinstance(plan, _Project):
        state.projected_messages.update(plan.messages)
        if not any(existing is plan.projection for existing in state.projections):
            state.projections.append(plan.projection)
        state.content_generation += 1
    return None


def fold_surface(
    events: Sequence[SessionEvent], projections: Sequence[SessionMessageProjection] = ()
) -> SurfaceFoldResult:
    """Replay a complete log through the canonical surface fold.

    Raises when an interpreter is missing or an event violates its projection, surface
    metadata, source attribution or replacement rules.
    """
    state = _FoldState()
    replacements: list[SurfaceFoldReplacement] = []
    for index, event in enumerate(events):
        plan = _plan_surface_event(state, event, index, events, 0, projections)
        replacement = _apply_surface_plan(state, plan)
        if replacement is not None:
            replacements.append(replacement)
    return SurfaceFoldResult(list(state.nodes), replacements, dict(state.projected_messages))


class SurfaceManager:
    """Incremental ordered surface view and append-boundary validator.

    ``log`` is the contiguous complete log or loaded event window (read live, never
    copied); ``base_seq`` is the absolute sequence of its first event. ``projections``
    are live borrowed definitions: removing one a session used invalidates further reads.
    """

    def __init__(
        self,
        log: Sequence[SessionEvent],
        base_seq: int = 0,
        projections: Sequence[SessionMessageProjection] = (),
    ) -> None:
        self._log = log
        self._base_seq = base_seq
        self._projections = projections
        self._state = _FoldState()
        self._last_processed_seq = base_seq - 1
        self._pending: tuple[SessionEvent, int, _Plan] | None = None

    def _catch_up(self) -> None:
        self._assert_projections()
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()

    def validate_next(self, event: SessionEvent) -> None:
        """Validate the next candidate without mutating the committed surface."""
        self._catch_up()
        expected = self._base_seq + len(self._log)
        plan = _plan_surface_event(
            self._state, event, expected, self._log, self._base_seq, self._projections
        )
        self._pending = (event, expected, plan)

    @property
    def replace_generation(self) -> int:
        """Monotonic count of folded positional replacements."""
        self._catch_up()
        return self._state.replace_generation

    @property
    def content_generation(self) -> int:
        """Monotonic count of committed changes to existing model-visible content."""
        self._catch_up()
        return self._state.content_generation

    def derive_event_message(self, event: SessionEvent) -> Message | None:
        """Project one message with every committed message projection applied."""
        self._catch_up()
        return derive_event_message(event, self._state.projected_messages)

    @property
    def nodes(self) -> tuple[SessionSeq, ...]:
        """Surface event sequences in model-visible order."""
        self._catch_up()
        return tuple(self._state.nodes)

    def _process_delta(self) -> None:
        tail = self._base_seq + len(self._log) - 1
        seq = self._last_processed_seq + 1
        while seq <= tail:
            event = self._log[seq - self._base_seq]
            pending = self._pending
            if pending is not None and pending[0] is event and pending[1] == seq:
                _apply_surface_plan(self._state, pending[2])
            else:
                plan = _plan_surface_event(
                    self._state, event, seq, self._log, self._base_seq, self._projections
                )
                _apply_surface_plan(self._state, plan)
            if pending is not None and pending[1] <= seq:
                self._pending = None
            self._last_processed_seq = seq
            seq += 1

    def _assert_projections(self) -> None:
        candidate = self._pending
        pending_plan: _Plan = None
        if candidate is not None:
            index = candidate[1] - self._base_seq
            if 0 <= index < len(self._log) and self._log[index] is candidate[0]:
                pending_plan = candidate[2]
        required = list(self._state.projections)
        if isinstance(pending_plan, _Project):
            required.append(pending_plan.projection)
        for projection in required:
            if not any(item is projection for item in self._projections):
                raise ValueError(
                    f'session message projection "{projection.type}" was removed or replaced; '
                    "restore the session with its owning plugin"
                )
