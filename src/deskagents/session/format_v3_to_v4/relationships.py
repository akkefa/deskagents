"""Upstream: packages/session/session-format-v3-to-v4/src/relationships.ts

Mandatory native V4 relationships; incomplete tails retain their open transactions.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from deskagents.session.format import (
    SessionFormatArtifact,
    SessionFormatError,
    SessionFormatEvent,
    session_format_count,
)
from deskagents.util.values import is_safe_integer

from ._js import (
    UNDEFINED,
    at,
    deep_strict_equal,
    greater_or_equal,
    has,
    includes,
    is_array,
    js_string,
    opt_prop,
    prop,
    strict_equal,
)

_SURFACE_TYPES = frozenset(
    {"system/message", "user/message", "developer/message", "assistant/message", "tool/result"}
)
_STEP_EVENT_TYPES = frozenset({"system/message", "developer/message", "assistant/attempt"})
_RELATIONSHIP_TYPES = frozenset(
    {
        *_SURFACE_TYPES,
        *_STEP_EVENT_TYPES,
        "turn/start",
        "turn/end",
        "step/start",
        "step/end",
        "tool/call",
        "request/header",
        "request/context",
        "tool/ptc-dispatch-start",
        "tool/ptc-dispatch",
        "llm/retry",
        "llm/retry-started",
        "session/title",
        "session/title-llm-request",
        "command/run",
        "command/done",
        "compaction/start",
        "compaction/summary",
        "compaction/end",
        "compaction/prune",
        "session/end-seed",
    }
)
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)")
_INTERRUPTED_TEXT = (
    "The tool call was interrupted before the Harness recorded it as started. "
    "Retry it if it is still needed."
)

_Record = Mapping[str, object]


def _record(value: object, subject: str) -> _Record:
    if not isinstance(value, Mapping):
        raise SessionFormatError(f"{subject} requires an object")
    return value


def _text(value: object, subject: str) -> str:
    if not isinstance(value, str) or value == "":
        raise SessionFormatError(f"{subject} requires a nonempty string")
    return value


def _array(value: object, subject: str) -> Sequence[object]:
    if not is_array(value):
        raise SessionFormatError(f"{subject} requires an array")
    assert isinstance(value, list | tuple)
    return value


def _earlier(value: object, seq: object, subject: str) -> int:
    coordinate = session_format_count(value, subject)
    if greater_or_equal(coordinate, seq):
        raise SessionFormatError(f"{subject} must name an earlier event")
    return coordinate


def _index_of(values: Sequence[object], value: object) -> int:
    return next((index for index, item in enumerate(values) if strict_equal(item, value)), -1)


@dataclass
class _ToolState:
    name: object
    arguments: object
    started: bool


@dataclass
class _Compaction:
    id: str
    command: object
    turn: int | None
    seq: object
    summarized: bool


@dataclass
class _Dispatch:
    data: _Record
    settled: bool


class _Relationships:
    """One native artifact's lifecycle state; no state survives between reads."""

    def __init__(self, artifact: SessionFormatArtifact, known_event_types: Collection[str]) -> None:
        self.artifact = artifact
        self.known = known_event_types
        self.turn: int | None = None
        self.step: int | None = None
        self.next_turn = 1
        self.next_step = 1
        self.provider: object = UNDEFINED
        self.surface: list[object] = []
        self.protected_head: object = UNDEFINED
        self.compaction: _Compaction | None = None
        self.tools: dict[str, _ToolState] = {}
        self.dispatches: dict[str, _Dispatch] = {}
        self.retries: list[_Record] = []
        self.started_retries: set[tuple[str, int]] = set()
        self.commands: set[str] = set()
        self.orphan_compactions: list[object] = []
        start: object = UNDEFINED
        for event in artifact.events:
            event_type = prop(event, "type")
            if not has(self.known, event_type):
                continue
            if event_type == "compaction/start":
                start = prop(event, "seq")
            if event_type == "compaction/end":
                start = UNDEFINED
            if event_type == "session/end-seed":
                if start is not UNDEFINED:
                    self.orphan_compactions.append(start)
                start = UNDEFINED

    def require_turn(self, event_type: str) -> None:
        if self.turn is None:
            raise SessionFormatError(f"{event_type} is outside an open turn")

    def require_step(self, event: SessionFormatEvent, data: _Record) -> None:
        if (
            self.turn is None
            or self.step is None
            or not strict_equal(prop(data, "turn"), self.turn)
            or not strict_equal(prop(data, "step"), self.step)
        ):
            raise SessionFormatError(
                f"{js_string(prop(event, 'type'))} does not match an open turn and step"
            )

    def close_tools(self, event_type: str) -> None:
        if self.tools:
            first = next(iter(self.tools))
            raise SessionFormatError(f"{event_type} leaves unresolved tool call {first}")
        self.tools.clear()

    def developer(self, event: SessionFormatEvent, data: _Record) -> None:
        if "headerSeq" not in data:
            return
        seq = prop(event, "seq")
        header_seq = _earlier(data["headerSeq"], seq, "developer/message headerSeq")
        header_event = at(self.artifact.events, header_seq)
        if opt_prop(header_event, "type") != "request/header":
            raise SessionFormatError(
                "developer/message headerSeq must reference an earlier request/header"
            )
        if not has(self.known, prop(header_event, "type")):
            raise SessionFormatError(
                "developer/message headerSeq must reference a known request/header"
            )
        request = _record(prop(header_event, "data"), "request/header data")
        tools = prop(_record(prop(request, "header"), "request header"), "tools")
        content = _array(
            prop(_record(prop(data, "message"), "developer message"), "content"),
            "developer content",
        )
        for value in content:
            if not isinstance(value, Mapping) or prop(value, "type") != "tool-addition":
                continue
            tool_name = prop(value, "toolName")
            definitions = (
                [
                    tool
                    for tool in tools
                    if isinstance(tool, Mapping) and strict_equal(prop(tool, "name"), tool_name)
                ]
                if isinstance(tools, list | tuple)
                else []
            )
            name = js_string(tool_name)
            if len(definitions) != 1:
                raise SessionFormatError(
                    f'developer/message tool-addition "{name}" must name exactly one tool '
                    f"in headerSeq {header_seq}"
                )
            definition = definitions[0]
            if not isinstance(prop(definition, "description"), str) or not isinstance(
                prop(definition, "parameters"), Mapping
            ):
                raise SessionFormatError(
                    f'developer/message tool-addition "{name}" requires a complete tool '
                    f"definition in headerSeq {header_seq}"
                )

    def fold_surface(self, event: SessionFormatEvent) -> None:
        event_type = prop(event, "type")
        if not has(_SURFACE_TYPES, event_type):
            return
        seq = prop(event, "seq")
        if event_type == "system/message" and self.surface and self.protected_head is UNDEFINED:
            raise SessionFormatError("system/message requires a protected first surface head")
        if prop(event, "surfaceOp") == "append":
            if event_type == "system/message" and not self.surface:
                self.protected_head = seq
            self.surface.append(seq)
            return
        operation = _record(prop(event, "surfaceOp"), f"{event_type} surfaceOp")
        first = _index_of(
            self.surface, _earlier(prop(operation, "startSeq"), seq, "replacement start")
        )
        last = _index_of(self.surface, _earlier(prop(operation, "endSeq"), seq, "replacement end"))
        if first < 0 or last < first:
            raise SessionFormatError(
                f"{event_type} replacement range is not on the current surface"
            )
        removed = self.surface[first : last + 1]
        sources = _array(prop(event, "sourceEventSeqs"), "replacement sourceEventSeqs")
        if any(not includes(sources, item) for item in removed):
            raise SessionFormatError("replacement sourceEventSeqs omit a shadowed surface node")
        if self.protected_head is not UNDEFINED and includes(removed, self.protected_head):
            if event_type != "system/message" or len(removed) != 1:
                raise SessionFormatError(
                    "surface replacement cannot shadow the protected system head"
                )
            self.protected_head = seq
        self.surface[first : first + len(removed)] = [seq]

    def tool(self, event: SessionFormatEvent, data: _Record) -> None:
        event_type = prop(event, "type")
        if event_type == "tool/result" and prop(event, "surfaceOp") != "append":
            self.require_turn(event_type)
            return
        self.require_step(event, data)
        if event_type == "assistant/message":
            assistant = _record(prop(data, "message"), "assistant message")
            for value in _array(prop(assistant, "content"), "assistant content"):
                block = _record(value, "assistant content block")
                if prop(block, "type") != "tool-call":
                    continue
                call_id = _text(prop(block, "id"), "tool call id")
                if call_id in self.tools:
                    raise SessionFormatError(
                        f"assistant/message repeats advertised tool call {call_id}"
                    )
                self.tools[call_id] = _ToolState(
                    prop(block, "name"), prop(block, "arguments"), started=False
                )
            return
        message: _Record | None = (
            _record(prop(data, "message"), "tool result message")
            if event_type == "tool/result"
            else None
        )
        call_id = _text(
            prop(data, "callId") if message is None else prop(message, "toolCallId"),
            "tool call id",
        )
        pending = self.tools.get(call_id)
        if pending is None:
            raise SessionFormatError(
                f"{js_string(event_type)} {call_id} has no advertised tool lifecycle"
            )
        if message is None:
            if (
                pending.started
                or not strict_equal(pending.name, prop(data, "name"))
                or not strict_equal(pending.arguments, prop(data, "arguments"))
            ):
                raise SessionFormatError(
                    f"tool/call {call_id} does not match one advertised tool call"
                )
            pending.started = True
        else:
            if not pending.started and not _not_started_repair(event, data, message, call_id):
                raise SessionFormatError(
                    f"tool/result {call_id} is not the exact TOOL_NOT_STARTED repair"
                )
            del self.tools[call_id]

    def dispatch(self, event: SessionFormatEvent, data: _Record) -> None:
        event_type = js_string(prop(event, "type"))
        self.require_turn(event_type)
        call_id = _text(prop(data, "subCallId"), "PTC subCallId")
        root = _text(prop(data, "rootCallId"), "PTC rootCallId")
        parent = _text(prop(data, "parentCallId"), "PTC parentCallId")
        existing = self.dispatches.get(call_id)
        if existing is not None and not strict_equal(prop(existing.data, "rootCallId"), root):
            raise SessionFormatError("PTC dispatch changes its rootCallId")
        parent_dispatch = self.dispatches.get(parent)
        parent_root = (
            UNDEFINED if parent_dispatch is None else prop(parent_dispatch.data, "rootCallId")
        )
        if parent != root and not strict_equal(parent_root, root):
            raise SessionFormatError("PTC parentCallId does not belong to rootCallId")
        if event_type == "tool/ptc-dispatch-start":
            if existing is not None:
                raise SessionFormatError("PTC dispatch repeats subCallId")
            self.dispatches[call_id] = _Dispatch(data, settled=False)
            return
        if existing is None or existing.settled:
            raise SessionFormatError("PTC dispatch has no unique start")
        if any(
            not deep_strict_equal(prop(existing.data, key), prop(data, key))
            for key in ("rootCallId", "parentCallId", "name", "arguments")
        ):
            raise SessionFormatError("PTC dispatch does not match its start")
        existing.settled = True

    def retry(self, event: SessionFormatEvent, data: _Record) -> None:
        retry_id = _text(prop(data, "retryId"), "retryId")
        attempt = session_format_count(prop(data, "retry"), "retry")
        if prop(event, "type") == "llm/retry-started":
            scheduled = next(
                (
                    item
                    for item in self.retries
                    if strict_equal(prop(item, "retryId"), retry_id)
                    and strict_equal(prop(item, "retry"), attempt)
                ),
                None,
            )
            if scheduled is None:
                raise SessionFormatError("llm/retry-started pairs no prior scheduled attempt")
            if not strict_equal(prop(scheduled, "turn"), prop(data, "turn")) or not strict_equal(
                prop(scheduled, "step"), prop(data, "step")
            ):
                raise SessionFormatError("llm/retry-started changes scheduled coordinates")
            key = (retry_id, attempt)
            if key in self.started_retries:
                raise SessionFormatError("llm/retry-started repeats one scheduled attempt")
            self.started_retries.add(key)
            return
        current_step = self.step if self.step is not None else self.next_step - 1
        if (
            self.turn is None
            or not strict_equal(prop(data, "turn"), self.turn)
            or not strict_equal(prop(data, "step"), current_step)
        ):
            raise SessionFormatError("llm/retry does not match the current turn and step")
        if not strict_equal(prop(data, "provider"), self.provider):
            raise SessionFormatError("llm/retry provider does not match the open request/header")
        prior = next(
            (
                item
                for item in reversed(self.retries)
                if all(
                    strict_equal(prop(item, key), prop(data, key))
                    for key in ("turn", "step", "provider", "policyKey")
                )
            ),
            None,
        )
        expected = (
            1 if prior is None else session_format_count(prop(prior, "retry"), "prior retry") + 1
        )
        if attempt != expected:
            raise SessionFormatError("llm/retry skips its policy attempt sequence")
        if (
            any(strict_equal(prop(item, "retryId"), retry_id) for item in self.retries)
            if prior is None
            else not strict_equal(prop(prior, "retryId"), retry_id)
        ):
            raise SessionFormatError("llm/retry must keep one retryId per policy chain")
        self.retries.append(data)

    def compact(self, event: SessionFormatEvent, data: _Record) -> None:
        event_type = js_string(prop(event, "type"))
        if event_type in ("compaction/prune", "compaction/summary"):
            self.span(event, data)
        if event_type == "compaction/prune":
            return
        if event_type == "compaction/start":
            if self.compaction is not None:
                raise SessionFormatError("compaction/start overlaps an open compaction")
            turn = prop(data, "turn")
            owner = None if turn is None else session_format_count(turn, "compaction turn")
            if owner != self.turn:
                raise SessionFormatError("compaction/start does not match the open turn")
            self.compaction = _Compaction(
                id=_text(prop(data, "compactionId"), "compactionId"),
                command=prop(data, "sourceCommandId"),
                turn=owner,
                seq=prop(event, "seq"),
                summarized=False,
            )
            return
        current = self.compaction_owner(data, event_type)
        if current.turn != self.turn:
            raise SessionFormatError(f"{event_type} does not match the open turn")
        if event_type == "compaction/summary":
            if current.summarized:
                raise SessionFormatError("compaction/summary repeats")
            current.summarized = True
        else:
            if not strict_equal(prop(data, "turn"), current.turn):
                raise SessionFormatError("compaction/end changes its owner turn")
            if "error" not in data and not current.summarized:
                raise SessionFormatError("successful compaction/end requires one summary")
            self.compaction = None

    def compaction_owner(self, data: _Record, subject: str) -> _Compaction:
        if (
            self.compaction is None
            or not strict_equal(self.compaction.id, prop(data, "compactionId"))
            or not strict_equal(self.compaction.command, prop(data, "sourceCommandId"))
        ):
            raise SessionFormatError(f"{subject} has no matching compaction/start")
        return self.compaction

    def span(self, event: SessionFormatEvent, data: _Record) -> None:
        seq = prop(event, "seq")
        shadowed = _record(prop(data, "shadowedRange"), "compaction shadowedRange")
        start = _index_of(
            self.surface, _earlier(prop(shadowed, "start"), seq, "compaction range start")
        )
        end = _index_of(self.surface, _earlier(prop(shadowed, "end"), seq, "compaction range end"))
        seqs = _array(prop(data, "shadowedSeqs"), "compaction shadowedSeqs")
        if start < 0 or end < start or not deep_strict_equal(self.surface[start : end + 1], seqs):
            raise SessionFormatError(
                f"{js_string(prop(event, 'type'))} shadowedSeqs do not name an exact current "
                "surface span"
            )
        if self.protected_head is not UNDEFINED and includes(seqs, self.protected_head):
            raise SessionFormatError("compaction cannot shadow the protected system head")

    def accept(self, event: SessionFormatEvent) -> None:
        event_type = prop(event, "type")
        if not has(self.known, event_type) or not has(_RELATIONSHIP_TYPES, event_type):
            return
        assert isinstance(event_type, str)
        data = _record(prop(event, "data"), event_type)
        if event_type in _STEP_EVENT_TYPES:
            self.require_step(event, data)
        if (
            event_type.startswith("turn/")
            and self.compaction is not None
            and not includes(self.orphan_compactions, self.compaction.seq)
        ):
            raise SessionFormatError(f"{event_type} crosses an open compaction")
        self.fold_surface(event)
        seq = prop(event, "seq")
        if event_type == "turn/start":
            if self.turn is not None or not strict_equal(prop(data, "turn"), self.next_turn):
                raise SessionFormatError("turn/start does not open the expected turn")
            self.turn = self.next_turn
            self.next_step = 1
            self.tools.clear()
        elif event_type == "turn/end":
            if (
                self.turn is None
                or not strict_equal(prop(data, "turn"), self.turn)
                or self.step is not None
            ):
                raise SessionFormatError("turn/end does not match the open turn with no open step")
            self.close_tools(event_type)
            self.turn = None
            self.next_turn += 1
        elif event_type == "step/start":
            if (
                self.turn is None
                or not strict_equal(prop(data, "turn"), self.turn)
                or self.step is not None
                or not strict_equal(prop(data, "step"), self.next_step)
            ):
                raise SessionFormatError("step/start does not match the open turn and next step")
            self.step = self.next_step
        elif event_type == "step/end":
            self.require_step(event, data)
            self.close_tools(event_type)
            self.step = None
            self.next_step += 1
        elif event_type in ("assistant/message", "tool/call", "tool/result"):
            self.tool(event, data)
        elif event_type == "developer/message":
            self.developer(event, data)
        elif event_type == "request/header":
            self.require_turn(event_type)
            config = _record(
                prop(_record(prop(data, "header"), "request header"), "config"), "request config"
            )
            self.provider = prop(config, "provider")
        elif event_type == "request/context":
            self.require_turn(event_type)
        elif event_type in ("tool/ptc-dispatch-start", "tool/ptc-dispatch"):
            self.dispatch(event, data)
        elif event_type in ("llm/retry", "llm/retry-started"):
            self.retry(event, data)
        elif event_type in ("session/title", "session/title-llm-request"):
            _title_sources(self.artifact.events, event, data, self.known)
        elif event_type == "command/run":
            command_id = _text(prop(data, "commandId"), "commandId")
            if command_id in self.commands:
                raise SessionFormatError(f"command/run repeats commandId {command_id}")
            self.commands.add(command_id)
        elif event_type == "command/done":
            if _text(prop(data, "commandId"), "commandId") not in self.commands:
                raise SessionFormatError("command/done has no prior command/run")
            if "sourceEventSeq" in data:
                source = at(
                    self.artifact.events,
                    _earlier(data["sourceEventSeq"], seq, "command sourceEventSeq"),
                )
                source_type = opt_prop(source, "type")
                if prop(data, "kind") != "success" or source_type in (
                    "command/run",
                    "command/done",
                ):
                    raise SessionFormatError("command/done has invalid sourceEventSeq")
        elif event_type in (
            "compaction/start",
            "compaction/summary",
            "compaction/end",
            "compaction/prune",
        ):
            self.compact(event, data)
        elif event_type == "user/message":
            source_record = _record(prop(data, "source"), "user message source")
            if (
                prop(event, "surfaceOp") != "append"
                and prop(source_record, "kind") == "compact-checkpoint"
            ):
                self.compaction_owner(source_record, "compaction checkpoint")
        elif event_type == "session/end-seed":
            self.compaction = None


def _not_started_repair(
    event: SessionFormatEvent, data: _Record, message: _Record, call_id: str
) -> bool:
    error = _record(prop(data, "error"), "not-started error")
    if (
        prop(error, "name") != "ToolNotStartedError"
        or prop(error, "code") != "TOOL_NOT_STARTED"
        or prop(message, "isError") is not True
        or "sourceEventSeqs" in event
    ):
        return False
    message_id = _text(prop(message, "id"), "not-started message id")
    # Fork-specific fields and identity are checked by assert_v4_fork_result before relationships.
    if message_id.startswith(f"forked-tool-result-{call_id}-"):
        return True
    prefix = f"interrupted-tool-result-{call_id}-"
    suffix = message_id[len(prefix) :]
    content = _array(prop(message, "content"), "not-started content")
    block = at(content, 0)
    return (
        message_id.startswith(prefix)
        and _DECIMAL.fullmatch(suffix) is not None
        and is_safe_integer(int(suffix))
        and len(content) == 1
        and isinstance(block, Mapping)
        and prop(block, "type") == "text"
        and prop(block, "text") == _INTERRUPTED_TEXT
    )


def _title_sources(
    events: Sequence[SessionFormatEvent],
    event: SessionFormatEvent,
    data: _Record,
    known_event_types: Collection[str],
) -> None:
    event_type = prop(event, "type")
    seq = prop(event, "seq")
    references = _array(prop(data, "messageSeqs"), "title messageSeqs")
    if event_type == "session/title" and (len(references) == 0) != (
        prop(_record(prop(data, "source"), "title source"), "kind") == "user"
    ):
        raise SessionFormatError("session/title messageSeqs must be empty exactly for a user title")
    seen: set[int] = set()
    for value in references:
        reference = _earlier(value, seq, "title messageSeqs")
        source = at(events, reference)
        if (
            reference in seen
            or opt_prop(source, "type") != "user/message"
            or not has(known_event_types, prop(source, "type"))
            or prop(
                _record(
                    prop(_record(prop(source, "data"), "title input"), "source"),
                    "title input source",
                ),
                "kind",
            )
            != "user"
        ):
            raise SessionFormatError(
                f"{js_string(event_type)} messageSeqs must cite distinct earlier human "
                "user/message events"
            )
        seen.add(reference)
    if event_type != "session/title-llm-request":
        return
    messages = _array(prop(data, "messages"), "title messages")
    message = _record(at(messages, 0), "title message")
    content = _array(prop(message, "content"), "title content")
    block = at(content, 0)
    if (
        len(references) == 0
        or len(messages) != 1
        or prop(message, "role") != "user"
        or prop(_record(prop(message, "source"), "title source"), "kind") != "dsh-session-title-llm"
        or len(content) != 1
        or not isinstance(block, Mapping)
        or prop(block, "type") != "text"
    ):
        raise SessionFormatError("session/title-llm-request messages do not represent messageSeqs")


def assert_v4_lifecycle_relationships(
    artifact: SessionFormatArtifact, known_event_types: Collection[str]
) -> None:
    """Validate native V4 lifecycle and ownership facts without rewriting any event.

    ``artifact`` has had its V4 envelopes and messages admitted; ``known_event_types``
    names the installed event types whose payloads this reader interprets.
    """
    state = _Relationships(artifact, known_event_types)
    for event in artifact.events:
        state.accept(event)
