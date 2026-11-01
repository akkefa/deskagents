"""Ported tool-update projection cases from packages/llm/llm/tests/content.spec.ts."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from deskagents.llm import (
    ContentBlock,
    Message,
    ToolSchema,
    create_developer_message,
    create_user_message,
    project_tool_updates,
)

SPEC = "packages/llm/llm/tests/content.spec.ts"
SEARCH: ToolSchema = {"name": "search", "description": "Search", "parameters": {}}
FETCH: ToolSchema = {"name": "fetch", "description": "Fetch", "parameters": {}}
SOURCE: Mapping[str, Any] = {"kind": "test"}
PROMPT = create_user_message(source=SOURCE, content=[{"type": "text", "text": "hi"}])


def developer(content: Sequence[ContentBlock]) -> Message:
    return create_developer_message(source=SOURCE, content=content)


def add(name: str) -> ContentBlock:
    return {"type": "tool-addition", "toolName": name}


def remove(name: str) -> ContentBlock:
    return {"type": "tool-removal", "toolName": name}


def roles(messages: Sequence[Message]) -> list[Sequence[ContentBlock]]:
    return [message["content"] for message in messages if message["role"] == "developer"]


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > returns the input history and tools by identity when nothing "
    "is projected"
)
def test_returns_inputs_by_identity_when_nothing_is_projected() -> None:
    messages, tools = [PROMPT], [SEARCH]
    projected = project_tool_updates(messages, tools, "in-history", {"tools": tools, "updates": []})
    assert projected["messages"] is messages
    assert projected["tools"] == tools
    assert project_tool_updates(messages, tools, None)["messages"] is messages


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > strips deferred loading and developer updates when the route "
    "declares no mode"
)
def test_strips_deferred_loading_and_developer_updates_without_mode() -> None:
    added = developer([add("fetch")])
    history = {"tools": [SEARCH], "updates": [{"messageId": added["id"], "additions": [FETCH]}]}
    deferred = [SEARCH, {**FETCH, "deferLoading": True}]
    projected = project_tool_updates([PROMPT, added, PROMPT], deferred, None, history)
    assert projected == {"tools": [SEARCH, FETCH], "messages": [PROMPT, PROMPT]}
    plain = [SEARCH, FETCH]
    assert project_tool_updates([PROMPT], plain, None, history)["tools"] is plain


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > uses current declarations without updates when history is "
    "missing or the prefix omits an update"
)
def test_uses_current_declarations_for_missing_or_incomplete_history() -> None:
    added = developer([add("fetch")])
    history = {"tools": [SEARCH], "updates": [{"messageId": added["id"], "additions": [FETCH]}]}
    tools = [SEARCH, FETCH]
    assert project_tool_updates([PROMPT, added], tools, "in-history") == {
        "tools": tools,
        "messages": [PROMPT],
    }
    prefix = project_tool_updates([PROMPT], tools, "in-history", history)
    assert prefix["tools"] is tools
    assert prefix["messages"] == [PROMPT]


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > omits updates from an earlier declaration series"
)
def test_omits_updates_from_an_earlier_declaration_series() -> None:
    earlier, current = developer([add("search")]), developer([add("fetch")])
    history = {"tools": [SEARCH], "updates": [{"messageId": current["id"], "additions": [FETCH]}]}
    projected = project_tool_updates(
        [PROMPT, earlier, current], [SEARCH, FETCH], "in-history", history
    )
    assert projected["messages"] == [PROMPT, current]
    assert projected["tools"] == [SEARCH, {**FETCH, "deferLoading": True}]


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > defers historical additions and retains removed definitions "
    "for in-history routes"
)
def test_defers_additions_and_retains_removed_definitions_in_history() -> None:
    added = developer([add("fetch"), {"type": "text", "text": "fetch is available"}])
    removed = developer([remove("search")])
    history: dict[str, Any] = {
        "tools": [SEARCH],
        "updates": [
            {"messageId": added["id"], "additions": [FETCH]},
            {"messageId": removed["id"], "additions": []},
        ],
    }
    projected = project_tool_updates(
        [PROMPT, added, PROMPT, removed, PROMPT], [FETCH], "in-history", history
    )
    assert projected["tools"] == [SEARCH, {**FETCH, "deferLoading": True}]
    assert projected["messages"][1] is added
    assert roles(projected["messages"]) == [added["content"], removed["content"]]


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > omits removed definitions and removal blocks for addition-only "
    "routes"
)
def test_omits_removed_definitions_and_blocks_for_addition_only() -> None:
    swapped = developer([add("fetch"), remove("search")])
    removed = developer([remove("fetch")])
    history: dict[str, Any] = {
        "tools": [SEARCH],
        "updates": [
            {"messageId": swapped["id"], "additions": [FETCH]},
            {"messageId": removed["id"], "additions": []},
        ],
    }
    projected = project_tool_updates(
        [PROMPT, swapped, PROMPT, removed, PROMPT], [], "addition-only", history
    )
    assert projected == {"tools": [], "messages": [PROMPT, PROMPT, PROMPT]}
    partial = project_tool_updates(
        [PROMPT, swapped, PROMPT],
        [FETCH],
        "addition-only",
        {"tools": [SEARCH], "updates": [history["updates"][0]]},
    )
    assert partial["tools"] == [{**FETCH, "deferLoading": True}]
    assert roles(partial["messages"]) == [[add("fetch")]]


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > re-offers an unchanged restored tool through its recorded blocks"
)
def test_reoffers_an_unchanged_restored_tool() -> None:
    removed, restored = developer([remove("search")]), developer([add("search")])
    history = {
        "tools": [SEARCH, FETCH],
        "updates": [
            {"messageId": removed["id"], "additions": []},
            {"messageId": restored["id"], "additions": [SEARCH]},
        ],
    }
    messages = [PROMPT, removed, PROMPT, restored, PROMPT]
    projected = project_tool_updates(messages, [SEARCH, FETCH], "in-history", history)
    assert roles(projected["messages"]) == [
        [remove("search")],
        [add("search")],
    ]
    assert (
        roles(project_tool_updates(messages, [SEARCH, FETCH], "addition-only", history)["messages"])
        == []
    )


@pytest.mark.upstream(
    f"{SPEC}::projectToolUpdates > declares a deferred tool once across removal and re-addition"
)
def test_declares_a_deferred_tool_once_across_removal_and_readdition() -> None:
    added, removed, restored = (
        developer([add("fetch")]),
        developer([remove("fetch")]),
        developer([add("fetch")]),
    )
    history = {
        "tools": [SEARCH],
        "updates": [
            {"messageId": added["id"], "additions": [FETCH]},
            {"messageId": removed["id"], "additions": []},
            {"messageId": restored["id"], "additions": [FETCH]},
        ],
    }
    messages = [PROMPT, added, PROMPT, removed, PROMPT, restored, PROMPT]
    in_history = project_tool_updates(messages, [SEARCH, FETCH], "in-history", history)
    assert in_history["tools"] == [SEARCH, {**FETCH, "deferLoading": True}]
    assert len(roles(in_history["messages"])) == 3
    addition_only = project_tool_updates(messages, [SEARCH, FETCH], "addition-only", history)
    assert roles(addition_only["messages"]) == [[add("fetch")]]
