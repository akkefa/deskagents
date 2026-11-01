"""Ported from packages/core/session/tests/properties.spec.ts.

Upstream uses fast-check. Python has no property-testing dependency here, so each
property runs over ``RUNS`` logs drawn from a seeded :class:`random.Random` with the
same shapes and size bounds as upstream's arbitraries (no shrinking). The seed is
fixed per test so failures reproduce.
"""

from __future__ import annotations

import copy
import itertools
import random
import string
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from deskagents.core.session import Session, session_id
from deskagents.llm import (
    ToolCallId,
    create_message,
    create_tool_result_message,
    create_user_message,
)
from deskagents.util.values import FrozenJsonObject

SPEC = "packages/core/session/tests/properties.spec.ts"
GROUP = "Session properties"
RUNS = 100
"""fast-check's default ``numRuns``."""


@dataclass(frozen=True)
class Appendable:
    """One generated event; it supplies its own surface intent (``None`` for none)."""

    type: str
    data: Any
    surface_op: str | None = None


Arbitrary = Callable[[random.Random], Appendable]

# fast-check's default ``fc.string()`` draws printable ASCII, up to ten characters.
_CHARS = string.printable[:95]


def gen_string(rng: random.Random, min_length: int = 0) -> str:
    return "".join(rng.choice(_CHARS) for _ in range(rng.randint(min_length, 10)))


def gen_text_content(rng: random.Random) -> list[dict[str, str]]:
    return [{"type": "text", "text": gen_string(rng)} for _ in range(rng.randint(0, 3))]


def _assistant(rng: random.Random, usage: bool) -> Appendable:
    data: dict[str, Any] = {
        "turn": 1,
        "step": 1,
        "stream": [],
        "message": create_message(
            role="assistant",
            content=gen_text_content(rng),
            source={"kind": "model", "provider": "mock", "model": "mock"},
        ),
    }
    if usage:
        data["usage"] = {"inputTokens": 1, "outputTokens": 1}
    return Appendable("assistant/message", data, "append")


def _user(rng: random.Random) -> Appendable:
    message = create_user_message(content=gen_text_content(rng), source={"kind": "user"})
    return Appendable("user/message", message, "append")


def _tool_result(rng: random.Random) -> Appendable:
    call_id, content, is_error = gen_string(rng, 1), gen_text_content(rng), rng.random() < 0.5
    message = create_tool_result_message(
        call_id=ToolCallId(call_id), content=content, is_error=is_error
    )
    return Appendable("tool/result", {"turn": 1, "step": 1, "message": message}, "append")


# A message-producing event (these DO affect derived history). Each carries an
# explicit ``surface_op="append"`` intent — the marker the real loop passes.
MESSAGE_ARBITRARIES: list[Arbitrary] = [
    _user,
    lambda rng: _assistant(rng, usage=False),
    lambda rng: _assistant(rng, usage=True),
    _tool_result,
]

# A non-message event (trace/replay data — must NOT affect derived history).
NON_MESSAGE_ARBITRARIES: list[Arbitrary] = [
    lambda rng: Appendable("turn/start", {"turn": 1}),
    lambda rng: Appendable("turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    lambda rng: Appendable("step/start", {"turn": 1, "step": 1}),
    lambda rng: Appendable("step/end", {"turn": 1, "step": 1}),
    lambda rng: Appendable(
        "assistant/attempt",
        {
            "turn": 1,
            "step": 1,
            "stream": [
                {
                    "type": "text-chunks",
                    "time0": 1,
                    "index": 0,
                    "dt": [],
                    "texts": [gen_string(rng)],
                }
            ],
        },
    ),
]


def gen_message_event(rng: random.Random) -> Appendable:
    return rng.choice(MESSAGE_ARBITRARIES)(rng)


def gen_non_message_event(rng: random.Random) -> Appendable:
    return rng.choice(NON_MESSAGE_ARBITRARIES)(rng)


def gen_any_event(rng: random.Random) -> Appendable:
    return (gen_message_event if rng.random() < 0.5 else gen_non_message_event)(rng)


def gen_array(rng: random.Random, item: Arbitrary, max_length: int) -> list[Appendable]:
    return [item(rng) for _ in range(rng.randint(0, max_length))]


def gen_log(rng: random.Random) -> list[Appendable]:
    return gen_array(rng, gen_any_event, 25)


def logs(seed: int) -> Iterator[list[Appendable]]:
    rng = random.Random(seed)
    for _ in range(RUNS):
        yield gen_log(rng)


_counter = itertools.count()


def build(events: list[Appendable]) -> Session:
    session = Session.create(session_id(f"prop-{next(_counter)}"))
    for event in events:
        # Forward the generated intent verbatim; non-surface events carry none.
        if event.surface_op is not None:
            session.append(event.type, event.data, surface_op=event.surface_op)
        else:
            session.append(event.type, event.data)
    return session


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > deriveMessages is deterministic (same log → identical derivation)"
)
def test_derive_messages_is_deterministic() -> None:
    for events in logs(1):
        a = build(events)
        assert a.derive_messages() == a.derive_messages()


@pytest.mark.upstream(f"{SPEC}::{GROUP} > seq is strictly monotonic and zero-based contiguous")
def test_seq_is_monotonic_and_contiguous() -> None:
    for events in logs(2):
        session = build(events)
        for index, event in enumerate(session.snapshot_events()):
            assert event["seq"] == index
        assert session.seq == len(events)


@pytest.mark.upstream(f"{SPEC}::{GROUP} > replay-from-seed reproduces the derivation identically")
def test_replay_reproduces_derivation() -> None:
    for events in logs(3):
        original = build(events)
        replayed = Session.create(
            session_id(f"replay-{next(_counter)}"), original.snapshot_events()
        )
        assert replayed.derive_messages() == original.derive_messages()
        # Every explicit replay grows by exactly one log-only boundary.
        assert replayed.snapshot_events()[: original.seq] == original.snapshot_events()
        assert replayed.seq == original.seq + 1


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > replaying a log that already ends in end-seed adds no further marker"
)
def test_replaying_end_seed_log_adds_no_marker() -> None:
    for events in logs(4):
        original = build(events)
        once = Session.create(session_id(f"idem-a-{next(_counter)}"), original.snapshot_events())
        twice = Session.create(session_id(f"idem-b-{next(_counter)}"), once.snapshot_events())
        # Lazy resume makes browsing a pickup, so this must not grow per open.
        assert twice.snapshot_events() == once.snapshot_events()


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > non-message events never affect derived history (any interleaving)"
)
def test_non_message_events_never_affect_history() -> None:
    rng = random.Random(5)
    for _ in range(RUNS):
        messages = gen_array(rng, gen_message_event, 12)
        noise = gen_array(rng, gen_non_message_event, 12)
        clean = build(messages).derive_messages()
        # An arbitrary merge of the two streams that PRESERVES each stream's
        # relative order (a random interleaving, not a fixed alternation).
        interleaved: list[Appendable] = []
        mi = ni = 0
        while mi < len(messages) or ni < len(noise):
            # take from noise when chosen and available, else from messages
            take_noise = ni < len(noise) and (mi >= len(messages) or rng.random() < 0.5)
            if take_noise:
                interleaved.append(noise[ni])
                ni += 1
            else:
                interleaved.append(messages[mi])
                mi += 1
        assert build(interleaved).derive_messages() == clean


@pytest.mark.upstream(
    f"{SPEC}::{GROUP} > every derived message has a known role and is frozen (append-only contract)"
)
def test_derived_messages_have_known_role_and_are_frozen() -> None:
    for events in logs(6):
        session = build(events)
        messages = session.derive_messages()
        before = copy.deepcopy(session.snapshot_events())
        for message in messages:
            assert message["role"] in ("user", "assistant", "system", "tool")
            # Derived messages are frozen shared projections: mutation RAISES
            # instead of relying on per-call clones for isolation.
            assert isinstance(message, FrozenJsonObject)
            content = message["content"]
            assert isinstance(content, list)
            with pytest.raises(TypeError):
                content.append({"type": "text", "text": "mutation"})
        assert session.snapshot_events() == before
