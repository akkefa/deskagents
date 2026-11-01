"""Check JSON values against upstream's persistence type inventory.

``tests/upstream_assets/persistence-schema.json`` is upstream's normalized type graph of
the session format: ``roots`` name the durable shapes (``SessionHeader``,
``SessionEventEnvelope``, ``event:<type>``) and ``nodes`` hold each structural type by
digest prefix. A value conforms when it matches the root's node: required properties
present, no property the type doesn't declare (unless it has an index signature), and
every nested value matching its node.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

INVENTORY = Path(__file__).resolve().parents[1] / "upstream_assets" / "persistence-schema.json"


@cache
def _inventory() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = json.loads(INVENTORY.read_text())
    roots = {root["key"]: root for root in data["roots"]}
    return roots, data["nodes"]


def root_keys() -> set[str]:
    return set(_inventory()[0])


def _primitive(kind: str, value: object) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return False  # never


def _literal(expected: object, value: object) -> bool:
    if isinstance(expected, bool) or isinstance(value, bool):
        return type(expected) is type(value) and expected == value
    return expected == value


def _check(key: str, value: object, path: str, problems: list[str]) -> None:
    node = _inventory()[1][key]
    kind = node["kind"]
    if kind == "opaque":
        return
    if kind == "primitive":
        if not _primitive(node["type"], value):
            problems.append(f"{path}: expected {node['type']}, got {value!r}"[:200])
    elif kind == "literal":
        if not _literal(node["value"], value):
            problems.append(f"{path}: expected {node['value']!r}, got {value!r}"[:200])
    elif kind == "union":
        for branch in node["types"]:
            attempt: list[str] = []
            _check(branch, value, path, attempt)
            if not attempt:
                return
        problems.append(f"{path}: matches no union branch: {value!r}"[:200])
    elif kind == "array":
        if not isinstance(value, list | tuple):
            problems.append(f"{path}: expected array, got {type(value).__name__}")
            return
        for index, item in enumerate(value):
            _check(node["element"], item, f"{path}[{index}]", problems)
    elif kind == "tuple":
        if not isinstance(value, list | tuple):
            problems.append(f"{path}: expected tuple, got {type(value).__name__}")
            return
        elements = node["elements"]
        required = sum(1 for element in elements if not element["optional"] and not element["rest"])
        if len(value) < required or (
            not any(e["rest"] for e in elements) and len(value) > len(elements)
        ):
            problems.append(f"{path}: tuple length {len(value)} does not fit")
            return
        for index, item in enumerate(value):
            element = elements[min(index, len(elements) - 1)]
            _check(element["type"], item, f"{path}[{index}]", problems)
    elif kind == "object":
        if not isinstance(value, Mapping):
            problems.append(f"{path}: expected object, got {type(value).__name__}")
            return
        declared = {prop["name"]: prop for prop in node["properties"]}
        for name, prop in declared.items():
            if name not in value:
                if not prop["optional"]:
                    problems.append(f"{path}.{name}: required property missing")
                continue
            _check(prop["type"], value[name], f"{path}.{name}", problems)
        for name in value:
            if name in declared:
                continue
            if not node["indices"]:
                problems.append(f"{path}.{name}: property not declared by the v4 type")
                continue
            for index in node["indices"]:
                _check(index["value"], value[name], f"{path}.{name}", problems)
    else:
        problems.append(f"{path}: unknown node kind {kind}")


def conformance_problems(root: str, value: object) -> list[str]:
    """Every way ``value`` departs from the inventory's ``root`` shape (empty when it conforms)."""
    roots, _ = _inventory()
    if root not in roots:
        return [f"inventory has no root {root!r}"]
    problems: list[str] = []
    _check(roots[root]["digest"][:16], value, root, problems)
    return problems


def event_problems(event: Mapping[str, Any]) -> list[str]:
    """Check one session event against its ``event:<type>`` root and the envelope.

    The ``SessionEventEnvelope`` root declares the fields around ``data``; each event
    root declares ``data`` itself.
    """
    envelope = {key: value for key, value in event.items() if key != "data"}
    return conformance_problems(f"event:{event['type']}", event) + conformance_problems(
        "SessionEventEnvelope", envelope
    )
