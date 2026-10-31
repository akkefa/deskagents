"""The enforced lossless-JSON Schema subset used by tools."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from deskagents.llm import HarnessError
from deskagents.util.values import is_json_value, stringify_json

_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_KEYS = {
    "type",
    "oneOf",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "description",
    "title",
    "default",
    "examples",
}


class JsonSchemaError(HarnessError):
    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = list(violations)
        super().__init__(f"unsupported JSON schema: {'; '.join(violations)}", "UNSUPPORTED_SCHEMA")


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _matches_type(kind: str, value: object) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": _number(value),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[kind]


def check_supported_json_schema(schema: object) -> list[str]:
    violations: list[str] = []
    seen: set[int] = set()

    def visit(node: object, path: str) -> None:
        if not isinstance(node, Mapping):
            violations.append(f"{path} must be a schema object")
            return
        identity = id(node)
        if identity in seen:
            violations.append(f"{path} is circular")
            return
        seen.add(identity)
        for key in node:
            if key not in _KEYS:
                violations.append(f"{path}.{key} is not supported")
        if "default" in node and not is_json_value(node["default"]):
            violations.append(f"{path}.default must be lossless JSON")
        if "examples" in node and not is_json_value(node["examples"]):
            violations.append(f"{path}.examples must be lossless JSON")
        if "oneOf" in node:
            branches = node["oneOf"]
            if not isinstance(branches, list) or len(branches) < 2 or "type" in node:
                violations.append(f"{path}.oneOf must contain at least two branches")
            else:
                for index, branch in enumerate(branches):
                    visit(branch, f"{path}.oneOf[{index}]")
            seen.remove(identity)
            return
        kind = node.get("type")
        if kind is not None and kind not in _TYPES:
            violations.append(f"{path}.type is unsupported")
        if "properties" in node:
            properties = node["properties"]
            if kind != "object" or not isinstance(properties, Mapping):
                violations.append(f"{path}.properties requires object type")
            else:
                for name, child in properties.items():
                    visit(child, f"{path}.properties.{name}")
        if "required" in node:
            required = node["required"]
            properties = node.get("properties", {})
            if not isinstance(required, list) or not all(
                isinstance(item, str) for item in required
            ):
                violations.append(f"{path}.required must be an array of strings")
            elif any(item not in properties for item in required):
                violations.append(f"{path}.required names an undeclared property")
        if "additionalProperties" in node and (
            kind != "object" or not isinstance(node["additionalProperties"], bool)
        ):
            violations.append(f"{path}.additionalProperties requires object type")
        if "items" in node:
            if kind != "array":
                violations.append(f"{path}.items requires array type")
            else:
                visit(node["items"], f"{path}.items")
        for literal in ("enum", "const"):
            if literal in node:
                values = node[literal] if literal == "enum" else [node[literal]]
                if (
                    kind not in {"string", "number", "integer", "boolean", "null"}
                    or not isinstance(values, list)
                    or not values
                    or not all(_matches_type(kind, value) for value in values)
                ):
                    violations.append(f"{path}.{literal} is invalid for its type")
        seen.remove(identity)

    visit(schema, "schema")
    return violations


def assert_supported_json_schema(schema: object) -> None:
    violations = check_supported_json_schema(schema)
    if violations:
        raise JsonSchemaError(violations)


def assert_object_json_schema(schema: object) -> None:
    assert_supported_json_schema(schema)
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise JsonSchemaError(["schema must declare type object"])


def _diagnostic_path(path: str) -> str:
    """The parameter validator's empty root path reads as ``arguments``."""
    return "arguments" if path == "" else path


def _property_path(path: str, key: str) -> str:
    return key if path == "" else f"{path}.{key}"


def _lossless_violation(path: str) -> list[str]:
    return [f'"{_diagnostic_path(path)}" must be a lossless JSON value']


def _lossless(value: object) -> bool:
    try:
        return is_json_value(value)
    except Exception:
        return False


def _json_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if isinstance(value, int):
        return True
    return math.isfinite(value) and not (value == 0 and math.copysign(1.0, value) < 0)


def _same_scalar(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _check_scalar(node: Mapping[str, Any], value: object, path: str) -> list[str]:
    if "enum" in node and not any(_same_scalar(value, item) for item in node["enum"]):
        return [f'"{_diagnostic_path(path)}" must be one of {stringify_json(node["enum"])}']
    if "const" in node and not _same_scalar(value, node["const"]):
        return [f'"{_diagnostic_path(path)}" must be {stringify_json(node["const"])}']
    return []


def _check_primitive(kind: str, node: Mapping[str, Any], value: object, path: str) -> list[str]:
    label = f'"{_diagnostic_path(path)}"'
    if kind == "string":
        return (
            _check_scalar(node, value, path)
            if isinstance(value, str)
            else [f"{label} must be a string"]
        )
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return [f"{label} must be a number"]
        if not _json_number(value):
            return [f"{label} must be a finite JSON number"]
        return _check_scalar(node, value, path)
    if kind == "integer":
        integral = _json_number(value) and (
            isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        )
        return _check_scalar(node, value, path) if integral else [f"{label} must be an integer"]
    if kind == "boolean":
        return (
            _check_scalar(node, value, path)
            if isinstance(value, bool)
            else [f"{label} must be a boolean"]
        )
    if kind == "null":
        return _check_scalar(node, value, path) if value is None else [f"{label} must be null"]
    raise ValueError(f"unexpected JsonSchemaType: {kind!r}")


@dataclass(eq=False)
class _Frame:
    node: Mapping[str, Any]
    value: object
    path: str
    catches: bool = False
    started: bool = False
    kind: str | None = None
    children: list[tuple[Mapping[str, Any], object, str]] = field(default_factory=list)
    child_index: int = 0
    violations: list[str] = field(default_factory=list)
    tail_violations: list[str] = field(default_factory=list)
    matches: int = 0


def validate_json_schema_value(
    schema: Mapping[str, Any], value: object, path: str = "value"
) -> list[str]:
    """Violations of ``value`` against an asserted schema, in walk order.

    Total for arbitrary values: a value that cannot be inspected yields a lossless-JSON
    violation instead of an exception. Uses explicit frames, so depth is not limited by
    the interpreter's recursion limit. ``path`` labels diagnostics; ``""`` is the
    parameter root and reads as ``arguments``.
    """
    frames = [_Frame(schema, value, path)]
    result: list[str] | None = None

    def receive(outcome: list[str]) -> None:
        nonlocal result
        if not frames:
            result = outcome
        elif frames[-1].kind == "oneOf":
            if not outcome:
                frames[-1].matches += 1
        else:
            frames[-1].violations.extend(outcome)

    def finish(outcome: list[str]) -> None:
        frames.pop()
        receive(outcome)

    while frames:
        frame = frames[-1]
        try:
            if frame.started:
                if frame.child_index < len(frame.children):
                    node, child, child_path = frame.children[frame.child_index]
                    frame.child_index += 1
                    frames.append(_Frame(node, child, child_path))
                    continue
                label = f'"{_diagnostic_path(frame.path)}"'
                if frame.kind == "oneOf":
                    matched = frame.matches
                    finish(
                        []
                        if matched == 1
                        else [f"{label} must match exactly one oneOf branch (matched {matched})"]
                    )
                    continue
                frame.violations.extend(frame.tail_violations)
                if frame.violations:
                    finish(frame.violations)
                elif frame.kind == "object":
                    lossless = _lossless(frame.value)
                    finish([] if lossless else [f"{label} must be a lossless JSON object"])
                else:
                    lossless = _lossless(frame.value)
                    finish([] if lossless else [f"{label} must be a dense lossless JSON array"])
                continue

            node = frame.node
            node_type = node.get("type")
            frame.catches = node_type is None or node_type in _TYPES
            frame.started = True
            if "oneOf" in node:
                frame.kind = "oneOf"
                frame.children = [(branch, frame.value, frame.path) for branch in node["oneOf"]]
                continue
            if node_type is None:
                finish([] if _lossless(frame.value) else _lossless_violation(frame.path))
                continue
            label = f'"{_diagnostic_path(frame.path)}"'
            if node_type == "object":
                if not isinstance(frame.value, dict):
                    finish([f"{label} must be an object"])
                    continue
                record = frame.value
                properties = node.get("properties") or {}
                frame.kind = "object"
                frame.violations = [
                    f'missing required property "{_property_path(frame.path, key)}"'
                    for key in node.get("required") or ()
                    if key not in record
                ]
                frame.children = [
                    (child, record[key], _property_path(frame.path, key))
                    for key, child in properties.items()
                    if key in record
                ]
                if node.get("additionalProperties") is False:
                    frame.tail_violations = [
                        f'"{_property_path(frame.path, key)}" is not a declared property '
                        "(additionalProperties: false)"
                        for key in record
                        if key not in properties
                    ]
                continue
            if node_type == "array":
                if not isinstance(frame.value, list | tuple):
                    finish([f"{label} must be an array"])
                    continue
                items = node.get("items")
                frame.kind = "array"
                if items is not None:
                    frame.children = [
                        (items, entry, f"{frame.path}[{index}]")
                        for index, entry in enumerate(frame.value)
                    ]
                continue
            finish(_check_primitive(node_type, node, frame.value, frame.path))
        except Exception:
            failed = frames.pop() if frames else None
            while failed is not None and not failed.catches:
                failed = frames.pop() if frames else None
            if failed is None:
                raise
            receive(_lossless_violation(failed.path))
    return result if result is not None else _lossless_violation(path)
