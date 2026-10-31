"""Author-facing tool schema helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from deskagents.llm import HarnessError
from deskagents.util.values import snapshot_json_value

from .json_schema import (
    assert_supported_json_schema,
    validate_json_schema_value,
)


class ToolArgsError(HarnessError):
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"invalid arguments: {'; '.join(violations)}", "INVALID_ARGS")


def value_schema_spec_to_json_schema(spec: Mapping[str, Any]) -> dict[str, Any]:
    if "oneOf" in spec:
        result = {
            key: value
            for key, value in spec.items()
            if key in {"description", "title", "default", "examples"}
        }
        result["oneOf"] = [value_schema_spec_to_json_schema(branch) for branch in spec["oneOf"]]
    else:
        kind = spec.get("type")
        if kind == "json":
            result = {
                key: value
                for key, value in spec.items()
                if key in {"description", "title", "default", "examples"}
            }
        else:
            result = {key: value for key, value in spec.items() if key != "required"}
            if kind == "object" and "properties" in result:
                compiled = parameter_schema_spec_to_json_schema(result["properties"])
                result["properties"] = compiled["properties"]
                if "required" in compiled:
                    result["required"] = compiled["required"]
            if kind == "array" and "items" in result:
                result["items"] = value_schema_spec_to_json_schema(result["items"])
    assert_supported_json_schema(result)
    return result


def parameter_schema_spec_to_json_schema(
    spec: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, node in spec.items():
        if node.get("required") is True:
            required.append(name)
        properties[name] = value_schema_spec_to_json_schema(node)
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    assert_supported_json_schema(result)
    return result


def validate_args(spec: Mapping[str, Any], args: object) -> list[str]:
    """Violations of model arguments against a parameter spec; never raises for bad input."""
    return validate_json_schema_value(parameter_schema_spec_to_json_schema(spec), args, "")


def project_tool_schema(
    definition: Mapping[str, Any], detach_parameters: bool = True
) -> dict[str, Any]:
    parameters = (
        snapshot_json_value(definition["parameters"])
        if detach_parameters
        else definition["parameters"]
    )
    if parameters is None:
        raise ValueError(
            f'tool "{definition["name"]}" parameters must be lossless JSON before schema projection'
        )
    return {
        "name": definition["name"],
        "description": definition["description"],
        "parameters": parameters,
        **({"deferLoading": True} if definition.get("deferLoading") is True else {}),
    }


def define_tool(**options: Any) -> dict[str, Any]:
    timeout = options.get("timeoutMs")
    if timeout is not None and not (
        isinstance(timeout, int | float)
        and not isinstance(timeout, bool)
        and math.isfinite(timeout)
        and timeout > 0
    ):
        raise ValueError(
            f"defineTool({options['name']}): timeoutMs must be a positive finite number"
        )
    parameters = parameter_schema_spec_to_json_schema(options["parameters"])
    output = options["output"]
    output_schema = value_schema_spec_to_json_schema(output["schema"])
    execute = options["execute"]

    def valid(args: object) -> bool:
        return not validate_json_schema_value(parameters, args, "")

    async def checked_execute(args: object, execution: Mapping[str, Any]) -> object:
        violations = validate_json_schema_value(parameters, args, "")
        if violations:
            raise ToolArgsError(violations)
        return await execute(args, execution)

    presentation_meta = output.get("presentationMeta")
    definition: dict[str, Any] = {
        "name": options["name"],
        "description": options["description"],
        "parameters": parameters,
        "output": {
            "schema": output_schema,
            "render": output["render"],
            **({"presentationMeta": presentation_meta} if presentation_meta is not None else {}),
        },
        **({"deferLoading": True} if options.get("deferLoading") is True else {}),
        **({"timeoutMs": timeout} if timeout is not None else {}),
        "execute": checked_execute,
    }
    for key in ("projectContent", "finalizeContent"):
        if options.get(key) is not None:
            definition[key] = options[key]
    # Presenters run on replay of arbitrary logged arguments, possibly from an older
    # schema, so they validate softly and fall back to the generic view instead of raising.
    present_call = options.get("presentCall")
    if present_call is not None:
        definition["presentCall"] = lambda args: present_call(args) if valid(args) else None
    present_result = options.get("presentResult")
    if present_result is not None:
        definition["presentResult"] = lambda args, result: (
            present_result(args, result) if valid(args) else None
        )
    is_concurrency_safe = options.get("isConcurrencySafe")
    if is_concurrency_safe is not None:
        definition["isConcurrencySafe"] = lambda args: valid(args) and is_concurrency_safe(args)
    return definition
