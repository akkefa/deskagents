"""Ported from packages/core/tools/tests/json-schema.spec.ts and tools.spec.ts.

Covers value validation, argument validation, ``define_tool`` validation, registration
and schema projection. JavaScript-only inputs (``undefined``, sparse arrays, getters,
proxies, other realms) are adapted to their nearest Python equivalents.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from typing import Any

import pytest
from deepseek_cordis import Context, ValidationError

from deskagents.core import SystemPrompt, ToolsService, define_tool
from deskagents.core.tools import (
    JsonSchemaError,
    NativeToolExecutor,
    ToolArgsError,
    ToolNotFoundError,
    assert_supported_json_schema,
    parameter_schema_spec_to_json_schema,
    validate_args,
    validate_json_schema_value,
)
from deskagents.llm import HarnessError
from deskagents.util.abort import AbortController

JSON_SPEC = "packages/core/tools/tests/json-schema.spec.ts::validateJsonSchemaValue > "
TOOLS = "packages/core/tools/tests/tools.spec.ts::"
RUNTIME = f"{TOOLS}ToolRuntime > "
ARGS = f"{TOOLS}validateArgs (the runtime-validation Agent Note, part 1) > "
DEFINE = f"{TOOLS}defineTool validation (the runtime-validation Agent Note, part 1) > "
GET = f"{TOOLS}ToolRuntime.get > "


def asserted(schema: dict[str, Any]) -> dict[str, Any]:
    assert_supported_json_schema(schema)
    return schema


async def tool_runtime() -> tuple[Context, ToolsService]:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolsService)
    tools = ctx.get("tools")
    assert isinstance(tools, ToolsService)
    tools.set_executor(NativeToolExecutor(ctx), ctx=ctx)
    return ctx, tools


def echo_tool(**overrides: Any) -> dict[str, Any]:
    async def execute(args: Mapping[str, Any], execution: object) -> str:
        return str(args.get("text", ""))

    return {
        **define_tool(
            name="echo",
            description="echo arguments back",
            parameters={"text": {"type": "string"}},
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
        ),
        **overrides,
    }


def content_tool(name: str, parameters: Mapping[str, Any], execute: Any, **extra: Any) -> Any:
    """Upstream's ``defineContentToolFixture``: the canonical value is the content."""
    return define_tool(
        name=name,
        description=name,
        parameters=parameters,
        output={"schema": {"type": "json"}, "render": lambda args, value: value},
        execute=execute,
        **extra,
    )


async def execute_one(tools: ToolsService, name: str, arguments: str = "{}") -> Mapping[str, Any]:
    batch = await tools.execute(
        {
            "calls": [{"id": "c1", "name": name, "arguments": arguments}],
            "signal": AbortController().signal,
        }
    )
    result: Mapping[str, Any] = batch["results"][0]["result"]
    return result


def names(tools: ToolsService) -> list[str]:
    return [schema["name"] for schema in tools.schemas()]


class Hostile(dict[str, Any]):
    """A container whose member access fails, like a throwing JavaScript getter."""

    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("getter exploded")

    def items(self) -> Any:
        raise RuntimeError("getter exploded")


class HostileMember:
    """A member whose type probe raises: ``isinstance`` reads ``__class__``."""

    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":
            raise RuntimeError("getter exploded")
        return object.__getattribute__(self, name)


# --- validateJsonSchemaValue ------------------------------------------------------------


@pytest.mark.upstream(f"{JSON_SPEC}validates scalar, array, object, and null roots")
def test_value_roots() -> None:
    assert validate_json_schema_value(asserted({"type": "string"}), "x") == []
    assert validate_json_schema_value(asserted({"type": "number"}), 1.5) == []
    assert validate_json_schema_value(asserted({"type": "integer"}), 2) == []
    assert validate_json_schema_value(asserted({"type": "boolean"}), True) == []
    assert validate_json_schema_value(asserted({"type": "null"}), None) == []
    schema = asserted({"type": "array", "items": {"type": "string"}})
    assert validate_json_schema_value(schema, ["x"]) == []
    assert validate_json_schema_value(asserted({"type": "object"}), {"x": 1}) == []


@pytest.mark.upstream(f"{JSON_SPEC}rejects wrong scalar types and lossy numbers")
def test_wrong_scalars_and_lossy_numbers() -> None:
    cases: list[tuple[str, object, str]] = [
        ("string", 1, '"value" must be a string'),
        ("number", "1", '"value" must be a number'),
        ("number", math.nan, '"value" must be a finite JSON number'),
        ("number", -0.0, '"value" must be a finite JSON number'),
        ("integer", 1.5, '"value" must be an integer'),
        ("boolean", "true", '"value" must be a boolean'),
        ("null", 0, '"value" must be null'),
    ]
    for kind, value, message in cases:
        assert validate_json_schema_value(asserted({"type": kind}), value) == [message]


@pytest.mark.upstream(f"{JSON_SPEC}enforces scalar enum and const together")
def test_enum_and_const() -> None:
    schema = asserted({"type": "string", "enum": ["a", "b"], "const": "a"})
    assert validate_json_schema_value(schema, "a") == []
    assert validate_json_schema_value(schema, "c") == ['"value" must be one of ["a","b"]']
    assert validate_json_schema_value(schema, "b") == ['"value" must be "a"']


@pytest.mark.upstream(
    f"{JSON_SPEC}validates object requiredness, nested values, and raw open defaults"
)
def test_object_requiredness_and_nesting() -> None:
    schema = asserted(
        {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {"line": {"type": "integer"}},
                    "required": ["line"],
                    "additionalProperties": False,
                },
            },
            "required": ["file"],
        }
    )
    assert (
        validate_json_schema_value(schema, {"file": "a", "extra": [1], "nested": {"line": 2}}) == []
    )
    assert validate_json_schema_value(schema, {"nested": {"line": 1}}) == [
        'missing required property "value.file"'
    ]
    assert validate_json_schema_value(schema, {"file": 1, "nested": {}}) == [
        '"value.file" must be a string',
        'missing required property "value.nested.line"',
    ]
    assert validate_json_schema_value(
        schema, {"file": "a", "nested": {"line": 1, "extra": True}}
    ) == ['"value.nested.extra" is not a declared property (additionalProperties: false)']
    assert validate_json_schema_value(schema, "x") == ['"value" must be an object']


@pytest.mark.upstream(
    f"{JSON_SPEC}treats present undefined as missing when required, then rejects other lossy "
    "objects"
)
def test_lossy_objects() -> None:
    required = asserted({"type": "object", "properties": {"x": {}}, "required": ["x"]})
    assert validate_json_schema_value(required, {}) == ['missing required property "value.x"']
    assert validate_json_schema_value(asserted({"type": "object"}), {"x": math.inf}) == [
        '"value" must be a lossless JSON object'
    ]
    assert validate_json_schema_value(asserted({"type": "object"}), datetime(1970, 1, 1)) == [
        '"value" must be an object'
    ]


@pytest.mark.upstream(
    f"{JSON_SPEC}returns a violation instead of throwing for a container with a hostile getter"
)
def test_hostile_container_is_a_violation() -> None:
    schema = asserted(
        {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
    )
    value = Hostile(answer=1)
    assert validate_json_schema_value(schema, value) == ['"value" must be a lossless JSON value']
    # A failing probe is contained by the nearest frame with a valid schema type.
    assert validate_json_schema_value(schema, {"answer": HostileMember()}) == [
        '"value.answer" must be a lossless JSON value'
    ]


@pytest.mark.upstream(f"{JSON_SPEC}validates dense arrays per index and rejects lossy arrays")
def test_arrays_per_index() -> None:
    schema = asserted({"type": "array", "items": {"type": "integer"}})
    assert validate_json_schema_value(schema, [1, 2]) == []
    assert validate_json_schema_value(schema, (1, 2)) == []
    assert validate_json_schema_value(schema, [1, 1.5]) == ['"value[1]" must be an integer']
    assert validate_json_schema_value(schema, "x") == ['"value" must be an array']
    assert validate_json_schema_value(asserted({"type": "array"}), [1, object()]) == [
        '"value" must be a dense lossless JSON array'
    ]


@pytest.mark.upstream(f"{JSON_SPEC}validates exact-one oneOf semantics, including overlap")
def test_exact_one_of() -> None:
    disjoint = asserted({"oneOf": [{"type": "string"}, {"type": "number"}]})
    assert validate_json_schema_value(disjoint, "x") == []
    assert validate_json_schema_value(disjoint, None) == [
        '"value" must match exactly one oneOf branch (matched 0)'
    ]
    overlap = asserted({"oneOf": [{"type": "number"}, {"type": "integer"}]})
    assert validate_json_schema_value(overlap, 1) == [
        '"value" must match exactly one oneOf branch (matched 2)'
    ]
    assert validate_json_schema_value(overlap, 1.5) == []


@pytest.mark.upstream(
    f"{JSON_SPEC}validates deeply nested exact-one unions without using the JavaScript call stack"
)
def test_deep_unions_are_validated_iteratively() -> None:
    schema: dict[str, Any] = {"type": "string"}
    for _ in range(5_000):
        schema = {"oneOf": [schema, {"type": "null"}]}
    assert validate_json_schema_value(schema, "leaf") == []
    assert validate_json_schema_value(schema, 42) == [
        '"value" must match exactly one oneOf branch (matched 0)'
    ]


@pytest.mark.upstream(f"{JSON_SPEC}an unconstrained schema accepts only lossless JSON values")
def test_unconstrained_schema_accepts_only_lossless_json() -> None:
    any_json = asserted({})
    for value in (None, True, 1, "x", [1], {"x": None}):
        assert validate_json_schema_value(any_json, value) == []
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    lossy: tuple[object, ...] = (lambda: 1, math.inf, -0.0, set(), cyclic, Hostile(x=1))
    for candidate in lossy:
        assert validate_json_schema_value(any_json, candidate) == [
            '"value" must be a lossless JSON value'
        ]


@pytest.mark.upstream(
    f"{JSON_SPEC}uses own properties for requiredness, recursion, and closed-object checks"
)
def test_own_property_semantics() -> None:
    schema = asserted(
        {
            "type": "object",
            "properties": {"toString": {"type": "string"}},
            "required": ["toString"],
        }
    )
    assert validate_json_schema_value(schema, {}) == ['missing required property "value.toString"']
    closed = asserted({"type": "object", "additionalProperties": False})
    assert validate_json_schema_value(closed, {"toString": 1}) == [
        '"value.toString" is not a declared property (additionalProperties: false)'
    ]
    assert validate_json_schema_value({"type": "object", "properties": None}, {}) == []
    assert validate_json_schema_value({"type": "object", "required": None}, {}) == []


@pytest.mark.upstream(f"{JSON_SPEC}keeps assertNever as a forged-schema backstop")
def test_forged_schema_type_raises() -> None:
    with pytest.raises(ValueError, match="tuple"):
        validate_json_schema_value({"type": "tuple"}, 1)


# --- validateArgs ---------------------------------------------------------------------


@pytest.mark.upstream(f"{ARGS}returns [] for valid args and is total over malformed input")
def test_validate_args_is_total() -> None:
    spec = {"path": {"type": "string", "required": True}, "limit": {"type": "number"}}
    assert validate_args(spec, {"path": "/tmp"}) == []
    assert validate_args(spec, {"path": "/tmp", "limit": 5}) == []
    malformed: tuple[object, ...] = (None, "nope", [])
    for args in malformed:
        assert len(validate_args(spec, args)) == 1


@pytest.mark.upstream(f"{ARGS}flags a missing required key and a required key present as undefined")
def test_validate_args_flags_missing_required_keys() -> None:
    spec = {"path": {"type": "string", "required": True}}
    assert validate_args(spec, {}) == ['missing required property "path"']


@pytest.mark.upstream(
    f"{ARGS}allows extra keys (no additionalProperties:false) and omitted optionals"
)
def test_validate_args_allows_extra_keys() -> None:
    spec = {"path": {"type": "string", "required": True}}
    assert validate_args(spec, {"path": "/tmp", "extra": 1}) == []


@pytest.mark.upstream(f"{ARGS}does not apply defaults (validation only)")
def test_validate_args_does_not_apply_defaults() -> None:
    args: dict[str, Any] = {}
    assert validate_args({"limit": {"type": "number", "default": 25}}, args) == []
    assert args == {}


@pytest.mark.upstream(f"{ARGS}type-checks primitives")
def test_validate_args_type_checks_primitives() -> None:
    spec = {"s": {"type": "string"}, "n": {"type": "number"}, "b": {"type": "boolean"}}
    assert validate_args(spec, {"s": 1}) == ['"s" must be a string']
    assert validate_args(spec, {"n": "x"}) == ['"n" must be a number']
    assert validate_args(spec, {"b": "x"}) == ['"b" must be a boolean']


@pytest.mark.upstream(f"{ARGS}checks enum membership")
def test_validate_args_checks_enum_membership() -> None:
    spec = {"color": {"type": "string", "enum": ["red", "green"]}}
    assert validate_args(spec, {"color": "red"}) == []
    assert validate_args(spec, {"color": "blue"}) == ['"color" must be one of ["red","green"]']


@pytest.mark.upstream(f"{ARGS}enforces type-correct scalar enum declarations")
def test_validate_args_enforces_typed_enums() -> None:
    spec = {"n": {"type": "number", "enum": [1, 2]}}
    assert validate_args(spec, {"n": 1}) == []
    assert validate_args(spec, {"n": 3}) == ['"n" must be one of [1,2]']
    assert validate_args(spec, {"n": True}) == ['"n" must be a number']
    with pytest.raises(JsonSchemaError):
        validate_args({"n": {"type": "number", "enum": ["1", "2"]}}, {"n": 1})


@pytest.mark.upstream(f"{ARGS}rejects an unknown schema type at the author boundary")
def test_validate_args_rejects_unknown_types() -> None:
    with pytest.raises(JsonSchemaError):
        validate_args({"x": {"type": "weird"}}, {"x": 1})


@pytest.mark.upstream(
    f"{ARGS}recurses into nested objects (and an object without properties only type-checks)"
)
def test_validate_args_recurses_into_objects() -> None:
    spec = {
        "config": {
            "type": "object",
            "additionalProperties": True,
            "required": True,
            "properties": {
                "host": {"type": "string", "required": True},
                "port": {"type": "number"},
            },
        },
        "bag": {"type": "object", "additionalProperties": True},
    }
    assert validate_args(spec, {"config": {"host": "h"}, "bag": {"anything": True}}) == []
    assert validate_args(spec, {"config": {"port": 9}, "bag": 5}) == [
        'missing required property "config.host"',
        '"bag" must be an object',
    ]


@pytest.mark.upstream(
    f"{ARGS}recurses into array items (and an array without items only type-checks)"
)
def test_validate_args_recurses_into_arrays() -> None:
    spec = {"tags": {"type": "array", "items": {"type": "string"}}, "raw": {"type": "array"}}
    assert validate_args(spec, {"tags": ["a", "b"], "raw": [1, {}, "x"]}) == []
    assert validate_args(spec, {"tags": ["a", 2]}) == ['"tags[1]" must be a string']
    assert validate_args(spec, {"tags": "nope"}) == ['"tags" must be an array']


@pytest.mark.upstream(f"{ARGS}validates arrays of objects element-wise")
def test_validate_args_checks_array_elements() -> None:
    spec = {
        "servers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {"host": {"type": "string", "required": True}},
            },
        }
    }
    assert validate_args(spec, {"servers": [{"host": "a"}, {}]}) == [
        'missing required property "servers[1].host"'
    ]


# --- defineTool validation ------------------------------------------------------------


def reader(execute: Any) -> Any:
    return content_tool("reader", {"path": {"type": "string", "required": True}}, execute)


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{DEFINE}returns an isError result with the violations when the model sends bad args"
)
@pytest.mark.upstream(f"{DEFINE}a schema-invalid call surfaces the structured error on the result")
async def test_bad_arguments_become_a_structured_error() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: Mapping[str, Any], execution: object) -> list[Mapping[str, Any]]:
        raise AssertionError("invalid arguments must not reach the body")

    tools.register(reader(execute))
    result = await execute_one(tools, "reader")
    assert result["isError"] is True
    assert result["content"] == [
        {"type": "text", "text": 'Error: invalid arguments: missing required property "path"'}
    ]
    assert result["error"] == {
        "message": 'invalid arguments: missing required property "path"',
        "info": {"name": "ToolArgsError", "code": "INVALID_ARGS"},
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{DEFINE}runs execute normally when args are valid")
async def test_valid_arguments_run_the_body() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: Mapping[str, Any], execution: object) -> list[Mapping[str, Any]]:
        return [{"type": "text", "text": f"read {args['path']}"}]

    tools.register(reader(execute))
    result = await execute_one(tools, "reader", '{"path":"/x"}')
    assert result == {
        "content": [{"type": "text", "text": "read /x"}],
        "isError": False,
        "value": [{"type": "text", "text": "read /x"}],
    }
    await ctx.fiber.dispose()


@pytest.mark.upstream(f"{DEFINE}ToolArgsError carries a stable code and the violation list")
def test_tool_args_error_shape() -> None:
    violations = ['missing required property "a"', '"b" must be a number']
    error = ToolArgsError(violations)
    assert isinstance(error, HarnessError)
    assert type(error).__name__ == "ToolArgsError"
    assert error.code == "INVALID_ARGS"
    assert error.violations == violations
    assert str(error) == 'invalid arguments: missing required property "a"; "b" must be a number'


@pytest.mark.asyncio
@pytest.mark.upstream(f"{DEFINE}a tool throwing a HarnessError surfaces its name and code")
@pytest.mark.upstream(f"{DEFINE}a non-HarnessError throw retains only its message")
async def test_thrown_errors_keep_their_identity() -> None:
    ctx, tools = await tool_runtime()

    async def coded(args: Mapping[str, Any], execution: object) -> str:
        raise HarnessError("disk full", "ENOSPC")

    async def plain(args: Mapping[str, Any], execution: object) -> str:
        raise RuntimeError("just a message")

    tools.register(echo_tool(name="coded", execute=coded))
    tools.register(echo_tool(name="plain", execute=plain))
    coded_result = await execute_one(tools, "coded")
    assert coded_result["error"] == {
        "message": "disk full",
        "info": {"name": "HarnessError", "code": "ENOSPC"},
    }
    assert coded_result["content"] == [{"type": "text", "text": "Error: disk full"}]
    plain_result = await execute_one(tools, "plain")
    assert plain_result["error"] == {"message": "just a message"}
    assert plain_result["content"] == [{"type": "text", "text": "Error: just a message"}]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{DEFINE}raw-registered tools are NOT validated by defineTool (MCP keeps its own)"
)
async def test_raw_tools_validate_their_own_input() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: object, execution: object) -> str:
        return type(args).__name__

    tools.register(
        {
            "name": "raw",
            "description": "raw tool",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "output": {
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            "execute": execute,
        }
    )
    assert (await execute_one(tools, "raw"))["isError"] is False
    await ctx.fiber.dispose()


@pytest.mark.upstream(f"{DEFINE}attaches a positive-finite timeoutMs to the definition")
@pytest.mark.upstream(f"{DEFINE}omits timeoutMs when not declared")
def test_define_tool_keeps_a_valid_timeout() -> None:
    async def execute(args: object, execution: object) -> list[object]:
        return []

    assert content_tool("x", {}, execute, timeoutMs=30_000)["timeoutMs"] == 30_000
    assert "timeoutMs" not in content_tool("x", {}, execute)


@pytest.mark.upstream(f"{DEFINE}throws when timeoutMs is zero or negative")
@pytest.mark.upstream(f"{DEFINE}throws when timeoutMs is non-finite")
@pytest.mark.parametrize("timeout", [0, -5, math.inf])
def test_define_tool_rejects_an_invalid_timeout(timeout: float) -> None:
    async def execute(args: object, execution: object) -> list[object]:
        return []

    with pytest.raises(ValueError, match="timeoutMs must be a positive finite number"):
        content_tool("x", {}, execute, timeoutMs=timeout)


# --- registration, lookup and schema projection ---------------------------------------


@pytest.mark.asyncio
@pytest.mark.upstream(f"{GET}get() returns the registered tool definition")
@pytest.mark.upstream(f"{GET}get() returns undefined for unknown tool names")
async def test_get_returns_the_definition_or_none() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo_tool())
    tool = tools.get("echo")
    assert tool is not None and tool["name"] == "echo"
    assert tools.get("nope") is None
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}requires every raw registration to declare its canonical output")
async def test_registration_requires_output() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: object, execution: object) -> list[object]:
        return []

    missing_output = {
        "name": "legacy-content-tool",
        "description": "missing output",
        "parameters": {},
        "execute": execute,
    }
    with pytest.raises(
        TypeError, match=r"must declare output \{ schema, render, presentationMeta\? \}"
    ):
        tools.register(missing_output)
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}rejects a non-positive or non-finite registration timeout")
async def test_registration_rejects_an_invalid_timeout() -> None:
    ctx, tools = await tool_runtime()
    for name, timeout in (("zero-timeout", 0), ("infinite-timeout", math.inf)):
        with pytest.raises(TypeError, match="timeoutMs must be a positive finite number"):
            tools.register(echo_tool(name=name, timeoutMs=timeout))
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}rejects duplicate names and unregisters on fiber dispose (HMR safety)"
)
async def test_duplicates_are_rejected_and_unload_unregisters() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo_tool())
    with pytest.raises(ValueError, match="already registered"):
        tools.register(echo_tool())

    class Scoped:
        inject = ["tools"]

        def apply(self, plugin_ctx: Context, config: object = None) -> None:
            plugin_tools = plugin_ctx.get("tools")
            assert isinstance(plugin_tools, ToolsService)
            plugin_tools.register(echo_tool(name="scoped"), ctx=plugin_ctx)

    fiber = await ctx.plugin(Scoped())
    assert names(tools) == ["echo", "scoped"]
    await fiber.dispose()
    assert names(tools) == ["echo"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}returns a callable disposer from register() that unregisters the tool"
)
async def test_register_returns_a_disposer() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo_tool())
    dispose = tools.register(echo_tool(name="disposable"))
    assert names(tools) == ["echo", "disposable"]
    dispose()
    assert names(tools) == ["echo"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}rolls back the tool entry when a tools/change listener throws (P1-1)"
)
async def test_a_throwing_change_listener_rolls_the_registration_back() -> None:
    ctx, tools = await tool_runtime()
    threw = False

    def on_change() -> None:
        nonlocal threw
        if not threw:
            threw = True
            raise RuntimeError("boom change listener")

    ctx.on("tools/change", on_change)
    with pytest.raises(RuntimeError, match="boom change listener"):
        tools.register(echo_tool())
    assert tools.get("echo") is None
    assert tools.schemas() == []
    dispose = tools.register(echo_tool())
    assert names(tools) == ["echo"]
    dispose()
    assert tools.get("echo") is None
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}schemas() snapshots tool schemas instead of exposing registry objects"
)
async def test_schemas_are_snapshots() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo_tool())
    first = tools.schemas()
    first[0]["parameters"]["properties"]["mutated"] = {"type": "string"}
    first[0]["description"] = "mutated"
    assert tools.schemas() == [
        {
            "name": "echo",
            "description": "echo arguments back",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
    ]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}rejects schema projection when a raw registration is not lossless JSON"
)
async def test_lossy_parameters_fail_schema_projection() -> None:
    ctx, tools = await tool_runtime()
    tools.register(
        echo_tool(name="lossy-schema", parameters={"type": "object", "default": math.nan})
    )
    with pytest.raises(
        ValueError,
        match='tool "lossy-schema" parameters must be lossless JSON before schema projection',
    ):
        tools.schemas()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}schemas() drops host callbacks — they must never reach the model")
@pytest.mark.upstream(
    f"{RUNTIME}schemas() excludes timeoutMs — the budget must never reach the model"
)
async def test_schemas_carry_only_model_fields() -> None:
    ctx, tools = await tool_runtime()
    tools.register(
        echo_tool(
            name="present",
            timeoutMs=5_000,
            finalizeContent=lambda execution, result: result["content"],
            presentCall=lambda args: {"card": "generic"},
            presentResult=lambda args, result: {"card": "generic"},
        )
    )
    assert sorted(tools.schemas()[0]) == ["description", "name", "parameters"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}registers tools, exposes schemas, and feeds the system-prompt assembly"
)
async def test_registration_feeds_the_prompt_assembly() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo_tool())
    prompt = ctx.get("systemPrompt")
    assert isinstance(prompt, SystemPrompt)
    assembly = await prompt.assemble({})
    assert [tool["name"] for tool in assembly["tools"]] == ["echo"]
    await ctx.fiber.dispose()


# --- executing and normalizing results --------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}executes a tool and returns its content")
async def test_execution_returns_content_and_notifies() -> None:
    ctx, tools = await tool_runtime()
    tools.register(echo_tool())
    observed: list[object] = []
    ctx.on("tools/result", lambda execution, result: observed.append(result))
    result = await execute_one(tools, "echo", '{"text":"hi"}')
    assert result == {"content": [{"type": "text", "text": "hi"}], "isError": False, "value": "hi"}
    assert observed == [result]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}projects presentation metadata from the canonical value")
@pytest.mark.upstream(f"{RUNTIME}omits meta when no presentation projector is declared")
async def test_presentation_metadata_is_projected_when_declared() -> None:
    ctx, tools = await tool_runtime()
    base = echo_tool()

    async def execute(args: object, execution: object) -> str:
        return "ok"

    diffs = {"diffs": [{"path": "a", "oldText": None, "newText": "x"}]}
    tools.register(
        {
            **base,
            "name": "meta-tool",
            "output": {**base["output"], "presentationMeta": lambda args, value: diffs},
            "execute": execute,
        }
    )
    tools.register({**base, "name": "no-meta-tool", "execute": execute})
    assert await execute_one(tools, "meta-tool") == {
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
        "meta": diffs,
        "value": "ok",
    }
    plain = await execute_one(tools, "no-meta-tool")
    assert "meta" not in plain
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}normalizes a contract-violating non-cloneable result before final notification"
)
async def test_non_json_presentation_metadata_is_an_output_error() -> None:
    ctx, tools = await tool_runtime()
    base = echo_tool()
    observed: list[bool] = []
    ctx.on("tools/result", lambda execution, result: observed.append(result["isError"]))
    tools.register(
        {
            **base,
            "name": "bad-meta",
            "output": {**base["output"], "presentationMeta": lambda args, value: lambda: None},
        }
    )
    result = await execute_one(tools, "bad-meta")
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("Error:")
    assert result["error"]["info"] == {"name": "ToolOutputError", "code": "INVALID_TOOL_OUTPUT"}
    assert observed == [True]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}normalizes a throwing final content callback without invoking it again"
)
async def test_a_throwing_finalizer_runs_once() -> None:
    ctx, tools = await tool_runtime()
    calls = 0

    def finalize(execution: object, result: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("finalizer violated its total contract")

    tools.register(echo_tool(name="throwing-finalizer", finalizeContent=finalize))
    result = await execute_one(tools, "throwing-finalizer")
    assert result == {
        "content": [{"type": "text", "text": "Error: finalizer violated its total contract"}],
        "isError": True,
        "error": {"message": "finalizer violated its total contract"},
    }
    assert calls == 1
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}rejects lossy and schema-mismatched body values before post-execute"
)
async def test_lossy_and_mismatched_values_are_output_errors() -> None:
    ctx, tools = await tool_runtime()

    def returning(value: object) -> Callable[..., Any]:
        async def execute(args: object, execution: object) -> object:
            return value

        return execute

    tools.register(
        define_tool(
            name="lossy-output",
            description="lossy",
            parameters={},
            output={"schema": {"type": "json"}, "render": lambda args, value: []},
            execute=returning(lambda: None),
        )
    )
    tools.register(
        define_tool(
            name="wrong-output",
            description="wrong schema",
            parameters={},
            output={"schema": {"type": "string"}, "render": lambda args, value: []},
            execute=returning(42),
        )
    )
    output_error = {"name": "ToolOutputError", "code": "INVALID_TOOL_OUTPUT"}
    lossy = await execute_one(tools, "lossy-output")
    assert lossy["error"]["info"] == output_error
    assert "not lossless JSON" in lossy["content"][0]["text"]
    mismatch = await execute_one(tools, "wrong-output")
    assert mismatch["error"]["info"] == output_error
    assert '"value" must be a string' in mismatch["content"][0]["text"]
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}contains a throwing output.%s projector as one failed call")
@pytest.mark.parametrize("projector", ["render", "presentationMeta"])
async def test_a_throwing_projector_is_one_failed_call(projector: str) -> None:
    ctx, tools = await tool_runtime()

    def render(args: object, value: object) -> list[Mapping[str, Any]]:
        if projector == "render":
            raise RuntimeError("renderer exploded")
        return [{"type": "text", "text": "ok"}]

    def presentation_meta(args: object, value: object) -> None:
        if projector == "presentationMeta":
            raise RuntimeError("metadata exploded")

    async def execute(args: object, execution: object) -> str:
        return "ok"

    tools.register(
        define_tool(
            name=f"throwing-{projector}",
            description=projector,
            parameters={},
            output={
                "schema": {"type": "string"},
                "render": render,
                "presentationMeta": presentation_meta,
            },
            execute=execute,
        )
    )
    result = await execute_one(tools, f"throwing-{projector}")
    assert result["isError"] is True
    expected = "renderer exploded" if projector == "render" else "metadata exploded"
    assert expected in result["error"]["message"]
    assert result["error"]["info"] == {"name": "ToolOutputError", "code": "INVALID_TOOL_OUTPUT"}
    assert "value" not in result
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}returns isError results for unknown tools and throwing tools")
async def test_unknown_and_throwing_tools_are_error_results() -> None:
    ctx, tools = await tool_runtime()

    async def boom(args: object, execution: object) -> str:
        raise RuntimeError("exploded")

    tools.register(echo_tool(name="boom", execute=boom))
    unknown = await execute_one(tools, "nope")
    assert unknown["content"] == [{"type": "text", "text": 'Error: unknown tool "nope"'}]
    assert unknown["error"] == {
        "message": 'unknown tool "nope"',
        "info": {"name": "ToolNotFoundError", "code": "UNKNOWN_TOOL"},
    }
    thrown = await execute_one(tools, "boom")
    assert thrown["content"] == [{"type": "text", "text": "Error: exploded"}]
    await ctx.fiber.dispose()


@pytest.mark.upstream(f"{RUNTIME}ToolNotFoundError carries a stable message and code")
def test_tool_not_found_error_shape() -> None:
    error = ToolNotFoundError("ghost")
    assert isinstance(error, HarnessError)
    assert error.code == "UNKNOWN_TOOL"
    assert str(error) == 'unknown tool "ghost"'


@pytest.mark.asyncio
@pytest.mark.upstream(f"{RUNTIME}rejects removed execution configuration %j during plugin mounting")
@pytest.mark.parametrize("config", [{"mode": "ptc"}, {"maxParallelSubCalls": 4}])
async def test_removed_execution_configuration_is_rejected(config: Mapping[str, Any]) -> None:
    ctx = Context()
    try:
        await ctx.plugin(SystemPrompt)
        with pytest.raises(ValidationError, match="invalid config"):
            await ctx.plugin(ToolsService, config)
        assert ctx.get("tools") is None
    finally:
        await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(
    f"{RUNTIME}register() returns the EXACT effect disposer: a composite yield nests the "
    "teardown in order"
)
async def test_register_disposer_nests_inside_a_composite_effect() -> None:
    ctx, tools = await tool_runtime()
    order: list[str] = []

    def plugin(inner: Context, config: object = None) -> None:
        def composite() -> Iterator[Callable[[], object]]:
            yield lambda: order.append("disposed-last")
            yield tools.register(echo_tool(name="nested"), ctx=inner)
            order.append("registered")

            async def first() -> None:
                await asyncio.sleep(0)
                still = tools.get("nested") is not None
                order.append("first: still registered" if still else "first: already gone")

            yield first

        inner.effect(composite)

    fiber = await ctx.plugin(plugin)
    await fiber.dispose()
    assert order == ["registered", "first: still registered", "disposed-last"]
    assert tools.get("nested") is None
    await ctx.fiber.dispose()


# --- the schema DSL ------------------------------------------------------------------------

DSL = f"{TOOLS}defineTool / schema DSL > "
EDGE = f"{TOOLS}schema DSL edge cases > "
CONTRACTS = f"{TOOLS}schema DSL optional and nested contracts > "
PRESENT = f"{TOOLS}defineTool presentation (presentCall / presentResult) > "


@pytest.mark.upstream(
    f"{DSL}converts ParameterSchemaSpec to standard JSON Schema with required array"
)
@pytest.mark.upstream(f"{DSL}handles empty spec (no properties, no required)")
def test_parameter_specs_compile_to_json_schema() -> None:
    spec: dict[str, dict[str, Any]] = {
        "path": {"type": "string", "required": True, "description": "Absolute path"},
        "offset": {"type": "number"},
        "limit": {"type": "number", "description": "Max lines"},
    }
    assert parameter_schema_spec_to_json_schema(spec) == {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path"},
            "offset": {"type": "number"},
            "limit": {"type": "number", "description": "Max lines"},
        },
        "required": ["path"],
    }
    assert parameter_schema_spec_to_json_schema({}) == {"type": "object", "properties": {}}


@pytest.mark.upstream(f"{DSL}handles nested object spec")
@pytest.mark.upstream(f"{CONTRACTS}runtime JSON Schema matches the array-of-objects inference")
def test_nested_specs_compile_their_own_required_lists() -> None:
    server: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {"host": {"type": "string", "required": True}, "port": {"type": "number"}},
    }
    compiled_server = {
        "type": "object",
        "additionalProperties": True,
        "properties": {"host": {"type": "string"}, "port": {"type": "number"}},
        "required": ["host"],
    }
    assert parameter_schema_spec_to_json_schema({"config": {**server, "required": True}}) == {
        "type": "object",
        "properties": {"config": compiled_server},
        "required": ["config"],
    }
    assert parameter_schema_spec_to_json_schema(
        {"servers": {"type": "array", "items": server}}
    ) == {
        "type": "object",
        "properties": {"servers": {"type": "array", "items": compiled_server}},
    }


@pytest.mark.upstream(f"{EDGE}emits enum values in JSON Schema property")
@pytest.mark.upstream(f"{EDGE}emits default value in JSON Schema property")
@pytest.mark.upstream(f"{EDGE}handles enum and default together in one property")
@pytest.mark.upstream(f"{EDGE}omits description, enum, default keys when not specified")
def test_property_annotations_are_emitted_only_when_given() -> None:
    properties = parameter_schema_spec_to_json_schema(
        {
            "color": {
                "type": "string",
                "enum": ["red", "green", "blue"],
                "description": "Color choice",
            },
            "limit": {"type": "number", "default": 25},
            "level": {"type": "string", "enum": ["low", "high"], "default": "low"},
            "bare": {"type": "string"},
        }
    )["properties"]
    assert properties == {
        "color": {
            "type": "string",
            "enum": ["red", "green", "blue"],
            "description": "Color choice",
        },
        "limit": {"type": "number", "default": 25},
        "level": {"type": "string", "enum": ["low", "high"], "default": "low"},
        "bare": {"type": "string"},
    }


@pytest.mark.upstream(f"{EDGE}handles array items without nested properties (plain type array)")
@pytest.mark.upstream(f"{EDGE}handles array with no items (items omitted)")
@pytest.mark.upstream(
    f"{EDGE}handles nested object with all-optional properties (no required array)"
)
def test_arrays_and_all_optional_objects_compile_plainly() -> None:
    properties = parameter_schema_spec_to_json_schema(
        {
            "tags": {"type": "array", "items": {"type": "string"}},
            "raw": {"type": "array"},
            "config": {
                "type": "object",
                "additionalProperties": True,
                "properties": {"host": {"type": "string"}, "port": {"type": "number"}},
            },
        }
    )["properties"]
    assert properties["tags"] == {"type": "array", "items": {"type": "string"}}
    assert properties["raw"] == {"type": "array"}
    assert properties["config"] == {
        "type": "object",
        "additionalProperties": True,
        "properties": {"host": {"type": "string"}, "port": {"type": "number"}},
    }


@pytest.mark.asyncio
@pytest.mark.upstream(f"{DSL}defineTool returns a valid ToolDefinition with typed execute")
async def test_a_defined_tool_registers_and_runs() -> None:
    ctx, tools = await tool_runtime()

    async def execute(args: Mapping[str, Any], execution: object) -> str:
        text = str(args["text"])
        return text.upper() if args.get("uppercase") else text

    tools.register(
        define_tool(
            name="typed-echo",
            description="A typed echo tool",
            parameters={
                "text": {"type": "string", "required": True},
                "uppercase": {"type": "boolean"},
            },
            output={
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            execute=execute,
        )
    )
    assert tools.schemas() == [
        {
            "name": "typed-echo",
            "description": "A typed echo tool",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "uppercase": {"type": "boolean"}},
                "required": ["text"],
            },
        }
    ]
    result = await execute_one(tools, "typed-echo", '{"text":"hello","uppercase":true}')
    assert result == {
        "isError": False,
        "value": "HELLO",
        "content": [{"type": "text", "text": "HELLO"}],
    }
    await ctx.fiber.dispose()


@pytest.mark.asyncio
@pytest.mark.upstream(f"{DSL}still accepts raw JSON-Schema ToolDefinition directly (MCP interop)")
async def test_raw_json_schema_definitions_register_directly() -> None:
    ctx, tools = await tool_runtime()
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def execute(args: Mapping[str, Any], execution: object) -> str:
        return str(args["path"])

    tools.register(
        {
            "name": "raw-tool",
            "description": "Raw JSON Schema tool (like an MCP adapter would register)",
            "parameters": parameters,
            "output": {
                "schema": {"type": "string"},
                "render": lambda args, value: [{"type": "text", "text": value}],
            },
            "execute": execute,
        }
    )
    assert tools.schemas()[0]["parameters"] == parameters
    result = await execute_one(tools, "raw-tool", '{"path":"/tmp"}')
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": "/tmp"}]
    await ctx.fiber.dispose()


async def no_content(args: object, execution: object) -> list[Any]:
    return []


@pytest.mark.upstream(
    f"{PRESENT}threads presentCall/presentResult onto the ToolDefinition with typed args"
)
def test_presenters_are_threaded_onto_the_definition() -> None:
    tool = content_tool(
        "demo",
        {"path": {"type": "string", "required": True}, "n": {"type": "number"}},
        no_content,
        presentCall=lambda args: {
            "card": "generic",
            "title": f"Open {args['path']}",
            "kind": "read",
            "rawInput": args["path"],
        },
        presentResult=lambda args, result: {
            "card": "generic",
            "title": f"Opened {args['path']}",
            "content": result["content"],
        },
    )
    assert tool["presentCall"]({"path": "/a", "n": 2}) == {
        "card": "generic",
        "title": "Open /a",
        "kind": "read",
        "rawInput": "/a",
    }
    content = [{"type": "text", "text": "x"}]
    assert tool["presentResult"]({"path": "/a"}, {"content": content, "isError": False}) == {
        "card": "generic",
        "title": "Opened /a",
        "content": content,
    }


@pytest.mark.upstream(
    f"{PRESENT}a tool without presentCall/presentResult leaves them undefined (UI falls back "
    "generically)"
)
def test_presenters_are_absent_unless_declared() -> None:
    tool = content_tool("plain", {"x": {"type": "string", "required": True}}, no_content)
    assert "presentCall" not in tool
    assert "presentResult" not in tool


@pytest.mark.upstream(
    f"{PRESENT}presentCall/presentResult validate softly: malformed args return undefined, "
    "never throw (display runs on replay)"
)
def test_presenters_validate_softly() -> None:
    tool = content_tool(
        "demo",
        {"path": {"type": "string", "required": True}},
        no_content,
        presentCall=lambda args: {"card": "generic", "title": args["path"]},
        presentResult=lambda args, result: {"card": "generic", "title": args["path"]},
    )
    assert tool["presentCall"]({}) is None
    assert tool["presentResult"]({"wrong": 1}, {"content": [], "isError": False}) is None
