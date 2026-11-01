"""Scoped tool registry, execution pipeline, schemas, and native executor.

Upstream: packages/core/tools and packages/core/tool-executor-native at ce7073d.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from deepseek_cordis import Context, Service, ValidationError, ValidationIssue, current_context

from deskagents._cordis import emit_contained
from deskagents.core.scope import (
    AnonymousEntries,
    NamedEntries,
    ScopedLayers,
    ScopeKey,
    scope_of,
    scope_target,
)
from deskagents.llm import HarnessError, create_tool_result_message
from deskagents.util.abort import AbortSignal, is_aborted, throw_if_aborted
from deskagents.util.values import (
    deep_freeze,
    is_json_value,
    parse_json,
    snapshot_json_value,
    stringify_json,
)

from .json_schema import (
    JsonSchemaError,
    assert_object_json_schema,
    assert_supported_json_schema,
    check_supported_json_schema,
    validate_json_schema_value,
)
from .schema import (
    ToolArgsError,
    define_tool,
    parameter_schema_spec_to_json_schema,
    project_tool_schema,
    validate_args,
    value_schema_spec_to_json_schema,
)

logger = logging.getLogger(__name__)

RUN_CODE_NAME = "run_code"
TOOL_ABORTED = "ABORTED"
TOOL_ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"


class ToolNotFoundError(HarnessError):
    def __init__(self, name: str, reachable_from: str | None = None) -> None:
        message = (
            f'unknown tool "{name}"'
            if reachable_from is None
            else f'unknown tool "{name}": {reachable_from}'
        )
        super().__init__(message, "UNKNOWN_TOOL")


class ToolOutputError(HarnessError):
    def __init__(self, name: str, violations: Sequence[str]) -> None:
        self.violations = list(violations)
        super().__init__(
            f'tool "{name}" returned invalid output: {"; ".join(violations)}',
            "INVALID_TOOL_OUTPUT",
        )


class ToolActivationError(HarnessError):
    def __init__(self, name: str, cause: object) -> None:
        super().__init__(
            f'cannot resolve availability of tool "{name}"',
            "TOOL_ACTIVATION_FAILED",
            cause=cause,
        )


class ToolExecutorUnavailableError(HarnessError):
    def __init__(self) -> None:
        super().__init__(
            "The selected tool executor is unavailable. Load an executor before running the agent.",
            "TOOL_EXECUTOR_UNAVAILABLE",
        )


class ToolBatchExecutionError(Exception):
    def __init__(self, cause: object, partial: Mapping[str, Any]) -> None:
        super().__init__("Tool batch execution failed")
        self.cause = cause
        self.partial = partial


def _aborted(signal: object) -> bool:
    return is_aborted(signal)


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _error_message(error: BaseException) -> str:
    try:
        return str(error)
    except Exception:
        return "<unprintable thrown value>"


def _failure(error: BaseException) -> dict[str, Any]:
    info = (
        {"name": type(error).__name__, "code": error.code}
        if isinstance(error, HarnessError)
        else None
    )
    message = _error_message(error)
    return {
        "isError": True,
        "error": {"message": message, **({"info": info} if info else {})},
        "content": [{"type": "text", "text": f"Error: {message}"}],
    }


def _denied(reason: str, info: object = None) -> dict[str, Any]:
    return {
        "isError": True,
        "error": {
            "message": reason,
            **({"info": info} if info is not None else {}),
        },
        "content": [{"type": "text", "text": f"Error: {reason}"}],
    }


def _cancelled(message: str, code: str, prior: Mapping[str, Any] | None) -> dict[str, Any]:
    contexts = list((prior or {}).get("additionalContexts", ()))
    return {
        "isError": True,
        "error": {"message": message, "info": {"name": "AbortError", "code": code}},
        "content": [{"type": "text", "text": f"Error: {message}"}],
        **({"additionalContexts": contexts} if contexts else {}),
    }


def _aborted_result(prior: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _cancelled("tool call aborted", TOOL_ABORTED, prior)


def _aborted_before_dispatch_result(prior: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _cancelled("tool call aborted before dispatch", TOOL_ABORTED_BEFORE_DISPATCH, prior)


def _failure_message_from_content(content: Sequence[Mapping[str, Any]]) -> str:
    text = "\n".join(
        block["text"] if block["type"] == "text" else f"[{block['type']} content]"
        for block in content
    )
    return text or "tool result blocked by post-execute policy"


def _materialize(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Snapshot and freeze one result, rejecting data that is not lossless JSON."""
    presentation = {
        "content": result["content"],
        **({"meta": result["meta"]} if "meta" in result else {}),
        **(
            {"additionalContexts": result["additionalContexts"]}
            if "additionalContexts" in result
            else {}
        ),
    }
    candidate: dict[str, Any] = (
        {"isError": True, "error": result["error"], **presentation}
        if result["isError"]
        else {
            "isError": False,
            **presentation,
            **({"concludesTurn": True} if result.get("concludesTurn") is True else {}),
        }
    )
    if not is_json_value(candidate):
        raise TypeError("tool result must be losslessly JSON-serializable")
    frozen = cast(Mapping[str, Any], snapshot_json_value(candidate, freeze=True))
    if result["isError"]:
        return frozen
    return cast(Mapping[str, Any], deep_freeze({**frozen, "value": result.get("value")}))


def _snapshot_value(name: str, candidate: object) -> object:
    """Detach a body or policy value, classifying any failure as invalid output."""
    try:
        lossless = is_json_value(candidate)
    except Exception as error:
        raise ToolOutputError(name, [f"value snapshot failed: {_error_message(error)}"]) from error
    if not lossless:
        raise ToolOutputError(name, ["value is not lossless JSON"])
    return snapshot_json_value(candidate)


def _projection_error(name: str, projector: str, error: BaseException) -> ToolOutputError:
    return ToolOutputError(name, [f"output.{projector} failed: {_error_message(error)}"])


def _snapshot_projection(name: str, projector: str, candidate: object) -> Any:
    """Detach one render or presentationMeta result, or raise the invalid-output error."""
    try:
        lossless = is_json_value(candidate)
    except Exception as error:
        raise _projection_error(name, projector, error) from error
    if not lossless or (projector == "render" and not isinstance(candidate, list)):
        raise ToolOutputError(name, [f"output.{projector} returned non-lossless JSON"])
    return snapshot_json_value(candidate)


def _parse_arguments(raw: object) -> object:
    """Model arguments as ``JSON.parse`` reads them; empty is ``{}``, malformed stays text."""
    if not raw:
        return {}
    if not isinstance(raw, str):
        return raw
    try:
        return parse_json(raw)
    except ValueError:
        return raw


class ToolRunContext(Mapping[str, Any]):
    """One call's execution view. A ``tools/execute`` wrapper may replace ``signal``."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)
        self.additional_contexts: list[Mapping[str, Any]] = []
        self.concludes_turn = False
        self._settled = False

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def signal(self) -> Any:
        return self._values["signal"]

    @signal.setter
    def signal(self, signal: object) -> None:
        if self._settled:
            raise TypeError("a settled tool execution is read-only")
        self._values["signal"] = signal

    def defer_context(self, context: Mapping[str, Any]) -> None:
        self.additional_contexts.append(context)

    def conclude_turn(self) -> None:
        self.concludes_turn = True


@dataclass(eq=False)
class _Call:
    """Scheduler state for one call, kept outside the wrapper-visible execution."""

    block: Mapping[str, Any]
    execution: ToolRunContext
    caller_signal: object
    scope: Callable[[Context], bool]
    project: Callable[..., Any] | None = None
    finalize: Callable[..., Any] | None = None
    body_invoked: bool = False


@dataclass(eq=False)
class _Scheduled:
    """A prepared call: ``dispatch``, or a result that does or does not get post-execute."""

    kind: Literal["dispatch", "post-result", "final-result"]
    call: _Call
    result: Mapping[str, Any] | None = None


@dataclass
class _Registration:
    definition: Mapping[str, Any]
    when: Callable[[Mapping[str, Any]], bool | Awaitable[bool]] | None


class _ToolLayer:
    def __init__(self, scope: ScopeKey | None) -> None:
        suffix = (
            " (for a per-agent variant, register through that agent's `agent.ctx` instead)"
            if scope is None
            else " in this scope"
        )
        self.tools: NamedEntries[_Registration] = NamedEntries(
            lambda name: ValueError(f'tool "{name}" is already registered{suffix}')
        )
        self.restrictions: AnonymousEntries[Mapping[str, frozenset[str]]] = AnonymousEntries()

    def is_empty(self) -> bool:
        return self.tools.is_empty() and self.restrictions.is_empty()

    def admits(self, name: str) -> bool:
        return all(
            ("allow" not in item or name in item["allow"])
            and ("deny" not in item or name not in item["deny"])
            for item in self.restrictions.values()
        )


class _GuardLayer:
    def __init__(self, scope: ScopeKey | None) -> None:
        del scope
        self.guards: AnonymousEntries[Callable[[Mapping[str, Any]], str | None]] = (
            AnonymousEntries()
        )

    def is_empty(self) -> bool:
        return self.guards.is_empty()


class ResolvedToolSet:
    def __init__(
        self,
        owner: ScopeKey,
        admitted: Mapping[str, tuple[Mapping[str, Any], _Registration]],
        service: ToolsService,
    ) -> None:
        self._owner, self._admitted, self._service = owner, dict(admitted), service
        self._closed = False

    def dispose(self) -> None:
        self._closed = True

    def view(self, scope: ScopeKey | None) -> Mapping[str, Mapping[str, Any]]:
        """Admitted tools still registered as admitted; a same-name re-registration is not."""
        if self._closed or scope is not self._owner:
            return {}
        visible, registrations, _ = self._service._view(scope)
        return {
            name: definition
            for name, (definition, registration) in self._admitted.items()
            if visible.get(name) is definition and registrations.get(name) is registration
        }


class _ToolsConfig:
    """Rejects the removed execution keys, which untyped config files can still carry."""

    REMOVED = {
        "mode": "mode was removed; mount a native or PTC tool executor instead",
        "maxParallelSubCalls": "maxParallelSubCalls moved to the PTC tool executor configuration",
    }

    def validate(self, value: object) -> object:
        if isinstance(value, Mapping):
            issues = [
                ValidationIssue(message, (key,))
                for key, message in self.REMOVED.items()
                if key in value
            ]
            if issues:
                raise ValidationError(issues)
        return value


class ToolsService(Service):
    name = "tools"
    Config = _ToolsConfig()

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, config)
        self._layers = ScopedLayers(_ToolLayer, lambda: ctx.emit("tools/change"))
        self._guards = ScopedLayers(_GuardLayer, lambda: None)
        self._executors: dict[int, object | None] = {}
        self._root_executor: object | None = None
        system_prompt = ctx.get("systemPrompt")
        from deskagents.core.system_prompt import SystemPrompt

        if not isinstance(system_prompt, SystemPrompt):
            raise RuntimeError("ToolsService requires SystemPrompt")
        system_prompt.tools(self._describe, ctx=ctx)

    def caller(self, ctx: Context | None = None) -> Context:
        candidate = ctx or current_context()
        return candidate if candidate is not None and candidate.root is self.ctx.root else self.ctx

    def register(
        self,
        definition: Mapping[str, Any],
        *,
        when: Callable[[Mapping[str, Any]], bool | Awaitable[bool]] | None = None,
        ctx: Context | None = None,
    ) -> Callable[[], object]:
        name = definition["name"]
        output = definition.get("output")
        if (
            not isinstance(output, Mapping)
            or not callable(output.get("render"))
            or ("presentationMeta" in output and not callable(output["presentationMeta"]))
        ):
            raise TypeError(
                f'tool "{name}" must declare output {{ schema, render, presentationMeta? }}'
            )
        assert_supported_json_schema(output.get("schema"))
        if not _positive_finite(definition.get("timeoutMs", 1)):
            raise TypeError(f'tool "{name}" timeoutMs must be a positive finite number')
        if name == RUN_CODE_NAME:
            raise ValueError(
                f'tool name "{RUN_CODE_NAME}" is reserved for the PTC mode presentation '
                "transport and cannot be registered or shadowed"
            )
        return self._layers.effect(
            self.caller(ctx),
            lambda layer: layer.tools.insert(name, _Registration(definition, when)),
            label="tools.register()",
        )

    def _view(
        self, scope: ScopeKey | None
    ) -> tuple[dict[str, Mapping[str, Any]], dict[str, _Registration], set[str]]:
        layers = self._layers.chain_layers(scope)
        own = self._layers.peek(scope)
        inherited = dict(self._layers.global_.tools.entries())
        for layer in layers:
            if layer is own:
                continue
            inherited.update(layer.tools.entries())
        visible: dict[str, Mapping[str, Any]] = {}
        registrations: dict[str, _Registration] = {}
        known = set(inherited)
        for name, registration in inherited.items():
            if all(layer.admits(name) for layer in layers):
                visible[name] = registration.definition
                registrations[name] = registration
        if own is not None:
            for name, registration in own.tools.entries():
                known.add(name)
                visible[name] = registration.definition
                registrations[name] = registration
        return visible, registrations, known

    def restrict(
        self,
        *,
        allow: Sequence[str] | None = None,
        deny: Sequence[str] | None = None,
        ctx: Context | None = None,
    ) -> Callable[[], object]:
        owner = self.caller(ctx)
        scope = scope_of(owner)
        if scope is None:
            raise ValueError("tools.restrict() requires a scoped context")
        if allow is None and deny is None:
            raise ValueError("tools.restrict({}) is a no-op")
        known = self._view(scope)[2]
        unknown = (set(allow or ()) | set(deny or ())) - known
        if unknown:
            raise ValueError(f"tools.restrict() names unknown global tools {sorted(unknown)}")
        compiled = {
            **({"allow": frozenset(allow)} if allow is not None else {}),
            **({"deny": frozenset(deny)} if deny is not None else {}),
        }
        return self._layers.effect(
            owner,
            lambda layer: layer.restrictions.append(compiled),
            label="tools.restrict()",
        )

    def get(
        self,
        name: str,
        scope: ScopeKey | None = None,
        tool_set: ResolvedToolSet | None = None,
    ) -> Mapping[str, Any] | None:
        return (
            self._view(scope)[0].get(name) if tool_set is None else tool_set.view(scope).get(name)
        )

    def executable(
        self,
        name: str,
        scope: ScopeKey | None = None,
        tool_set: ResolvedToolSet | None = None,
    ) -> Mapping[str, Any] | None:
        if tool_set is not None:
            return tool_set.view(scope).get(name)
        visible, registrations, _ = self._view(scope)
        registration = registrations.get(name)
        return visible.get(name) if registration is not None and registration.when is None else None

    def definitions(
        self, scope: ScopeKey | None = None, tool_set: ResolvedToolSet | None = None
    ) -> list[Mapping[str, Any]]:
        return list(
            self._view(scope)[0].values() if tool_set is None else tool_set.view(scope).values()
        )

    def schemas(
        self, scope: ScopeKey | None = None, tool_set: ResolvedToolSet | None = None
    ) -> list[dict[str, Any]]:
        return [project_tool_schema(item) for item in self.definitions(scope, tool_set)]

    async def resolve(self, context: Mapping[str, Any]) -> ResolvedToolSet:
        agent, signal = context["agent"], context["signal"]
        throw_if_aborted(signal)
        visible, registrations, _ = self._view(agent)
        admitted: dict[str, tuple[Mapping[str, Any], _Registration]] = {}
        for name, definition in visible.items():
            registration = registrations[name]
            try:
                enabled = True if registration.when is None else registration.when(context)
                if inspect.isawaitable(enabled):
                    enabled = await enabled
            except Exception as error:
                throw_if_aborted(signal)
                raise ToolActivationError(name, error) from error
            throw_if_aborted(signal)
            if enabled:
                admitted[name] = (definition, registration)
        return ResolvedToolSet(agent, admitted, self)

    def _describe(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        executor = self._select_executor(context.get("scope"))
        result = executor.describe(self.definitions(context.get("scope"), context.get("toolSet")))
        return {
            **result,
            "knownNames": result.get("knownNames", list(self._view(context.get("scope"))[2])),
        }

    def set_executor(self, executor: object, *, ctx: Context | None = None) -> Callable[[], object]:
        owner = self.caller(ctx)
        scope = scope_of(owner)
        key = 0 if scope is None else id(scope)
        if (
            self._root_executor is not None
            if scope is None
            else key in self._executors and self._executors[key] is not None
        ):
            raise ValueError("Only one tool executor may be registered in each scope")

        def setup() -> Callable[[], None]:
            if scope is None:
                self._root_executor = executor
            else:
                self._executors[key] = executor

            def cleanup() -> None:
                if scope is None:
                    self._root_executor = None
                else:
                    self._executors[key] = None

            return cleanup

        return owner.effect(setup, "tools.setExecutor()")

    def _select_executor(self, scope: ScopeKey | None) -> Any:
        from deskagents.core.scope import scope_chain_of

        for key in scope_chain_of(scope):
            if id(key) in self._executors:
                executor = self._executors[id(key)]
                if executor is None:
                    break
                return executor
        if self._root_executor is None:
            raise ToolExecutorUnavailableError
        return self._root_executor

    async def execute(self, input_: Mapping[str, Any]) -> Mapping[str, Any]:
        result = await self._select_executor(input_.get("agent")).execute(input_)
        return cast(Mapping[str, Any], result)

    async def execute_call(
        self, input_: Mapping[str, Any], *, parent: ToolRunContext | None = None
    ) -> Mapping[str, Any]:
        """Execute one parsed call through policy and finalization, without a log batch.

        Composite bodies may supply their execution as ``parent`` to inherit the
        scope, admission, signal and root correlation. Conclusions and deferred
        contexts are returned to the composite; forwarding them is explicit.
        """
        options = dict(input_)
        if parent is not None:
            if parent._settled:
                raise TypeError("cannot dispatch from a settled parent execution")
            for key in ("signal", "agent", "toolSet", "rootCallId"):
                options.setdefault(key, parent.get(key))
            options["parent"] = parent["token"]
        executor = self._select_executor(options.get("agent"))
        execute_call = getattr(executor, "execute_call", None)
        if not callable(execute_call):
            raise TypeError("selected executor does not support single-call dispatch")
        result = await execute_call(options)
        return cast(Mapping[str, Any], result)

    def guard(
        self,
        guard: Callable[[Mapping[str, Any]], str | None],
        *,
        ctx: Context | None = None,
    ) -> Callable[[], object]:
        return self._guards.effect(
            self.caller(ctx),
            lambda layer: layer.guards.append(guard),
            label="tools.guard()",
        )

    def check_guards(self, execution: Mapping[str, Any]) -> str | None:
        for layer in [
            self._guards.global_,
            *self._guards.chain_layers(execution.get("agent")),
        ]:
            for guard in layer.guards.values():
                if (reason := guard(execution)) is not None:
                    return reason
        return None


class NativeToolExecutor:
    def __init__(
        self,
        ctx: Context,
        max_parallel: Callable[[], int] = lambda: 10,
        *,
        source: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.ctx, self.max_parallel = ctx, max_parallel
        self.source = dict(source or {})
        tools = ctx.get("tools")
        if not isinstance(tools, ToolsService):
            raise RuntimeError("NativeToolExecutor requires ToolsService")
        self.tools = tools

    def _lookup(self, name: str, input_: Mapping[str, Any]) -> Mapping[str, Any] | None:
        get = cast(Callable[..., Any], self.source.get("get", self.tools.executable))
        result = get(name, input_.get("agent"), input_.get("toolSet"))
        return cast(Mapping[str, Any] | None, result)

    def _refusal(self, name: str, input_: Mapping[str, Any]) -> BaseException | None:
        reject = self.source.get("reject")
        return (
            None if reject is None else cast(BaseException | None, reject(name, "parent" in input_))
        )

    async def execute_call(self, input_: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run a parsed single call through the same stages used by batch dispatch.

        Unlike a skipped batch slot, a pre-aborted single call snapshots arguments,
        runs its finalizer and publishes one result. Nested calls do not acquire a
        batch worker slot, so a composite works even with a pool limit of one.
        """
        options = {**input_, "signal": input_.get("signal"), "_arguments_are_parsed": True}
        block = {
            "id": input_["callId"],
            "name": input_["name"],
            "arguments": input_.get("arguments", {}),
        }
        prepared = await self._prepare(block, options)
        if prepared.kind == "dispatch":
            prepared = await self._dispatch(prepared.call, options)
        return await self._complete(prepared, options)

    def describe(self, tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return {"schemas": [project_tool_schema(tool, False) for tool in tools]}

    async def _resolve_ask(
        self,
        execution: ToolRunContext,
        decision: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], bool]:
        approval = self.ctx.get("approval")
        request_approval = getattr(approval, "request", None)
        reason = decision.get("reason")
        if not callable(request_approval):
            return (
                {
                    "kind": "deny",
                    "reason": (
                        reason
                        if isinstance(reason, str)
                        else f'tool "{execution["name"]}" requires approval (not yet supported)'
                    ),
                },
                False,
            )
        agent = execution.get("agent")
        if agent is None:
            return (
                {
                    "kind": "deny",
                    "reason": (
                        f'tool "{execution["name"]}" requires approval, but the call has '
                        "no agent to route it through"
                    ),
                },
                False,
            )
        request = {
            "agent": agent,
            "toolName": execution["name"],
            "callId": execution["callId"],
            **({"reason": reason} if isinstance(reason, str) else {}),
            **({"displayReason": decision["displayReason"]} if "displayReason" in decision else {}),
            "signal": execution["signal"],
        }
        outcome = request_approval(request)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        if outcome == "allowed-once":
            return {"kind": "allow"}, False
        if outcome == "rejected":
            return {
                "kind": "deny",
                "reason": f'the user rejected tool "{execution["name"]}"',
            }, False
        if outcome == "cancelled":
            return {
                "kind": "deny",
                "reason": f'approval for tool "{execution["name"]}" was cancelled',
            }, True
        if outcome == "unavailable":
            return {
                "kind": "deny",
                "reason": (
                    f'tool "{execution["name"]}" requires approval, but no approval '
                    "channel is available"
                ),
            }, False
        rendered = stringify_json(outcome) if is_json_value(outcome) else str(outcome)
        raise ValueError(f"unreachable variant in ApprovalOutcome: {rendered}")

    def _execution_mode(self, block: Mapping[str, Any], input_: Mapping[str, Any]) -> str:
        """Only an exact ``True`` from the visible tool's classifier runs in parallel."""
        tool = self._lookup(block["name"], input_)
        classifier = (
            None
            if tool is None or self._refusal(block["name"], input_) is not None
            else tool.get("isConcurrencySafe")
        )
        if not callable(classifier):
            return "exclusive"
        try:
            parallel = classifier(_parse_arguments(block.get("arguments"))) is True
        except Exception:
            return "exclusive"
        return "parallel" if parallel else "exclusive"

    def _create(self, block: Mapping[str, Any], input_: Mapping[str, Any]) -> _Call | _Scheduled:
        signal = input_["signal"]
        agent = input_.get("agent")
        tool = self._lookup(block["name"], input_)
        refusal = None if tool is None else self._refusal(block["name"], input_)
        project = None if tool is None else tool.get("projectContent")
        finalize = (
            None
            if tool is None or (refusal is not None and not _aborted(signal))
            else tool.get("finalizeContent")
        )
        values = {
            "callId": block["id"],
            "rootCallId": input_.get("rootCallId", block["id"]),
            **({"parent": input_["parent"]} if "parent" in input_ else {}),
            **({"schema": input_["schema"]} if "schema" in input_ else {}),
            "name": block["name"],
            "agent": agent,
            "toolSet": input_.get("toolSet"),
            "signal": signal,
            "token": object(),
        }
        arguments = (
            block.get("arguments")
            if input_.get("_arguments_are_parsed")
            else _parse_arguments(block.get("arguments"))
        )
        scope = scope_target(self.tools, agent)
        if not is_json_value(arguments):
            failed = _Call(
                block,
                ToolRunContext({**values, "arguments": None}),
                signal,
                scope,
                finalize=finalize if callable(finalize) else None,
            )
            error = TypeError("tool execution arguments must be losslessly JSON-serializable")
            return _Scheduled("final-result", failed, _failure(error))
        call = _Call(
            block,
            ToolRunContext({**values, "arguments": deep_freeze(snapshot_json_value(arguments))}),
            signal,
            scope,
            project=project if callable(project) else None,
            finalize=finalize if callable(finalize) else None,
        )

        if refusal is not None:
            result = _aborted_before_dispatch_result() if _aborted(signal) else _failure(refusal)
            return _Scheduled("final-result", call, result)
        return call

    def _success_result(
        self,
        call: _Call,
        input_: Mapping[str, Any],
        value: object,
    ) -> dict[str, Any]:
        execution = call.execution
        name = execution["name"]
        tool = self._lookup(name, input_)
        if tool is None:
            raise ToolNotFoundError(name)
        detached = _snapshot_value(name, value)
        violations = validate_json_schema_value(tool["output"]["schema"], detached)
        if violations:
            raise ToolOutputError(name, violations)
        frozen = deep_freeze(detached)
        arguments = execution["arguments"]
        try:
            content = tool["output"]["render"](arguments, frozen)
        except Exception as error:
            raise _projection_error(name, "render", error) from error
        result: dict[str, Any] = {
            "isError": False,
            "value": frozen,
            "content": _snapshot_projection(name, "render", content),
            **({"concludesTurn": True} if execution.concludes_turn else {}),
        }
        presentation_meta = tool["output"].get("presentationMeta")
        if callable(presentation_meta):
            try:
                meta = presentation_meta(arguments, frozen)
            except Exception as error:
                raise _projection_error(name, "presentationMeta", error) from error
            result["meta"] = _snapshot_projection(name, "presentationMeta", meta)
        return result

    def _normalize_dispatch_result(
        self,
        call: _Call,
        input_: Mapping[str, Any],
        candidate: object,
    ) -> dict[str, Any]:
        if not isinstance(candidate, Mapping):
            raise TypeError("tools/execute must return a tool result")
        if candidate.get("isError") is True:
            return {
                "isError": True,
                "error": candidate["error"],
                "content": candidate["content"],
                **({"meta": candidate["meta"]} if "meta" in candidate else {}),
                **(
                    {"additionalContexts": candidate["additionalContexts"]}
                    if "additionalContexts" in candidate
                    else {}
                ),
            }
        if candidate.get("isError") is not False:
            raise TypeError("tools/execute returned a result without isError")
        result = self._success_result(call, input_, candidate.get("value"))
        if "additionalContexts" in candidate:
            result["additionalContexts"] = candidate["additionalContexts"]
        return result

    def _cancellation_result(self, call: _Call, prior: Mapping[str, Any]) -> dict[str, Any]:
        """``ABORTED`` once the body started, otherwise ``ABORTED_BEFORE_DISPATCH``."""
        if call.body_invoked:
            return _aborted_result(prior)
        return _aborted_before_dispatch_result(prior)

    async def _post_execute(
        self,
        call: _Call,
        input_: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        post = self.ctx.waterfall(
            "tools/post-execute",
            call.execution,
            result,
            next_=lambda: {"kind": "accept"},
            filter_=call.scope,
        )
        if inspect.isawaitable(post):
            post = await post
        if not isinstance(post, Mapping):
            raise TypeError("tools/post-execute must return a decision")
        decision_contexts = list(post.get("additionalContexts", ()))
        if post.get("kind") == "block":
            feedback = post["feedback"]
            return {
                "isError": True,
                "error": {"message": _failure_message_from_content(feedback)},
                "content": feedback,
                **({"additionalContexts": decision_contexts} if decision_contexts else {}),
            }
        if post.get("kind") != "accept":
            raise TypeError(f"unknown tools/post-execute decision: {post.get('kind')!r}")
        if "content" in post and "value" in post:
            raise TypeError(
                "tools/post-execute accept decision cannot replace both value and content"
            )
        contexts = [*result.get("additionalContexts", ()), *decision_contexts]
        if "value" in post:
            if result.get("isError") is True:
                raise TypeError("tools/post-execute cannot replace the value of a failed result")
            replaced = self._success_result(call, input_, post["value"])
            if contexts:
                replaced["additionalContexts"] = contexts
            return replaced
        accepted = dict(result)
        if "content" in post:
            accepted["content"] = post["content"]
        if contexts:
            accepted["additionalContexts"] = contexts
        else:
            accepted.pop("additionalContexts", None)
        return accepted

    async def _prepare(self, block: Mapping[str, Any], input_: Mapping[str, Any]) -> _Scheduled:
        """Run pre-execute policy, approval and guards. Cancellation skips an unstarted body."""
        created = self._create(block, input_)
        if isinstance(created, _Scheduled):
            return created
        call, execution = created, created.execution
        if _aborted(call.caller_signal):
            return _Scheduled("final-result", call, _aborted_before_dispatch_result())
        try:
            decision = self.ctx.waterfall(
                "tools/pre-execute",
                execution,
                next_=lambda: {"kind": "allow"},
                filter_=call.scope,
            )
            if inspect.isawaitable(decision):
                decision = await decision
            decision = cast(Mapping[str, Any], decision)
            approval_cancelled = False
            if decision.get("kind") == "ask":
                decision, approval_cancelled = await self._resolve_ask(execution, decision)
            if approval_cancelled and _aborted(call.caller_signal):
                return _Scheduled("post-result", call, _aborted_before_dispatch_result())
            if decision.get("kind") == "cancel":
                return _Scheduled("post-result", call, _aborted_before_dispatch_result())
            allowed = decision.get("kind") == "allow"
            reason = self.tools.check_guards(execution) if allowed else decision.get("reason")
            if reason is not None:
                info = decision.get("info") if decision.get("kind") == "deny" else None
                return _Scheduled("post-result", call, _materialize(_denied(str(reason), info)))
            if _aborted(call.caller_signal):
                return _Scheduled("post-result", call, _aborted_before_dispatch_result())
            return _Scheduled("dispatch", call)
        except Exception as error:
            return _Scheduled("final-result", call, _failure(error))

    async def _dispatch(self, call: _Call, input_: Mapping[str, Any]) -> _Scheduled:
        """Run ``tools/execute`` around the body; tool failures still get post-execute."""
        execution = call.execution
        try:
            canonical: Mapping[str, Any] | None = None

            async def body() -> Mapping[str, Any]:
                nonlocal canonical
                canonical = deep_freeze(await self._dispatch_body(call, input_))
                return canonical

            dispatched = self.ctx.waterfall(
                "tools/execute", execution, next_=body, filter_=call.scope
            )
            if inspect.isawaitable(dispatched):
                dispatched = await dispatched
            result: Mapping[str, Any] = (
                canonical
                if canonical is not None and dispatched is canonical
                else self._normalize_dispatch_result(call, input_, dispatched)
            )
            if execution.additional_contexts:
                result = {
                    **result,
                    "additionalContexts": [
                        *execution.additional_contexts,
                        *result.get("additionalContexts", ()),
                    ],
                }
            if _aborted(call.caller_signal) and not result["isError"]:
                result = self._cancellation_result(call, result)
            return _Scheduled("post-result", call, result)
        except Exception as error:
            return _Scheduled("final-result", call, _failure(error))

    async def _complete(
        self, scheduled: _Scheduled, input_: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Run post-execute when required, then publish the final materialized result."""
        call, result = scheduled.call, scheduled.result
        assert result is not None
        if scheduled.kind == "final-result":
            return self._finish(call, result)
        try:
            project, call.project = call.project, None
            content = None if project is None else project(call.execution, result)
            projected = result if content is None else _materialize({**result, "content": content})
            post = await self._post_execute(call, input_, deep_freeze(projected))
            if _aborted(call.caller_signal) and not post["isError"]:
                post = self._cancellation_result(call, post)
            return self._finish(call, post)
        except Exception as error:
            return self._finish(call, _failure(error))

    def _finish(self, call: _Call, result: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            materialized = _materialize(result)
        except Exception as error:
            materialized = _materialize(_failure(error))
        try:
            final = _materialize(self._final_content(call, materialized))
        except Exception as error:
            final = _materialize(_failure(error))
        execution = call.execution
        execution._settled = True
        emit_contained(
            self.ctx,
            "tools/result",
            execution,
            final,
            filter_=call.scope,
            describe=f'tool "{execution["name"]}" ({execution["callId"]}): tools/result observer',
            log=logger,
        )
        return final

    @staticmethod
    def _final_content(call: _Call, result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply the tool's ``finalizeContent``, which may only replace ``content``."""
        if call.finalize is None:
            return result
        content = call.finalize(call.execution, result)
        return result if content is None else {**result, "content": content}

    async def _dispatch_body(self, call: _Call, input_: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run the body with the caller signal fused into any wrapper replacement.

        Cancellation never abandons a started body; a success that finishes after
        the signal aborted becomes ``ABORTED``.
        """
        execution = call.execution
        wrapper_signal = execution.signal
        fused: Any = (
            wrapper_signal
            if wrapper_signal is call.caller_signal
            else AbortSignal.any([call.caller_signal, wrapper_signal])
        )
        try:
            if _aborted(fused):
                return _aborted_before_dispatch_result()
            execution.signal = fused
            try:
                tool = self._lookup(execution["name"], input_)
                if tool is None:
                    raise ToolNotFoundError(execution["name"])
                call.body_invoked = True
                value = await tool["execute"](execution["arguments"], execution)
                result = self._success_result(call, input_, value)
                return _aborted_result(result) if _aborted(fused) else result
            except Exception as error:
                return _failure(error)
            finally:
                execution.signal = wrapper_signal
        finally:
            if fused is not wrapper_signal:
                fused.detach()

    async def execute(self, input_: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run one step's calls with model-ordered policy and results.

        Exclusive calls are barriers; parallel calls share a bounded pool and are
        reclassified before each start. Every call gets a result: on abort, started
        calls drain and the rest receive ``ABORTED_BEFORE_DISPATCH`` results.
        """
        override = input_.get("maxParallelToolCalls")
        if override is not None and (
            not isinstance(override, int) or isinstance(override, bool) or override < 1
        ):
            raise ValueError("maxParallelToolCalls must be a positive integer")
        if input_.get("log") is not None and input_.get("agent") is None:
            raise ValueError("logged tool batches require an agent")
        limit = override if override is not None else self.max_parallel()
        calls = list(input_["calls"])
        settlement = _BatchSettlement(input_)
        try:
            index, concluded = 0, False
            while index < len(calls):
                mode = self._execution_mode(calls[index], input_)
                group = calls[index:] if mode == "parallel" else calls[index : index + 1]
                consumed, aborted, group_concluded = await self._run_group(
                    group, mode, limit, settlement, input_
                )
                index += consumed
                concluded = concluded or group_concluded
                if aborted:
                    for block in calls[index:]:
                        settlement.skip(block)
                    break
            return settlement.result(concluded)
        except Exception as error:
            raise ToolBatchExecutionError(error, settlement.result()) from error

    async def _run_group(
        self,
        group: Sequence[Mapping[str, Any]],
        mode: str,
        limit: int,
        settlement: _BatchSettlement,
        input_: Mapping[str, Any],
    ) -> tuple[int, bool, bool]:
        """Run one exclusive call or one parallel pool; returns consumed, aborted, concluded."""
        signal = input_["signal"]
        slots: list[_Scheduled | None] = [None] * len(group)
        call_seqs: list[int | None] = [None] * len(group)
        in_flight: dict[int, asyncio.Task[None]] = {}
        failures: list[BaseException] = []
        started = committed = 0
        aborted, concluded = _aborted(signal), False

        def raise_failure() -> None:
            if failures:
                raise failures[0]

        async def dispatch(index: int, scheduled: _Scheduled) -> None:
            try:
                slots[index] = await self._dispatch(scheduled.call, input_)
            except Exception as error:
                failures.append(error)

        async def commit_ready() -> None:
            nonlocal committed, concluded
            while committed < len(group):
                slot = slots[committed]
                if slot is None:
                    return
                result = await self._complete(slot, input_)
                settlement.commit(group[committed], result, call_seqs[committed])
                concluded = concluded or result.get("concludesTurn") is True
                committed += 1

        async def fill_pool() -> None:
            nonlocal started, aborted
            while not aborted and started < len(group) and len(in_flight) < limit:
                if (
                    started > 0
                    and mode == "parallel"
                    and self._execution_mode(group[started], input_) != "parallel"
                ):
                    return
                index = started
                call_seqs[index] = settlement.start(group[index])
                started += 1
                prepared = await self._prepare(group[index], input_)
                if prepared.kind == "dispatch":
                    in_flight[index] = asyncio.ensure_future(dispatch(index, prepared))
                else:
                    slots[index] = prepared
                raise_failure()
                await commit_ready()
                aborted = aborted or _aborted(signal)

        try:
            await fill_pool()
            while in_flight:
                done, _ = await asyncio.wait(
                    in_flight.values(), return_when=asyncio.FIRST_COMPLETED
                )
                for index in [index for index, task in in_flight.items() if task in done]:
                    del in_flight[index]
                raise_failure()
                await commit_ready()
                aborted = aborted or _aborted(signal)
                await fill_pool()
        except asyncio.CancelledError:
            for task in in_flight.values():
                task.cancel()
            await asyncio.gather(*in_flight.values(), return_exceptions=True)
            raise
        except Exception:
            await asyncio.gather(*in_flight.values(), return_exceptions=True)
            raise
        if aborted:
            for block in group[started:]:
                settlement.skip(block)
            return len(group), True, concluded
        if committed != started:
            raise RuntimeError("tool-call scheduler: uncommitted settled calls")
        return started, False, concluded


class _BatchSettlement:
    """Committed results and contexts, and their durable events for a logged batch."""

    def __init__(self, input_: Mapping[str, Any]) -> None:
        self._input = input_
        self._results: list[Mapping[str, Any]] = []
        self._contexts: list[Mapping[str, Any]] = []

    def start(self, block: Mapping[str, Any]) -> int | None:
        log = self._input.get("log")
        if log is None:
            return None
        event = self._input["agent"].session.append(
            "tool/call",
            {
                **log,
                "callId": block["id"],
                "name": block["name"],
                "arguments": block.get("arguments", ""),
            },
        )
        return cast(int, event["seq"])

    def commit(
        self, block: Mapping[str, Any], result: Mapping[str, Any], call_seq: int | None
    ) -> None:
        log = self._input.get("log")
        if log is not None:
            info = result.get("error", {}).get("info") if result["isError"] else None
            self._input["agent"].session.append(
                "tool/result",
                {
                    **log,
                    "message": create_tool_result_message(
                        call_id=block["id"],
                        content=result["content"],
                        is_error=result["isError"],
                    ),
                    **({"error": info} if info else {}),
                    **({"meta": result["meta"]} if "meta" in result else {}),
                },
                surface_op="append",
                source_event_seqs=[call_seq],
            )
        self._results.append({"callId": block["id"], "result": result})
        self._contexts.extend(result.get("additionalContexts", ()))

    def skip(self, block: Mapping[str, Any]) -> None:
        """Record a synthetic result for a call the abort kept from starting."""
        self.commit(block, deep_freeze(_aborted_before_dispatch_result()), self.start(block))

    def result(self, concluded: bool | None = None) -> dict[str, Any]:
        if concluded is None:
            concluded = any(item["result"].get("concludesTurn") is True for item in self._results)
        return {
            "results": list(self._results),
            "additionalContexts": list(self._contexts),
            "concluded": concluded,
        }


def install_native_tool_executor(ctx: Context, config: object = None) -> NativeToolExecutor:
    del config
    tools = ctx.get("tools")
    if not isinstance(tools, ToolsService):
        raise RuntimeError("ToolsService is not installed")
    executor = NativeToolExecutor(ctx)
    tools.set_executor(executor, ctx=ctx)
    return executor


__all__ = [
    "JsonSchemaError",
    "NativeToolExecutor",
    "RUN_CODE_NAME",
    "ResolvedToolSet",
    "TOOL_ABORTED",
    "TOOL_ABORTED_BEFORE_DISPATCH",
    "ToolActivationError",
    "ToolArgsError",
    "ToolBatchExecutionError",
    "ToolExecutorUnavailableError",
    "ToolNotFoundError",
    "ToolOutputError",
    "ToolRunContext",
    "ToolsService",
    "assert_object_json_schema",
    "assert_supported_json_schema",
    "check_supported_json_schema",
    "define_tool",
    "install_native_tool_executor",
    "parameter_schema_spec_to_json_schema",
    "project_tool_schema",
    "validate_args",
    "validate_json_schema_value",
    "value_schema_spec_to_json_schema",
]
