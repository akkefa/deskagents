"""Core provider registry and Cordis-waterfall LLM streaming service.

This module ports the pinned upstream routing, preparation, registration
ownership, and adapter-failure boundary. Attachment-specific file/image request
projection is deliberately omitted: attachments are an external plugin in the
Python architecture. Completed attachment blocks remain valid stream/message
vocabulary and can be projected by that plugin before dispatch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping, Sequence
from typing import Any, cast

from deepseek_cordis import Context, Service, current_context

from deskagents._cordis import emit_contained
from deskagents.util.values import deep_freeze, is_safe_integer, snapshot_json_value, thaw_json

from .content import project_tool_updates
from .errors import LlmError, normalize_llm_failure
from .retry_policy import resolve_retry_policy

logger = logging.getLogger(__name__)
_CONFIG_FIELDS = ("provider", "model", "reasoningEffort", "temperature", "maxTokens", "stop")


def _config_of(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value[field] for field in _CONFIG_FIELDS if field in value}


def _config_equals(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in _CONFIG_FIELDS)


def _aborted(signal: object) -> bool:
    return bool(getattr(signal, "aborted", False))


class LlmAdapter(ABC):
    """Provider-wire adapter for the harness message and stream vocabulary."""

    def provider_info(self, provider: str) -> Mapping[str, Any]:
        return {"id": provider, "name": provider}

    def provider_retry_policy(self, provider: str) -> Mapping[str, Any] | None:
        del provider
        return None

    async def list_models(self, provider: str) -> Sequence[Mapping[str, Any]]:
        del provider
        return ()

    async def resolve_model(
        self, provider: str, model: str, signal: object = None
    ) -> Mapping[str, Any]:
        del signal
        return {"provider": provider, "id": model, "name": model}

    async def prepare_call(
        self, provider: str, model: str, signal: object = None
    ) -> PreparedAdapterCall:
        return PreparedAdapterCall(
            await self.resolve_model(provider, model, signal),
            self.stream,
        )

    @abstractmethod
    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        """Stream raw provider-neutral chunks."""


class PreparedAdapterCall:
    def __init__(
        self,
        model: Mapping[str, Any],
        stream: Callable[[Mapping[str, Any]], AsyncIterable[Mapping[str, Any]]],
    ) -> None:
        self.model = model
        self.stream = stream


class AdapterRegistrationHandle:
    """Disposable route registration with atomic replacement."""

    def __init__(
        self,
        dispose: Callable[[], object],
        replace: Callable[[Sequence[str]], None],
    ) -> None:
        self._dispose = dispose
        self._replace = replace

    def __call__(self) -> None:
        self._dispose()

    def replace(self, providers: Sequence[str]) -> None:
        self._replace(providers)


class PreparedLlmCall:
    """One registration-bound call that may be dispatched exactly once."""

    def __init__(
        self,
        runtime: LlmRuntime,
        registration: _Registration,
        adapter_call: PreparedAdapterCall,
        config: Mapping[str, Any],
        retry_policy: Mapping[str, Any],
        model_info: Mapping[str, Any],
        adapter_defaults: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.retry_policy = retry_policy
        self.adapter_defaults = adapter_defaults
        self.context = model_info.get("context")
        self.input_modalities = model_info.get("inputModalities")
        self.system_prompt_update = model_info.get("systemPromptUpdate")
        self.tool_update = model_info.get("toolUpdate")
        self._runtime = runtime
        self._registration = registration
        self._adapter_call = adapter_call
        self._model_info = model_info
        self._dispatched = False

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        if self._dispatched:
            raise LlmError(
                "a prepared LLM call can only be dispatched once", "INVALID_PREPARED_CALL"
            )
        if not _config_equals(options, self.config):
            raise LlmError(
                "prepared LLM call config changed before adapter dispatch",
                "INVALID_PREPARED_CALL",
            )
        self._dispatched = True
        return self._runtime._stream_with_registration(
            options,
            _PreparedDispatch(
                self._registration,
                self.config,
                self._model_info,
                self._adapter_call.stream,
            ),
        )


class _Registration:
    def __init__(
        self,
        adapter: LlmAdapter,
        provider: Mapping[str, Any],
        retry_policy: Mapping[str, Any],
    ) -> None:
        self.adapter = adapter
        self.provider = provider
        self.retry_policy = retry_policy


class _PreparedDispatch:
    def __init__(
        self,
        registration: _Registration,
        config: Mapping[str, Any],
        model_info: Mapping[str, Any],
        dispatch_: Callable[[Mapping[str, Any]], AsyncIterable[Mapping[str, Any]]],
    ) -> None:
        self.registration = registration
        self.config = config
        self.model_info = model_info
        self.dispatch = dispatch_


class LlmRuntime(Service):
    """Adapter registry and waterfall-interceptable model streaming service."""

    name = "llm"

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, config)
        self._adapters: dict[str, _Registration] = {}

    def caller(self, ctx: Context | None = None) -> Context:
        candidate = ctx if ctx is not None else current_context()
        if candidate is None or candidate.root is not self.ctx.root:
            return self.ctx
        return candidate

    def _emit_updated(self) -> None:
        emit_contained(self.ctx, "llm/adapters-updated", log=logger)

    def models_updated(self, provider: str) -> None:
        """Publish that one provider's advertised model catalog changed."""
        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be a non-empty string")
        emit_contained(
            self.ctx,
            "llm/models-updated",
            provider,
            describe=f'llm/models-updated observer for provider "{provider}"',
            log=logger,
        )

    def _prepare_routes(
        self,
        providers: Sequence[str],
        adapter: LlmAdapter,
        owned: set[str],
    ) -> list[_Registration]:
        unique: set[str] = set()
        registrations: list[_Registration] = []
        for provider in providers:
            if not provider:
                raise LlmError("adapter provider names must be non-empty", "INVALID_ADAPTER")
            if provider in unique or (provider in self._adapters and provider not in owned):
                raise LlmError(
                    f'an adapter for provider "{provider}" is already registered',
                    "DUPLICATE_ADAPTER",
                )
            info = adapter.provider_info(provider)
            if (
                info.get("id") != provider
                or not isinstance(info.get("name"), str)
                or not info["name"]
            ):
                raise LlmError(
                    f'adapter metadata for provider "{provider}" must preserve its id and '
                    "have a non-empty name",
                    "INVALID_ADAPTER",
                )
            policy = adapter.provider_retry_policy(provider) or resolve_retry_policy(
                None, f'llm: provider "{provider}" retryPolicy'
            )
            registrations.append(_Registration(adapter, deep_freeze(dict(info)), policy))
            unique.add(provider)
        return registrations

    def _commit(self, owned: set[str], registrations: Sequence[_Registration]) -> None:
        for provider in owned:
            self._adapters.pop(provider, None)
        owned.clear()
        for registration in registrations:
            provider = cast(str, registration.provider["id"])
            self._adapters[provider] = registration
            owned.add(provider)
        self._emit_updated()

    def register_adapter(
        self,
        providers: Sequence[str],
        adapter: LlmAdapter,
        *,
        ctx: Context | None = None,
    ) -> AdapterRegistrationHandle:
        if not providers:
            raise LlmError("an adapter must register at least one provider", "INVALID_ADAPTER")
        owned: set[str] = set()
        released = False

        def setup() -> Callable[[], None]:
            self._commit(owned, self._prepare_routes(providers, adapter, owned))

            def cleanup() -> None:
                nonlocal released
                released = True
                for provider in owned:
                    self._adapters.pop(provider, None)
                owned.clear()
                self._emit_updated()

            return cleanup

        effect = self.caller(ctx).effect(setup, "llm.registerAdapter()")

        def replace(next_providers: Sequence[str]) -> None:
            if released:
                raise LlmError(
                    "a disposed adapter registration cannot replace its routes",
                    "REGISTRATION_DISPOSED",
                )
            self._commit(owned, self._prepare_routes(next_providers, adapter, owned))

        return AdapterRegistrationHandle(effect, replace)

    def list_providers(self) -> list[dict[str, Any]]:
        return [dict(registration.provider) for registration in self._adapters.values()]

    def _get_registration(self, provider: str) -> _Registration:
        registration = self._adapters.get(provider)
        if registration is None:
            raise LlmError(f'no adapter registered for provider "{provider}"', "NO_ADAPTER")
        return registration

    def provider_retry_policy(self, provider: str) -> Mapping[str, Any]:
        return self._get_registration(provider).retry_policy

    async def list_models(self, provider: str) -> list[dict[str, Any]]:
        models = await self._get_registration(provider).adapter.list_models(provider)
        seen: set[str] = set()
        detached: list[dict[str, Any]] = []
        for model in models:
            model_id = model.get("id")
            if (
                model.get("provider") != provider
                or not isinstance(model_id, str)
                or not model_id
                or model_id in seen
                or not isinstance(model.get("name"), str)
                or not model["name"]
                or ("description" in model and not isinstance(model["description"], str))
            ):
                raise LlmError(
                    "adapter returned invalid or duplicate model metadata "
                    f'for provider "{provider}"',
                    "INVALID_CATALOG",
                )
            seen.add(model_id)
            snapshot = snapshot_json_value(dict(model))
            if not isinstance(snapshot, dict):
                raise LlmError("adapter returned non-JSON model metadata", "INVALID_CATALOG")
            detached.append(snapshot)
        return detached

    def _normalize_model(
        self, registration: _Registration, model: str, resolved: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        provider = registration.provider["id"]
        if (
            resolved.get("provider") != provider
            or resolved.get("id") != model
            or not isinstance(resolved.get("name"), str)
            or not resolved["name"]
            or ("description" in resolved and not isinstance(resolved["description"], str))
        ):
            raise LlmError(
                f'adapter returned invalid exact model metadata for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_INFO",
            )
        context = resolved.get("context")
        if context is not None and (
            not isinstance(context, Mapping)
            or not isinstance(context.get("contextWindow"), int)
            or isinstance(context.get("contextWindow"), bool)
            or context["contextWindow"] <= 0
        ):
            raise LlmError(
                f'adapter returned invalid context metadata for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_CONTEXT",
            )
        maximum = resolved.get("defaultMaxTokens")
        if maximum is not None and (
            not is_safe_integer(maximum) or isinstance(maximum, bool) or maximum <= 0
        ):
            raise LlmError(
                f'adapter returned invalid default maxTokens for provider "{provider}" '
                f'model "{model}"',
                "INVALID_MODEL_MAX_TOKENS",
            )
        system_mode = resolved.get("systemPromptUpdate")
        tool_mode = resolved.get("toolUpdate")
        if system_mode is not None and system_mode != "in-history":
            raise LlmError(
                "adapter returned invalid system prompt update mode", "INVALID_MODEL_INFO"
            )
        if tool_mode is not None and tool_mode not in ("in-history", "addition-only"):
            raise LlmError("adapter returned invalid tool update mode", "INVALID_MODEL_INFO")
        reasoning = resolved.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, Mapping) or not isinstance(
                reasoning.get("efforts"), Sequence
            ):
                raise LlmError(
                    "adapter returned invalid reasoning metadata", "INVALID_MODEL_REASONING"
                )
            efforts = reasoning["efforts"]
            seen: set[str] = set()
            for effort in efforts:
                if (
                    not isinstance(effort, Mapping)
                    or not isinstance(effort.get("id"), str)
                    or not effort["id"]
                    or effort["id"] in seen
                    or not isinstance(effort.get("name"), str)
                    or not effort["name"]
                    or ("description" in effort and not isinstance(effort["description"], str))
                ):
                    raise LlmError(
                        "adapter returned invalid or duplicate reasoning effort metadata",
                        "INVALID_MODEL_REASONING",
                    )
                seen.add(effort["id"])
            if not efforts or (
                reasoning.get("defaultEffort") is not None
                and reasoning["defaultEffort"] not in seen
            ):
                raise LlmError(
                    "adapter returned invalid reasoning metadata", "INVALID_MODEL_REASONING"
                )
        snapshot = snapshot_json_value(dict(resolved), freeze=True)
        if not isinstance(snapshot, Mapping):
            raise LlmError("adapter returned non-JSON exact model metadata", "INVALID_MODEL_INFO")
        return snapshot

    async def resolve_model_info(
        self, provider: str, model: str, signal: object = None
    ) -> Mapping[str, Any]:
        registration = self._get_registration(provider)
        resolved = await registration.adapter.resolve_model(provider, model, signal)
        return self._normalize_model(registration, model, resolved)

    def _resolve_config(
        self, config: Mapping[str, Any], info: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        resolved = dict(config)
        if "maxTokens" not in resolved and "defaultMaxTokens" in info:
            resolved["maxTokens"] = info["defaultMaxTokens"]
        reasoning = info.get("reasoning")
        requested = resolved.get("reasoningEffort")
        if reasoning is None:
            if requested is not None:
                raise LlmError(
                    f'provider "{config["provider"]}" model "{config["model"]}" does not '
                    f'support reasoning effort "{requested}"',
                    "UNSUPPORTED_REASONING_EFFORT",
                )
        else:
            effective = requested if requested is not None else reasoning.get("defaultEffort")
            if effective is not None:
                if not any(item["id"] == effective for item in reasoning["efforts"]):
                    raise LlmError(
                        f'provider "{config["provider"]}" model "{config["model"]}" does not '
                        f'support reasoning effort "{effective}"',
                        "UNSUPPORTED_REASONING_EFFORT",
                    )
                resolved["reasoningEffort"] = effective
        return resolved

    async def resolve_call_config(
        self, config: Mapping[str, Any], signal: object = None
    ) -> Mapping[str, Any]:
        info = await self.resolve_model_info(config["provider"], config["model"], signal)
        resolved = self._resolve_config(config, info)
        return config if resolved == config else resolved

    async def prepare_call(
        self,
        config: Mapping[str, Any],
        signal: object = None,
        configure: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ) -> PreparedLlmCall:
        if _aborted(signal):
            raise LlmError("LLM call preparation was aborted", "ABORTED")
        provider, model = config["provider"], config["model"]
        registration = self._get_registration(provider)
        adapter_call = await registration.adapter.prepare_call(provider, model, signal)
        if _aborted(signal):
            raise LlmError("LLM call preparation was aborted", "ABORTED")
        info = self._normalize_model(registration, model, adapter_call.model)
        proposed: Mapping[str, Any] = config
        if configure is not None:
            controls = {
                key: value for key, value in config.items() if key not in ("provider", "model")
            }
            configured = configure(deep_freeze(controls), info)
            proposed = {**configured, "provider": provider, "model": model}
        if _aborted(signal):
            raise LlmError("LLM call preparation was aborted", "ABORTED")
        resolved = self._resolve_config(proposed, info)
        frozen_config = deep_freeze(thaw_json(resolved))
        defaults: dict[str, Any] = {}
        if "reasoningEffort" not in proposed and "reasoningEffort" in frozen_config:
            defaults["reasoningEffort"] = True
        if "maxTokens" not in proposed and "maxTokens" in frozen_config:
            defaults["maxTokens"] = True
        return PreparedLlmCall(
            self,
            registration,
            adapter_call,
            frozen_config,
            registration.retry_policy,
            info,
            deep_freeze(defaults),
        )

    def _for_adapter(self, options: Mapping[str, Any], adapter: LlmAdapter) -> Mapping[str, Any]:
        changed = False
        messages: list[Any] = []
        for message in options.get("messages", ()):
            if message.get("role") != "assistant":
                messages.append(message)
                continue
            source = message.get("source", {})
            if "replayState" not in source:
                messages.append(message)
                continue
            owner = self._adapters.get(source.get("provider"))
            if owner is not None and owner.adapter is adapter:
                messages.append(message)
                continue
            changed = True
            messages.append(
                deep_freeze(
                    {
                        **message,
                        "source": {
                            "kind": "model",
                            "provider": source.get("provider"),
                            "model": source.get("model"),
                        },
                    }
                )
            )
        return {**options, "messages": messages} if changed else options

    async def _adapter_stream(
        self, options: Mapping[str, Any], prepared: _PreparedDispatch | None
    ) -> AsyncIterator[Mapping[str, Any]]:
        try:
            registration = (
                prepared.registration
                if prepared is not None
                else self._get_registration(options["provider"])
            )
            adapter = registration.adapter
            if prepared is None:
                adapter_call = await adapter.prepare_call(
                    options["provider"], options["model"], options.get("signal")
                )
                info = self._normalize_model(registration, options["model"], adapter_call.model)
                resolved_config = self._resolve_config(_config_of(options), info)
                call = adapter_call.stream
            else:
                resolved_config = prepared.config
                call = prepared.dispatch
                info = prepared.model_info
            if prepared is not None and not _config_equals(options, resolved_config):
                raise LlmError(
                    "prepared LLM call config changed before adapter dispatch",
                    "INVALID_PREPARED_CALL",
                )
            resolved_options = (
                options
                if _config_equals(options, resolved_config)
                else {**options, **resolved_config}
            )
            projected = project_tool_updates(
                resolved_options.get("messages", ()),
                resolved_options.get("tools"),
                info.get("toolUpdate"),
                resolved_options.get("toolHistory"),
            )
            projected_options = resolved_options
            if projected["messages"] is not resolved_options.get("messages") or projected[
                "tools"
            ] is not resolved_options.get("tools"):
                projected_options = {
                    **resolved_options,
                    "messages": projected["messages"],
                }
                if projected["tools"] is not None:
                    projected_options["tools"] = projected["tools"]
            stream = call(self._for_adapter(projected_options, adapter))
            async for chunk in stream:
                yield chunk
        except Exception as error:
            failure = normalize_llm_failure(error)
            kind = (
                "aborted"
                if _aborted(options.get("signal")) or failure["code"] == "ABORTED"
                else "error"
            )
            yield {"type": "finish", "reason": {"kind": kind, "failure": failure}}

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        return self._stream_with_registration(options, None)

    def _stream_with_registration(
        self, options: Mapping[str, Any], prepared: _PreparedDispatch | None
    ) -> AsyncIterable[Mapping[str, Any]]:
        result = self.ctx.waterfall(
            "llm/stream",
            options,
            next_=lambda: self._adapter_stream(options, prepared),
        )
        return cast(AsyncIterable[Mapping[str, Any]], result)
