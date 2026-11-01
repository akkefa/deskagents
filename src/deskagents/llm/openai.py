"""OpenAI-compatible Chat Completions adapter.

This is a Python-native provider on the ``deskagents.llm`` seam. It supports OpenAI
Chat Completions-compatible endpoints and Azure OpenAI deployments.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from typing import Any

import httpx
from deepseek_cordis import Context

from deskagents.util.values import snapshot_json_value

from .api_key import assert_usable_api_key
from .attribution import attribution_headers
from .errors import (
    LlmError,
    is_context_window_exceeded_error,
    is_quota_exceeded_error,
)
from .runtime import LlmAdapter, LlmRuntime


def _text_content(blocks: Sequence[Mapping[str, Any]]) -> str:
    unsupported = [block["type"] for block in blocks if block["type"] != "text"]
    if unsupported:
        raise LlmError(
            f"OpenAI Chat Completions cannot represent content {unsupported[0]}",
            "UNSUPPORTED_CONTENT",
        )
    return "".join(str(block["text"]) for block in blocks)


def serialize_request(options: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize provider-neutral harness messages to Chat Completions JSON."""
    messages: list[dict[str, Any]] = []
    for message in options.get("messages", ()):
        role = message["role"]
        if role in ("system", "developer", "user"):
            messages.append({"role": role, "content": _text_content(message["content"])})
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message["toolCallId"],
                    "content": _text_content(message["content"]),
                }
            )
        elif role == "assistant":
            text = "".join(
                str(block["text"]) for block in message["content"] if block["type"] == "text"
            )
            reasoning = "".join(
                str(block["text"]) for block in message["content"] if block["type"] == "reasoning"
            )
            calls = [
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": block["arguments"],
                    },
                }
                for block in message["content"]
                if block["type"] == "tool-call"
            ]
            unsupported = [
                block["type"]
                for block in message["content"]
                if block["type"] not in ("text", "reasoning", "tool-call")
            ]
            if unsupported:
                raise LlmError(
                    f"OpenAI Chat Completions cannot represent content {unsupported[0]}",
                    "UNSUPPORTED_CONTENT",
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    **({"reasoning_content": reasoning} if reasoning else {}),
                    **({"tool_calls": calls} if calls else {}),
                }
            )
        else:
            raise LlmError(f'unsupported message role "{role}"', "UNSUPPORTED_CONTENT")
    body: dict[str, Any] = {
        "model": options["model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for source, target in (
        ("maxTokens", "max_tokens"),
        ("temperature", "temperature"),
        ("stop", "stop"),
        ("reasoningEffort", "reasoning_effort"),
    ):
        if source in options:
            body[target] = options[source]
    if "tools" in options:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in options["tools"]
        ]
    detached = snapshot_json_value(body)
    if not isinstance(detached, dict):
        raise LlmError("OpenAI request is not lossless JSON", "INVALID_REQUEST")
    return detached


def _finish_reason(value: object) -> dict[str, str]:
    if value in ("stop", "content_filter"):
        return {"kind": "stop"}
    if value in ("tool_calls", "function_call"):
        return {"kind": "tool-calls"}
    if value == "length":
        return {"kind": "max-tokens"}
    raise LlmError(f"unsupported OpenAI finish reason {value!r}", "MALFORMED_RESPONSE")


async def translate_sse(lines: AsyncIterable[str]) -> AsyncIterator[Mapping[str, Any]]:
    """Translate OpenAI data frames into the harness stream vocabulary."""
    next_index = 0
    text_index: int | None = None
    reasoning_index: int | None = None
    tool_indexes: dict[int, int] = {}
    tool_arguments: dict[int, str] = {}
    seen_content = False
    finish: dict[str, str] | None = None
    usage: Mapping[str, Any] | None = None
    async for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise LlmError("OpenAI SSE contains invalid JSON", "MALFORMED_RESPONSE") from error
        raw_usage = event.get("usage")
        if raw_usage is not None:
            completion_details = raw_usage.get("completion_tokens_details") or {}
            prompt_details = raw_usage.get("prompt_tokens_details") or {}
            usage = {
                "inputTokens": raw_usage.get("prompt_tokens", 0),
                "outputTokens": raw_usage.get("completion_tokens", 0),
                "totalTokens": raw_usage.get("total_tokens", 0),
                **(
                    {"reasoningTokens": completion_details["reasoning_tokens"]}
                    if "reasoning_tokens" in completion_details
                    else {}
                ),
                **(
                    {"cacheReadTokens": prompt_details["cached_tokens"]}
                    if "cached_tokens" in prompt_details
                    else {}
                ),
            }
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise LlmError("OpenAI response choices must be an array", "MALFORMED_RESPONSE")
        for choice in choices:
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            if reasoning:
                if reasoning_index is None:
                    reasoning_index, next_index = next_index, next_index + 1
                    yield {
                        "type": "block-start",
                        "index": reasoning_index,
                        "blockType": "reasoning",
                    }
                seen_content = True
                yield {
                    "type": "reasoning-delta",
                    "index": reasoning_index,
                    "text": reasoning,
                }
            content = delta.get("content")
            if content:
                if text_index is None:
                    text_index, next_index = next_index, next_index + 1
                    yield {"type": "block-start", "index": text_index, "blockType": "text"}
                seen_content = True
                yield {"type": "text-delta", "index": text_index, "text": content}
            for tool in delta.get("tool_calls") or ():
                wire_index = int(tool.get("index", 0))
                if wire_index not in tool_indexes:
                    tool_indexes[wire_index], next_index = next_index, next_index + 1
                    tool_arguments[wire_index] = ""
                    yield {
                        "type": "block-start",
                        "index": tool_indexes[wire_index],
                        "blockType": "tool-call",
                    }
                function = tool.get("function") or {}
                arguments_delta = function.get("arguments") or ""
                tool_arguments[wire_index] += arguments_delta
                seen_content = True
                yield {
                    "type": "tool-call-delta",
                    "index": tool_indexes[wire_index],
                    **({"id": tool["id"]} if tool.get("id") else {}),
                    **({"name": function["name"]} if function.get("name") else {}),
                    "argumentsDelta": arguments_delta,
                }
            if choice.get("finish_reason") is not None:
                finish = _finish_reason(choice["finish_reason"])
    if finish is None:
        raise LlmError("OpenAI stream ended without a finish reason", "STREAM_CLOSED")
    if not seen_content and finish["kind"] == "stop":
        raise LlmError("OpenAI returned no content", "EMPTY_RESPONSE")
    if finish["kind"] != "max-tokens":
        for arguments in tool_arguments.values():
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise LlmError(
                    "OpenAI tool arguments are invalid JSON",
                    "MALFORMED_RESPONSE",
                ) from error
            if not isinstance(parsed, Mapping):
                raise LlmError(
                    "OpenAI tool arguments must be a JSON object",
                    "MALFORMED_RESPONSE",
                )
    if usage is not None:
        yield {"type": "usage", "usage": usage}
    yield {"type": "finish", "reason": finish}


class OpenAIAdapter(LlmAdapter):
    """HTTPX-backed OpenAI-compatible adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        headers: Mapping[str, str] | None = None,
        models: Sequence[Mapping[str, Any]] = (),
        client: httpx.AsyncClient | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str = "2024-10-21",
        provider_name: str = "OpenAI",
    ) -> None:
        self.api_key = assert_usable_api_key(api_key, "deskagents.llm.openai", "api_key")
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.models = tuple(models)
        self.client = client
        self.azure_endpoint = None if azure_endpoint is None else azure_endpoint.rstrip("/")
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.provider_name = provider_name

    def provider_info(self, provider: str) -> Mapping[str, Any]:
        return {"id": provider, "name": self.provider_name}

    async def list_models(self, provider: str) -> Sequence[Mapping[str, Any]]:
        return tuple({**model, "provider": provider} for model in self.models)

    async def resolve_model(
        self, provider: str, model: str, signal: object = None
    ) -> Mapping[str, Any]:
        del signal
        declared = next((item for item in self.models if item.get("id") == model), None)
        return {
            "provider": provider,
            "id": model,
            "name": model if declared is None else declared.get("name", model),
            **({} if declared is None else dict(declared)),
        }

    def _request(self, options: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
        if self.azure_endpoint is not None:
            deployment = self.azure_deployment or str(options["model"])
            return (
                f"{self.azure_endpoint}/openai/deployments/{deployment}/chat/completions"
                f"?api-version={self.api_version}",
                {**self.headers, "api-key": self.api_key, **attribution_headers()},
            )
        return (
            f"{self.base_url}/chat/completions",
            {
                **self.headers,
                "authorization": f"Bearer {self.api_key}",
                **attribution_headers(),
            },
        )

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        async def generate() -> AsyncIterator[Mapping[str, Any]]:
            url, headers = self._request(options)
            own_client = self.client is None
            client = self.client or httpx.AsyncClient(timeout=None)
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers={"accept": "text/event-stream", **headers},
                    json=serialize_request(options),
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode(errors="replace")
                        code = (
                            "CONTEXT_WINDOW_EXCEEDED"
                            if is_context_window_exceeded_error(detail)
                            else "QUOTA"
                            if is_quota_exceeded_error(detail)
                            else "INVALID_CREDENTIAL"
                            if response.status_code in (401, 403)
                            else "RATE_LIMITED"
                            if response.status_code == 429
                            else "TRANSPORT"
                        )
                        raise LlmError(
                            f"OpenAI request failed ({response.status_code}): {detail}",
                            code,
                            status=response.status_code,
                            request_id=response.headers.get("x-request-id"),
                        )
                    async for chunk in translate_sse(response.aiter_lines()):
                        yield chunk
            except httpx.HTTPError as error:
                raise LlmError("OpenAI transport failed", "TRANSPORT", cause=error) from error
            finally:
                if own_client:
                    await client.aclose()

        return generate()


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> Any:
    """Mount one configured OpenAI-compatible provider route."""
    config = config or {}
    llm = ctx.get("llm")
    if not isinstance(llm, LlmRuntime):
        raise RuntimeError("deskagents.llm.openai requires LlmRuntime")
    ref = str(config.get("apiKeyEnv", "OPENAI_API_KEY"))
    raw = config.get("apiKey", os.environ.get(ref))
    if not isinstance(raw, str) or not raw.strip():
        raise LlmError(
            f"deskagents.llm.openai: set config.apiKey or export {ref}",
            "MISSING_CREDENTIAL",
        )
    adapter = OpenAIAdapter(
        api_key=raw,
        base_url=str(config.get("baseURL", "https://api.openai.com/v1")),
        headers=config.get("headers"),
        models=config.get("models", ()),
        azure_endpoint=config.get("azureEndpoint"),
        azure_deployment=config.get("azureDeployment"),
        api_version=str(config.get("apiVersion", "2024-10-21")),
        provider_name=str(config.get("displayName", "OpenAI")),
    )
    return llm.register_adapter(
        [str(config.get("provider", "openai"))],
        adapter,
        ctx=ctx,
    )


__all__ = ["OpenAIAdapter", "apply", "serialize_request", "translate_sse"]
