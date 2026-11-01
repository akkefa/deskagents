"""DeepSeek Messages API adapter.

Upstream: packages/llm/llm-deepseek at ce7073d.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from deskagents.llm.api_key import assert_usable_api_key
from deskagents.llm.attribution import attribution_headers
from deskagents.llm.deepseek_llm_api_extensions import DeepSeekLlmApiExtensionRegistry
from deskagents.llm.errors import (
    LlmError,
    is_context_window_exceeded_error,
    is_quota_exceeded_error,
)
from deskagents.llm.runtime import LlmAdapter
from deskagents.util.values import snapshot_json_value


def _object(value: object, detail: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LlmError(f"DeepSeek Messages {detail}", "MALFORMED_RESPONSE")
    return value


def _tool_input(raw: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, Mapping) else {}


def serialize_request(
    options: Mapping[str, Any],
    *,
    max_tokens: int = 8192,
    thinking: str = "enabled",
) -> dict[str, Any]:
    """Serialize harness messages to the DeepSeek Messages wire format."""
    system: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in options.get("messages", ()):
        role = message["role"]
        if role == "system":
            for block in message["content"]:
                if block["type"] != "text":
                    raise LlmError(
                        "DeepSeek Messages requires text system content",
                        "UNSUPPORTED_CONTENT",
                    )
                system.append(block["text"])
            continue
        content: list[dict[str, Any]] = []
        if role == "assistant":
            for block in message["content"]:
                if block["type"] == "text" and block["text"]:
                    content.append({"type": "text", "text": block["text"]})
                elif block["type"] == "reasoning":
                    replay = message.get("source", {}).get("replayState", {})
                    blocks = replay.get("blocks", ()) if isinstance(replay, Mapping) else ()
                    meta = blocks[len(content)] if len(blocks) > len(content) else {}
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": block["text"],
                            **(
                                {"signature": meta["signature"]}
                                if isinstance(meta, Mapping) and "signature" in meta
                                else {}
                            ),
                        }
                    )
                elif block["type"] == "tool-call":
                    content.append(
                        {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": _tool_input(block["arguments"]),
                        }
                    )
                else:
                    raise LlmError(
                        f"DeepSeek Messages cannot represent {block['type']}",
                        "UNSUPPORTED_CONTENT",
                    )
            wire_role = "assistant"
        elif role == "tool":
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message["toolCallId"],
                    "content": [
                        {"type": "text", "text": block["text"]}
                        for block in message["content"]
                        if block["type"] == "text"
                    ],
                    **({"is_error": message["isError"]} if "isError" in message else {}),
                }
            )
            wire_role = "user"
        elif role in ("user", "developer"):
            for block in message["content"]:
                if block["type"] != "text":
                    raise LlmError(
                        f"DeepSeek Messages cannot represent {block['type']}",
                        "UNSUPPORTED_CONTENT",
                    )
                if block["text"]:
                    content.append({"type": "text", "text": block["text"]})
            wire_role = "user" if role == "user" else "system"
        else:
            raise LlmError(f'unsupported message role "{role}"', "UNSUPPORTED_CONTENT")
        if not content:
            continue
        if messages and messages[-1]["role"] == wire_role:
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": wire_role, "content": content})
    effort = options.get("reasoningEffort")
    if effort not in (None, "off", "low", "high", "max"):
        raise LlmError(
            f"DeepSeek Messages does not support reasoning effort {effort}",
            "UNSUPPORTED_REASONING_EFFORT",
        )
    enabled = thinking != "disabled" and effort != "off"
    body: dict[str, Any] = {
        "model": options["model"],
        "stream": True,
        "messages": messages,
        "max_tokens": options.get("maxTokens", max_tokens),
        "thinking": {"type": "enabled" if enabled else "disabled"},
        **(
            {"output_config": {"effort": effort}}
            if enabled and effort in ("low", "high", "max")
            else {}
        ),
        **({"system": "\n\n".join(system)} if system else {}),
    }
    if "temperature" in options:
        body["temperature"] = options["temperature"]
    if "stop" in options:
        body["stop_sequences"] = options["stop"]
    if "tools" in options:
        body["tools"] = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
                **({"defer_loading": True} if tool.get("deferLoading") is True else {}),
            }
            for tool in options["tools"]
        ]
    detached = snapshot_json_value(body)
    if not isinstance(detached, dict):
        raise LlmError("DeepSeek request is not lossless JSON", "INVALID_REQUEST")
    return detached


def _stop_reason(raw: object) -> dict[str, str]:
    if raw in ("end_turn", "stop_sequence"):
        return {"kind": "stop"}
    if raw == "tool_use":
        return {"kind": "tool-calls"}
    if raw == "max_tokens":
        return {"kind": "max-tokens"}
    raise LlmError(
        f"unsupported DeepSeek stop reason {raw!r}",
        "MALFORMED_RESPONSE",
    )


async def translate_events(
    events: AsyncIterable[Mapping[str, Any]], model: str
) -> AsyncIterator[Mapping[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    usage: dict[str, int] = {"inputTokens": 0, "outputTokens": 0}
    reason: dict[str, str] | None = None
    started = False
    async for event in events:
        kind = event.get("type")
        if kind == "message_start":
            if started:
                raise LlmError("duplicate message_start", "MALFORMED_RESPONSE")
            started = True
            start_usage = _object(
                _object(event.get("message"), "expected message object").get("usage"),
                "expected usage object",
            )
            usage["inputTokens"] = int(start_usage.get("input_tokens", 0))
        elif kind == "content_block_start":
            wire_index = int(event["index"])
            native = _object(event.get("content_block"), "expected content block")
            block_type = native.get("type")
            index = len(blocks)
            if block_type == "text":
                block = {"type": "text", "text": str(native.get("text", ""))}
                meta = {"type": "text"}
            elif block_type == "thinking":
                block = {"type": "reasoning", "text": str(native.get("thinking", ""))}
                meta = {
                    "type": "reasoning",
                    **(
                        {"signature": native["signature"]}
                        if isinstance(native.get("signature"), str)
                        else {}
                    ),
                }
            elif block_type == "tool_use":
                block = {
                    "type": "tool-call",
                    "id": str(native.get("id", "")),
                    "name": str(native.get("name", "")),
                    "arguments": json.dumps(native.get("input", {}), separators=(",", ":")),
                }
                meta = {"type": "tool-call"}
            else:
                raise LlmError(
                    f"DeepSeek Messages does not support response block {block_type}",
                    "UNSUPPORTED_CONTENT",
                )
            blocks[wire_index] = {"index": index, "block": block, "json": ""}
            metadata.append(meta)
            yield {"type": "block-start", "index": index, "blockType": block["type"]}
            if block["type"] == "text" and block["text"]:
                yield {"type": "text-delta", "index": index, "text": block["text"]}
            elif block["type"] == "reasoning" and block["text"]:
                yield {"type": "reasoning-delta", "index": index, "text": block["text"]}
            elif block["type"] == "tool-call":
                yield {
                    "type": "tool-call-delta",
                    "index": index,
                    "id": block["id"],
                    "name": block["name"],
                    "argumentsDelta": "",
                }
        elif kind == "content_block_delta":
            current = blocks[int(event["index"])]
            delta = _object(event.get("delta"), "expected delta")
            if delta.get("type") == "text_delta":
                text = str(delta.get("text", ""))
                current["block"]["text"] += text
                yield {"type": "text-delta", "index": current["index"], "text": text}
            elif delta.get("type") == "thinking_delta":
                text = str(delta.get("thinking", ""))
                current["block"]["text"] += text
                yield {"type": "reasoning-delta", "index": current["index"], "text": text}
            elif delta.get("type") == "signature_delta":
                metadata[current["index"]]["signature"] = metadata[current["index"]].get(
                    "signature", ""
                ) + str(delta.get("signature", ""))
            elif delta.get("type") == "input_json_delta":
                text = str(delta.get("partial_json", ""))
                current["json"] += text
                yield {
                    "type": "tool-call-delta",
                    "index": current["index"],
                    "argumentsDelta": text,
                }
            else:
                raise LlmError("unsupported DeepSeek delta", "MALFORMED_RESPONSE")
        elif kind == "content_block_stop":
            current = blocks[int(event["index"])]
            block = current["block"]
            if block["type"] == "tool-call" and current["json"]:
                block["arguments"] = current["json"]
            yield {"type": "block-end", "index": current["index"], "block": block}
        elif kind == "message_delta":
            delta = _object(event.get("delta"), "expected message delta")
            if delta.get("stop_reason") is not None:
                reason = _stop_reason(delta["stop_reason"])
            delta_usage = event.get("usage")
            if isinstance(delta_usage, Mapping):
                usage["outputTokens"] = int(delta_usage.get("output_tokens", 0))
        elif kind == "message_stop":
            if reason is None:
                raise LlmError("message_stop without stop reason", "MALFORMED_RESPONSE")
            usage["totalTokens"] = usage["inputTokens"] + usage["outputTokens"]
            yield {"type": "usage", "usage": usage}
            yield {
                "type": "finish",
                "reason": reason,
                "replayState": {
                    "response": {
                        "kind": "deepseek-messages",
                        "version": 1,
                        "model": model,
                    },
                    "blocks": metadata,
                },
            }
            return
        elif kind == "error":
            error = _object(event.get("error"), "expected error")
            raise LlmError(str(error.get("message", "DeepSeek provider error")), "TRANSPORT")
    raise LlmError("DeepSeek Messages stream ended before message_stop", "STREAM_CLOSED")


async def parse_sse(lines: AsyncIterable[str]) -> AsyncIterator[Mapping[str, Any]]:
    event_name: str | None = None
    async for line in lines:
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError as error:
                raise LlmError(
                    "DeepSeek Messages SSE contains invalid JSON",
                    "MALFORMED_RESPONSE",
                ) from error
            if not isinstance(event, Mapping) or (
                event_name is not None and event.get("type") != event_name
            ):
                raise LlmError(
                    "DeepSeek Messages SSE event type mismatch",
                    "MALFORMED_RESPONSE",
                )
            yield event
            event_name = None


class DeepSeekAdapter(LlmAdapter):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/anthropic",
        models: Sequence[Mapping[str, Any]] = (),
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        max_tokens: int = 8192,
        thinking: str = "enabled",
        extensions: DeepSeekLlmApiExtensionRegistry | None = None,
    ) -> None:
        self.api_key = assert_usable_api_key(api_key, "deskagents.llm.deepseek", "api_key")
        self.base_url = base_url.rstrip("/")
        self.models = tuple(models)
        self.headers = dict(headers or {})
        self.client = client
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.extensions = extensions

    def provider_info(self, provider: str) -> Mapping[str, Any]:
        return {"id": provider, "name": "DeepSeek"}

    async def list_models(self, provider: str) -> Sequence[Mapping[str, Any]]:
        return tuple({**item, "provider": provider} for item in self.models)

    async def resolve_model(
        self, provider: str, model: str, signal: object = None
    ) -> Mapping[str, Any]:
        del signal
        declared = next((item for item in self.models if item.get("id") == model), None)
        return {
            "provider": provider,
            "id": model,
            "name": model if declared is None else declared.get("name", model),
            "systemPromptUpdate": "in-history",
            "toolUpdate": "in-history",
            **({} if declared is None else dict(declared)),
        }

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        async def generate() -> AsyncIterator[Mapping[str, Any]]:
            own_client = self.client is None
            client = self.client or httpx.AsyncClient(timeout=None)
            try:
                body = serialize_request(
                    options,
                    max_tokens=self.max_tokens,
                    thinking=self.thinking,
                )
                prepared = (
                    None
                    if self.extensions is None
                    else await self.extensions.prepare(
                        {
                            "body": body,
                            **(
                                {"sessionId": options["sessionId"]}
                                if "sessionId" in options
                                else {}
                            ),
                            **({"purpose": options["purpose"]} if "purpose" in options else {}),
                            "signal": options.get("signal"),
                        }
                    )
                )
                extension_fields: Mapping[str, Any] = {}
                if prepared is not None:
                    collisions = set(body) & set(prepared.fields)
                    if collisions:
                        prepared = None
                    else:
                        extension_fields = prepared.fields
                async with client.stream(
                    "POST",
                    (
                        f"{self.base_url}/messages"
                        if self.base_url.endswith("/v1")
                        else f"{self.base_url}/v1/messages"
                    ),
                    headers={
                        **self.headers,
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "accept": "text/event-stream",
                        **attribution_headers(),
                        **(
                            {"x-deepseek-harness-session-id": str(options["sessionId"])}
                            if "sessionId" in options
                            else {}
                        ),
                        **(
                            {"x-deepseek-harness-compact": "1"}
                            if options.get("purpose") == "compaction"
                            else {}
                        ),
                    },
                    json={
                        **body,
                        **extension_fields,
                    },
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
                            f"DeepSeek request failed ({response.status_code}): {detail}",
                            code,
                            status=response.status_code,
                            request_id=response.headers.get("request-id"),
                        )
                    if prepared is not None:
                        await prepared.accept()
                    events = parse_sse(response.aiter_lines())
                    async for chunk in translate_events(events, str(options["model"])):
                        yield chunk
            except httpx.HTTPError as error:
                raise LlmError("DeepSeek transport failed", "TRANSPORT", cause=error) from error
            finally:
                if own_client:
                    await client.aclose()

        return generate()


__all__ = [
    "DeepSeekAdapter",
    "parse_sse",
    "serialize_request",
    "translate_events",
]
