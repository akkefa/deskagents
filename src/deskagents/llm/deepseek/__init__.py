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
from deskagents.llm.errors import LlmError
from deskagents.llm.runtime import LlmAdapter
from deskagents.util.values import is_safe_integer, parse_json, snapshot_json_value

from .transport import provider_error


def _object(value: object, detail: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LlmError(f"DeepSeek Messages {detail}", "MALFORMED_RESPONSE")
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise LlmError("DeepSeek Messages expected a string field", "MALFORMED_RESPONSE")
    return value


def _unsupported(kind: str) -> None:
    raise LlmError(f"DeepSeek Messages cannot represent {kind}", "UNSUPPORTED_CONTENT")


def _tool_input(raw: str) -> Mapping[str, Any]:
    try:
        value = parse_json(raw)
    except ValueError:
        return {}
    return value if isinstance(value, Mapping) else {}


def _replay(message: Mapping[str, Any], model: str) -> Sequence[Mapping[str, Any]]:
    """Invalid/foreign replay degrades to durable content, never foreign signatures."""
    source = message.get("source", {})
    if not isinstance(source, Mapping) or source.get("kind") != "model":
        return ()
    envelope = source.get("replayState")
    if not isinstance(envelope, Mapping):
        return ()
    response, blocks = envelope.get("response"), envelope.get("blocks")
    if not isinstance(response, Mapping) or not isinstance(blocks, (list, tuple)):
        return ()
    if (
        response.get("kind") != "deepseek-messages"
        or response.get("version") != 1
        or isinstance(response.get("version"), bool)
        or response.get("model") != source.get("model")
        or response.get("model") != model
        or len(blocks) != len(message["content"])
    ):
        return ()
    for block, content in zip(blocks, message["content"], strict=True):
        if (
            not isinstance(block, Mapping)
            or block.get("type") != content["type"]
            or block.get("type") not in ("text", "reasoning", "tool-call")
            or (
                "signature" in block
                and (block["type"] != "reasoning" or not isinstance(block["signature"], str))
            )
        ):
            return ()
    return blocks


def serialize_request(
    options: Mapping[str, Any],
    *,
    max_tokens: int = 8192,
    thinking: str = "enabled",
    reasoning_effort: str | None = None,
    in_history: bool = True,
) -> dict[str, Any]:
    """Serialize core text/tool history, preserving native update ordering."""
    history_system: str | None = None
    messages: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []

    def flush_updates() -> None:
        if not updates:
            return
        if not messages or messages[-1]["role"] != "user":
            _unsupported("system update without a preceding user or tool-result turn")
        messages.extend(updates)
        updates.clear()

    def input_content(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for block in blocks:
            if block["type"] == "text":
                if block["text"]:
                    result.append({"type": "text", "text": block["text"]})
            elif block["type"] not in ("reasoning", "tool-call"):
                _unsupported(f"user/tool-result content {block['type']}")
        return result

    for message in options.get("messages", ()):
        role = message["role"]
        content: list[dict[str, Any]] = []
        if role == "developer":
            for block in message["content"]:
                if block["type"] == "text":
                    if block["text"]:
                        content.append({"type": "text", "text": block["text"]})
                elif block["type"] in ("tool-addition", "tool-removal"):
                    content.append(
                        {
                            "type": block["type"].replace("-", "_"),
                            "tool": {"type": "tool_reference", "name": block["toolName"]},
                        }
                    )
                else:
                    _unsupported(f"developer content {block['type']}")
            if content:
                updates.append({"role": "system", "content": content})
            continue
        if any(b["type"] in ("tool-addition", "tool-removal") for b in message["content"]):
            _unsupported("tool-change blocks outside developer messages")
        if role == "system":
            if any(b["type"] != "text" for b in message["content"]):
                _unsupported("non-text system message")
            text = "".join(b["text"] for b in message["content"])
            if in_history and messages:
                if not text:
                    _unsupported("empty in-history system update")
                updates.append({"role": "system", "content": [{"type": "text", "text": text}]})
            else:
                history_system = text
            continue
        if role == "assistant":
            replay = _replay(message, str(options["model"]))
            for index, block in enumerate(message["content"]):
                if block["type"] == "text":
                    if block["text"]:
                        content.append({"type": "text", "text": block["text"]})
                elif block["type"] == "reasoning":
                    meta = replay[index] if replay else {}
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": block["text"],
                            **({"signature": meta["signature"]} if "signature" in meta else {}),
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
                    _unsupported(f"assistant content {block['type']}")
            wire_role = "assistant"
        elif role == "tool":
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": message["toolCallId"],
                    "content": input_content(message["content"]),
                    **({"is_error": message["isError"]} if "isError" in message else {}),
                }
            ]
            wire_role = "user"
        elif role == "user":
            content, wire_role = input_content(message["content"]), "user"
        else:
            raise LlmError(f'unsupported message role "{role}"', "UNSUPPORTED_CONTENT")
        if not content:
            continue
        if role == "assistant":
            flush_updates()
        if messages and messages[-1]["role"] == wire_role:
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": wire_role, "content": content})
    flush_updates()
    pending: set[str] = set()
    for message in messages:
        if message["role"] == "assistant":
            calls = [b for b in message["content"] if b["type"] == "tool_use"]
            pending = {b["id"] for b in calls}
            if len(pending) != len(calls):
                raise LlmError("DeepSeek Messages duplicate tool call id", "INVALID_REQUEST")
        elif message["role"] == "user":
            results = [b for b in message["content"] if b["type"] == "tool_result"]
            for result in results:
                if result["tool_use_id"] not in pending:
                    raise LlmError(
                        "DeepSeek Messages tool result has no matching call", "INVALID_REQUEST"
                    )
                pending.remove(result["tool_use_id"])
            if pending:
                raise LlmError(
                    "DeepSeek Messages tool calls need immediate results", "INVALID_REQUEST"
                )
            message["content"] = results + [
                b for b in message["content"] if b["type"] != "tool_result"
            ]
    if pending:
        raise LlmError("DeepSeek Messages history ends with unresolved tools", "INVALID_REQUEST")
    effort = (
        options.get("reasoningEffort")
        or reasoning_effort
        or ("off" if thinking == "disabled" else "high")
    )
    if effort not in ("off", "low", "high", "max") or (thinking == "disabled" and effort != "off"):
        raise LlmError(
            f"DeepSeek Messages does not support reasoning effort {effort}",
            "UNSUPPORTED_REASONING_EFFORT",
        )
    system = "\n\n".join(value for value in (options.get("system"), history_system) if value)
    body: dict[str, Any] = {
        "model": options["model"],
        "stream": True,
        "messages": messages,
        "max_tokens": options.get("maxTokens", max_tokens),
        "thinking": {"type": "disabled" if effort == "off" else "enabled"},
        **({"output_config": {"effort": effort}} if effort != "off" else {}),
        **({"system": system} if system else {}),
    }
    for src, dst in (("temperature", "temperature"), ("stop", "stop_sequences")):
        if src in options:
            body[dst] = options[src]
    if "tools" in options:
        body["tools"] = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
                **({"defer_loading": True} if t.get("deferLoading") is True else {}),
            }
            for t in options["tools"]
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
    raise LlmError(f"unsupported DeepSeek stop reason {raw!r}", "MALFORMED_RESPONSE")


def _index(event: Mapping[str, Any]) -> int:
    value = event.get("index")
    if not isinstance(value, int | float) or not is_safe_integer(value) or value < 0:
        raise LlmError("DeepSeek Messages invalid block index", "MALFORMED_RESPONSE")
    return int(value)


def _usage(usage: dict[str, int], raw: object) -> None:
    fields = _object(raw, "expected usage object")
    for wire, local in (
        ("input_tokens", "inputTokens"),
        ("output_tokens", "outputTokens"),
        ("cache_read_input_tokens", "cacheReadTokens"),
        ("cache_creation_input_tokens", "cacheWriteTokens"),
    ):
        if wire in fields:
            value = fields[wire]
            if not isinstance(value, int | float) or not is_safe_integer(value) or value < 0:
                raise LlmError(f"DeepSeek Messages invalid {wire}", "MALFORMED_RESPONSE")
            usage[local] = int(value)


async def translate_events(
    events: AsyncIterable[Mapping[str, Any]], model: str
) -> AsyncIterator[Mapping[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    usage = {"inputTokens": 0, "outputTokens": 0}
    reason: dict[str, str] | None = None
    started = False
    async for event in events:
        kind = event.get("type")
        if kind == "error":
            raise provider_error(event)
        if kind == "message_start":
            if started:
                raise LlmError("duplicate message_start", "MALFORMED_RESPONSE")
            _usage(usage, _object(event.get("message"), "expected message").get("usage"))
            started = True
            continue
        if kind not in (
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ):
            continue
        if not started:
            raise LlmError("event precedes message_start", "MALFORMED_RESPONSE")
        if kind == "content_block_start":
            wire_index = _index(event)
            if wire_index in blocks or reason is not None:
                raise LlmError(
                    "block starts after settlement or repeats an index", "MALFORMED_RESPONSE"
                )
            native = _object(event.get("content_block"), "expected content block")
            block_type = native.get("type")
            index = len(blocks)
            if block_type == "text":
                block = {"type": "text", "text": _string(native.get("text"))}
                meta = {"type": "text"}
            elif block_type == "thinking":
                block = {"type": "reasoning", "text": _string(native.get("thinking"))}
                meta = {
                    "type": "reasoning",
                    **(
                        {"signature": _string(native["signature"])} if "signature" in native else {}
                    ),
                }
            elif block_type == "tool_use":
                block = {
                    "type": "tool-call",
                    "id": _string(native.get("id")),
                    "name": _string(native.get("name")),
                    "arguments": json.dumps(
                        dict(_object(native.get("input"), "expected tool input")),
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                }
                if not block["id"] or not block["name"]:
                    raise LlmError("empty tool identity", "MALFORMED_RESPONSE")
                meta = {"type": "tool-call"}
            else:
                raise LlmError(
                    f"DeepSeek Messages does not support response block {block_type}",
                    "UNSUPPORTED_CONTENT",
                )
            blocks[wire_index] = {"index": index, "block": block, "json": "", "closed": False}
            metadata.append(meta)
            yield {"type": "block-start", "index": index, "blockType": block["type"]}
            if block["type"] in ("text", "reasoning") and block["text"]:
                yield {
                    "type": "text-delta" if block["type"] == "text" else "reasoning-delta",
                    "index": index,
                    "text": block["text"],
                }
            elif block["type"] == "tool-call":
                yield {
                    "type": "tool-call-delta",
                    "index": index,
                    "id": block["id"],
                    "name": block["name"],
                    "argumentsDelta": "",
                }
        elif kind in ("content_block_delta", "content_block_stop"):
            current = blocks.get(_index(event))
            if current is None or current["closed"]:
                raise LlmError("delta/stop without an open block", "MALFORMED_RESPONSE")
            block = current["block"]
            if kind == "content_block_stop":
                current["closed"] = True
                if block["type"] == "tool-call" and current["json"]:
                    block["arguments"] = current["json"]
                yield {"type": "block-end", "index": current["index"], "block": dict(block)}
                continue
            delta = _object(event.get("delta"), "expected delta")
            delta_type = delta.get("type")
            if (delta_type == "text_delta" and block["type"] == "text") or (
                delta_type == "thinking_delta" and block["type"] == "reasoning"
            ):
                text = _string(delta.get("text" if delta_type == "text_delta" else "thinking"))
                block["text"] += text
                yield {
                    "type": "text-delta" if delta_type == "text_delta" else "reasoning-delta",
                    "index": current["index"],
                    "text": text,
                }
            elif delta_type == "signature_delta" and block["type"] == "reasoning":
                meta = metadata[current["index"]]
                meta["signature"] = meta.get("signature", "") + _string(delta.get("signature"))
            elif delta_type == "input_json_delta" and block["type"] == "tool-call":
                text = _string(delta.get("partial_json"))
                current["json"] += text
                yield {
                    "type": "tool-call-delta",
                    "index": current["index"],
                    "id": block["id"],
                    "argumentsDelta": text,
                }
            else:
                raise LlmError("unsupported DeepSeek delta for block type", "MALFORMED_RESPONSE")
        elif kind == "message_delta":
            delta = _object(event.get("delta"), "expected message delta")
            if delta.get("stop_reason") is not None:
                reason = _stop_reason(delta["stop_reason"])
            if "usage" in event:
                _usage(usage, event["usage"])
        elif kind == "message_stop":
            if reason is None or any(not b["closed"] for b in blocks.values()):
                raise LlmError(
                    "message_stop without settled blocks and stop reason", "MALFORMED_RESPONSE"
                )
            if not blocks and reason["kind"] == "stop":
                raise LlmError("DeepSeek Messages returned no content", "EMPTY_RESPONSE")
            if reason["kind"] != "max-tokens":
                for current in blocks.values():
                    if current["block"]["type"] == "tool-call":
                        try:
                            value = parse_json(current["block"]["arguments"])
                        except ValueError as error:
                            raise LlmError(
                                "tool input is invalid JSON", "MALFORMED_RESPONSE"
                            ) from error
                        _object(value, "tool input must be an object")
            usage["totalTokens"] = sum(usage.values())
            yield {"type": "usage", "usage": usage}
            yield {
                "type": "finish",
                "reason": reason,
                "replayState": {
                    "response": {"kind": "deepseek-messages", "version": 1, "model": model},
                    "blocks": metadata,
                },
            }
            return
    raise LlmError("DeepSeek Messages stream ended before message_stop", "STREAM_CLOSED")


async def parse_sse(lines: AsyncIterable[str]) -> AsyncIterator[Mapping[str, Any]]:
    """Dispatch complete SSE frames only; an unterminated tail is not an event."""
    event_name: str | None = None
    data: list[str] = []
    first = True
    async for line in lines:
        if first:
            line = line.removeprefix("\ufeff")
            first = False
        if line == "":
            if data:
                try:
                    event = parse_json("\n".join(data))
                except ValueError as error:
                    raise LlmError(
                        "DeepSeek Messages SSE contains invalid JSON", "MALFORMED_RESPONSE"
                    ) from error
                if (
                    not isinstance(event, Mapping)
                    or not isinstance(event.get("type"), str)
                    or (event_name is not None and event_name != event["type"])
                ):
                    raise LlmError(
                        "DeepSeek Messages SSE event type mismatch", "MALFORMED_RESPONSE"
                    )
                yield event
            event_name, data = None, []
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_name = value or None
            elif field == "data":
                data.append(value)


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
        reasoning_effort: str | None = None,
        extensions: DeepSeekLlmApiExtensionRegistry | None = None,
    ) -> None:
        self.api_key = assert_usable_api_key(api_key, "deskagents.llm.deepseek", "api_key")
        self.base_url = base_url.rstrip("/")
        self.models = tuple(models)
        self.headers = dict(headers or {})
        self.client = client
        self.max_tokens = max_tokens
        if thinking not in ("enabled", "disabled"):
            raise ValueError("thinking must be enabled or disabled")
        if reasoning_effort not in (None, "off", "low", "high", "max") or (
            thinking == "disabled" and reasoning_effort not in (None, "off")
        ):
            raise ValueError("reasoning effort conflicts with thinking policy")
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.extensions = extensions

    def provider_info(self, provider: str) -> Mapping[str, Any]:
        return {"id": provider, "name": "DeepSeek"}

    async def list_models(self, provider: str) -> Sequence[Mapping[str, Any]]:
        return tuple(
            {**item, "name": item.get("name", item["id"]), "provider": provider}
            for item in self.models
        )

    async def resolve_model(
        self, provider: str, model: str, signal: object = None
    ) -> Mapping[str, Any]:
        del signal
        declared = next((item for item in self.models if item.get("id") == model), None)
        return {
            "provider": provider,
            "id": model,
            "name": model if declared is None else declared.get("name", model),
            "inputModalities": ["text"],
            "defaultMaxTokens": self.max_tokens
            if declared is None
            else declared.get("maxTokens", self.max_tokens),
            "reasoning": {
                "efforts": [
                    {"id": effort, "name": effort.title()}
                    for effort in (
                        ("off",) if self.thinking == "disabled" else ("off", "low", "high", "max")
                    )
                ],
                "defaultEffort": self.reasoning_effort
                or ("off" if self.thinking == "disabled" else "high"),
            },
            "systemPromptUpdate": "in-history",
            "toolUpdate": "in-history",
            **({} if declared is None else dict(declared)),
        }

    def stream(self, options: Mapping[str, Any]) -> AsyncIterable[Mapping[str, Any]]:
        async def generate() -> AsyncIterator[Mapping[str, Any]]:
            own_client = self.client is None
            client = self.client or httpx.AsyncClient(timeout=None)
            try:
                model_info = await self.resolve_model(
                    str(options.get("provider", "deepseek")), str(options["model"])
                )
                body = serialize_request(
                    options,
                    max_tokens=int(model_info["defaultMaxTokens"]),
                    thinking=self.thinking,
                    reasoning_effort=self.reasoning_effort,
                    in_history=model_info.get("systemPromptUpdate") == "in-history",
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
                        await response.aread()
                        try:
                            failure = response.json()
                        except ValueError:
                            failure = None
                        raise provider_error(failure, response.status_code, response.headers)
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
