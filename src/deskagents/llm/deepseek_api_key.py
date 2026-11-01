"""Official DeepSeek API-key provider plugin.

Upstream: packages/llm/llm-deepseek-api-key at ce7073d.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from deepseek_cordis import Context

from deskagents.llm.deepseek import DeepSeekAdapter
from deskagents.llm.deepseek_llm_api_extensions import DeepSeekLlmApiExtensionRegistry
from deskagents.llm.errors import LlmError
from deskagents.llm.runtime import LlmRuntime

PROVIDER = "deepseek-official"


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> Any:
    config = config or {}
    llm = ctx.get("llm")
    if not isinstance(llm, LlmRuntime):
        raise RuntimeError("llm-deepseek-api-key requires LlmRuntime")
    ref = str(config.get("apiKeyEnv", "DEEPSEEK_API_KEY"))
    raw = config.get("apiKey", os.environ.get(ref))
    if not isinstance(raw, str) or not raw.strip():
        raise LlmError(
            f'llm-deepseek: no API key for provider route "{PROVIDER}"; '
            f"set config.apiKey or export {ref}",
            "MISSING_CREDENTIAL",
        )
    models = config.get("models", ())
    if not isinstance(models, Sequence):
        raise TypeError("models must be a sequence")
    extensions = ctx.get("deepseekLlmApiExtensions")
    adapter = DeepSeekAdapter(
        api_key=raw,
        base_url=str(config.get("baseURL", "https://api.deepseek.com/anthropic")),
        models=models,
        headers=config.get("headers"),
        max_tokens=int(config.get("maxTokens", 8192)),
        thinking=str(config.get("thinking", "enabled")),
        extensions=(
            extensions if isinstance(extensions, DeepSeekLlmApiExtensionRegistry) else None
        ),
    )
    return llm.register_adapter([str(config.get("provider", PROVIDER))], adapter, ctx=ctx)


__all__ = ["PROVIDER", "apply"]
