"""Public token-meter value contracts.

The mappings retain upstream camelCase keys because they are projection wire values.
"""

from __future__ import annotations

from typing import NotRequired

from typing_extensions import TypedDict

from deskagents.core.session import SessionSeq
from deskagents.llm import TokenUsage


class TokenUsageProjection(TypedDict):
    uncachedInputTokens: int
    outputTokens: int
    cacheReadTokens: int
    cacheWriteTokens: int


class ContextPressureProjection(TypedDict):
    pressureTokens: NotRequired[int]
    projectedTokens: NotRequired[int]
    contextWindow: NotRequired[int]


class ContextBreakdownProjection(TypedDict):
    systemTokens: int
    toolsTokens: int
    messageTokens: int


class TurnTokenUsageRoute(TypedDict):
    provider: str
    model: str


class TurnTokenUsage(TypedDict):
    uncachedInputTokens: int
    outputTokens: int
    totalTokens: int
    cacheReadTokens: NotRequired[int]
    cacheWriteTokens: NotRequired[int]
    reasoningTokens: NotRequired[int]
    routes: NotRequired[list[TurnTokenUsageRoute]]


class TokenSurfaceNode(TypedDict):
    seq: SessionSeq
    tokens: int
    heuristicTokens: int


class NoneBaseline(TypedDict):
    kind: str
    tokens: int


class UsageBaseline(NoneBaseline):
    usage: TokenUsage


TokenMeasurementBaseline = NoneBaseline | UsageBaseline


class TokenMeasurement(TypedDict):
    logRevision: int
    baseline: TokenMeasurementBaseline
    surfaceDeltaTokens: int
    totalTokens: int
    surfaceTokens: int
    nodes: list[TokenSurfaceNode]
