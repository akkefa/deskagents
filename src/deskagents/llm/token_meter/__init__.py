"""Replay-aware token estimation and session projections.

Adapted from ``packages/llm/token-meter`` at upstream commit ``ce7073d``. Because
attachments belong to an external plugin family, route-specific attachment pricing is
intentionally reduced to structural heuristic pricing; durable assistant messages supply
measurement anchors. Projection keys, versions, known-event handling, usage semantics,
shadow-price behavior, and turn accounting remain aligned.
"""

from .estimate import (
    BLOCK_OVERHEAD,
    CHARS_PER_TOKEN,
    ROLE_OVERHEAD,
    estimate_content,
    estimate_message,
    estimate_structural_block,
    estimate_system_message,
    estimate_tools_tokens,
)
from .projections import (
    BREAKDOWN_STATE_SCHEMA,
    BREAKDOWN_VIEW_SCHEMA,
    PRESSURE_STATE_SCHEMA,
    PRESSURE_VIEW_SCHEMA,
    USAGE_STATE_SCHEMA,
    USAGE_VIEW_SCHEMA,
    context_breakdown_projection_definition,
    context_pressure_projection_definition,
    token_usage_projection_definition,
    usage_of,
)
from .service import (
    MEASUREMENT_STATE_SCHEMA,
    MeasurementState,
    TokenMeter,
    advance_measurement,
    measurement_projection_definition,
)
from .surface import (
    MeterSurfaceNode,
    ShadowPriceClaim,
    fold_surface_projection,
    place_surface_node,
)
from .turn_usage import derive_turn_token_usage
from .types import (
    ContextBreakdownProjection,
    ContextPressureProjection,
    TokenMeasurement,
    TokenMeasurementBaseline,
    TokenSurfaceNode,
    TokenUsageProjection,
    TurnTokenUsage,
    TurnTokenUsageRoute,
)

__all__ = [
    "BLOCK_OVERHEAD",
    "BREAKDOWN_STATE_SCHEMA",
    "BREAKDOWN_VIEW_SCHEMA",
    "CHARS_PER_TOKEN",
    "MEASUREMENT_STATE_SCHEMA",
    "PRESSURE_STATE_SCHEMA",
    "PRESSURE_VIEW_SCHEMA",
    "ROLE_OVERHEAD",
    "USAGE_STATE_SCHEMA",
    "USAGE_VIEW_SCHEMA",
    "ContextBreakdownProjection",
    "ContextPressureProjection",
    "MeasurementState",
    "MeterSurfaceNode",
    "ShadowPriceClaim",
    "TokenMeasurement",
    "TokenMeasurementBaseline",
    "TokenMeter",
    "TokenSurfaceNode",
    "TokenUsageProjection",
    "TurnTokenUsage",
    "TurnTokenUsageRoute",
    "advance_measurement",
    "context_breakdown_projection_definition",
    "context_pressure_projection_definition",
    "derive_turn_token_usage",
    "estimate_content",
    "estimate_message",
    "estimate_structural_block",
    "estimate_system_message",
    "estimate_tools_tokens",
    "fold_surface_projection",
    "measurement_projection_definition",
    "place_surface_node",
    "token_usage_projection_definition",
    "usage_of",
]
