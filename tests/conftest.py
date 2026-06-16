"""Shared fixtures for TokenCircuit tests."""

from __future__ import annotations

import pytest

from tokencircuit.engine import InterventionConfig, InterventionEngine
from tokencircuit.types import (
    CanonicalMessage,
    CanonicalRole,
    InterventionStage,
    SignalType,
)


@pytest.fixture
def default_config() -> InterventionConfig:
    """Standard config with low thresholds for deterministic test behavior."""
    return InterventionConfig(
        nudge_threshold=3,
        override_threshold=5,
        hard_stop_threshold=8,
        window_size=5,
        similarity_threshold=0.92,
        enable_semantic_detection=False,
    )


@pytest.fixture
def engine(default_config: InterventionConfig) -> InterventionEngine:
    """InterventionEngine with default config."""
    return InterventionEngine(config=default_config)


def _canonical(
    role: CanonicalRole,
    content: str = "",
    *,
    idx: int,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
) -> CanonicalMessage:
    return CanonicalMessage(
        role=role,
        content=content,
        tool_calls=tool_calls or [],
        tool_call_id=tool_call_id,
        source_index=idx,
        name=name,
    )
