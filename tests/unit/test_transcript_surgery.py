"""Tests for surgical transcript surgery in Override stage."""

import pytest

from tokencircuit.engine import InterventionConfig, InterventionEngine
from tokencircuit.types import (
    CanonicalMessage,
    CanonicalRole,
    InterventionContext,
    InterventionStage,
    SignalType,
)


def test_override_surgery_compacts_repetitive_tool_calls():
    """
    Ensure Override stage compacts repetitive tool calls while preserving
    the original human task and system prompt.
    """
    engine = InterventionEngine(config=InterventionConfig(nudge_threshold=2, override_threshold=3))  # noqa: E501

    # 1. Build a repetitive transcript
    messages = [
        CanonicalMessage(role=CanonicalRole.SYSTEM, content="You are a helpful assistant."),  # noqa: E501
        CanonicalMessage(role=CanonicalRole.HUMAN, content="Fetch the data."),
        # Turn 1: Call tool A
        CanonicalMessage(role=CanonicalRole.AI, tool_calls=[{"id": "call_1", "name": "get_data", "args": {}}]),  # noqa: E501
        CanonicalMessage(role=CanonicalRole.TOOL, tool_call_id="call_1", content="Error: 403"),  # noqa: E501
        # Turn 2: Repeat call tool A (identical)
        CanonicalMessage(role=CanonicalRole.AI, tool_calls=[{"id": "call_2", "name": "get_data", "args": {}}]),  # noqa: E501
        CanonicalMessage(role=CanonicalRole.TOOL, tool_call_id="call_2", content="Error: 403"),  # noqa: E501
        # Turn 3: Repeat call tool A (identical) -> triggers OVERRIDE
        CanonicalMessage(role=CanonicalRole.AI, tool_calls=[{"id": "call_3", "name": "get_data", "args": {}}]),  # noqa: E501
        CanonicalMessage(role=CanonicalRole.TOOL, tool_call_id="call_3", content="Error: 403"),  # noqa: E501
    ]

    context = InterventionContext(
        thread_id="test_thread",
        node_name="agent",
        turn_number=4,
        active_signals=[SignalType.FUTILE_ACTION],
        consecutive_stagnation_count=5, # Above threshold
        current_stage=InterventionStage.OVERRIDE,
    )

    # 2. Run surgery
    decision = engine.decide(context, messages)

    # 3. Verify
    assert decision.stage == InterventionStage.OVERRIDE
    assert decision.llm_input_messages is not None

    # Check compaction:
    # Original: SYSTEM, HUMAN, AI(call_1), TOOL(call_1), AI(call_2), TOOL(call_2), AI(call_3), TOOL(call_3)  # noqa: E501
    # Expected: SYSTEM, HUMAN, AI(call_1), TOOL(call_1), SYSTEM(COACHING)
    # The later identical calls should be dropped.

    roles = [m["role"] for m in decision.llm_input_messages]
    assert roles.count("system") >= 2 # Original + Coaching
    assert roles.count("user") == 1 # Original human task

    # Verify that we don't have call_2 or call_3 in the final messages
    content_str = str(decision.llm_input_messages)
    assert "call_1" in content_str
    assert "call_2" not in content_str
    assert "call_3" not in content_str

    print("Surgery success: Repetitive transactions compacted.")

if __name__ == "__main__":
    pytest.main([__file__])
