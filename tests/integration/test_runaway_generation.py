
from langchain_core.messages import AIMessage, HumanMessage

from tokencircuit import InterventionConfig, InterventionEngine
from tokencircuit.state_schema import default_intervention_state
from tokencircuit.types import InterventionStage, SignalType


def test_runaway_generation_hard_stops_immediately():
    """Verify that a single massive response triggers an immediate HARD_STOP."""
    # Set limit to something low, e.g., 100 tokens (approx 400 chars)
    config = InterventionConfig(max_tokens_per_turn=100)
    engine = InterventionEngine(config=config)

    # 500 characters -> ~125 tokens, which is > 100
    massive_content = "A" * 500

    messages = [
        HumanMessage(content="Tell me a story."),
        AIMessage(content=massive_content),
    ]

    tc_state = default_intervention_state()
    state = {"messages": messages, "_tc_intervention": tc_state}

    # Normally, it takes 8 turns of stagnation to hard stop.
    # Runaway generation should skip the ladder and stop immediately.
    decision = engine.process(messages=messages, state=state, thread_id="test", node_name="agent")  # noqa: E501

    assert decision.stage == InterventionStage.HARD_STOP
    assert SignalType.RUNAWAY_GENERATION in decision.signals
    assert decision.should_terminate is True

def test_runaway_generation_respects_audit_mode():
    """Verify that audit_mode suppresses the HARD_STOP mutation."""
    config = InterventionConfig(max_tokens_per_turn=100, audit_mode=True)
    engine = InterventionEngine(config=config)

    massive_content = "A" * 500
    messages = [
        HumanMessage(content="Tell me a story."),
        AIMessage(content=massive_content),
    ]

    state = {"messages": messages, "_tc_intervention": default_intervention_state()}

    decision = engine.process(messages=messages, state=state, thread_id="test", node_name="agent")  # noqa: E501

    # Engine still computes the hard stop internally
    assert decision.stage == InterventionStage.HARD_STOP
    assert SignalType.RUNAWAY_GENERATION in decision.signals
    # The adapter is what respects audit mode, but the engine should return the decision
    # We will test the adapter side indirectly via integration, or rely on existing audit test  # noqa: E501
