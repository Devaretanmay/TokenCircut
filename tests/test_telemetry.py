from tokencircuit.telemetry import (
    TelemetryEvent,
    compute_cost_estimate,
)


class TestComputeCostEstimate:
    def test_known_model(self):
        tokens, cost = compute_cost_estimate("gpt-4", 5)
        assert tokens == 5120
        assert cost > 0

    def test_unknown_model_falls_back(self):
        tokens, cost = compute_cost_estimate("nonexistent-model", 5)
        assert tokens == 5120
        assert cost > 0

    def test_zero_iterations_saved(self):
        tokens, cost = compute_cost_estimate("gpt-4", 0)
        assert tokens == 0
        assert cost == 0.0

    def test_custom_avg_tokens(self):
        tokens, cost = compute_cost_estimate(
            "gpt-3.5-turbo", 3, avg_tokens_per_call=500
        )
        assert tokens == 1500
        assert cost > 0

    def test_gpt4o_mini_cost(self):
        tokens, cost = compute_cost_estimate("gpt-4o-mini", 10)
        assert tokens == 10240
        assert cost > 0

    def test_claude_sonnet_cost(self):
        tokens, cost = compute_cost_estimate("claude-3-sonnet", 5)
        assert tokens == 5120
        assert cost > 0


class TestTelemetryEvent:
    def test_default_timestamp(self):
        event = TelemetryEvent(
            agency_id="a",
            client_id="b",
            agent_framework="langgraph",
            signal_type="STATE_STAGNATION",
            node_name="node_x",
            iterations_at_detection=5,
            model_name="gpt-4",
            estimated_tokens_saved=1000,
            estimated_cost_saved_usd=0.05,
        )
        assert event.timestamp is not None
        assert "T" in event.timestamp
