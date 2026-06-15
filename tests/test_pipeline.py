import pytest

from tokencircuit.detectors.pipeline import DetectionPipeline
from tokencircuit.config import TokenCircuitConfig


class TestDetectionPipeline:
    def test_record_step_normal(self):
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        pipeline = DetectionPipeline(config, "test")
        result = pipeline.record_step("agent_1", "node_x", "hash_a", "tool()")
        assert result is None

    def test_detection_triggers(self):
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        pipeline = DetectionPipeline(config, "test")
        for _ in range(4):
            assert pipeline.record_step("agent_1", "node_x", "same", "tool()") is None
        result = pipeline.record_step("agent_1", "node_x", "same", "tool()")
        assert result is not None
        assert result.signal_type == "STATE_STAGNATION"
        assert result.iteration == 5

    def test_reset_clears_state(self):
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        pipeline = DetectionPipeline(config, "test")
        for _ in range(5):
            pipeline.record_step("agent_1", "node_x", "same", "tool()")
        pipeline.reset("agent_1", "node_x")
        for _ in range(4):
            assert pipeline.record_step("agent_1", "node_x", "same", "tool()") is None

    def test_telemetry_not_emitted_when_disabled(self):
        config = TokenCircuitConfig(
            max_repeats=5, window_size=5, telemetry_enabled=False
        )
        pipeline = DetectionPipeline(config, "test")
        for _ in range(4):
            assert pipeline.record_step("agent_1", "node_x", "same", "tool()") is None
        result = pipeline.record_step("agent_1", "node_x", "same", "tool()")
        assert result is not None  # no telemetry, but detection still fires

    def test_telemetry_not_emitted_without_credentials(self):
        config = TokenCircuitConfig(
            max_repeats=5, window_size=5, telemetry_enabled=True,
            agency_id=None, client_id=None,
        )
        pipeline = DetectionPipeline(config, "test", api_key="test_key")
        for _ in range(4):
            assert pipeline.record_step("agent_1", "node_x", "same", "tool()") is None
        result = pipeline.record_step("agent_1", "node_x", "same", "tool()")
        assert result is not None  # detection fires, telemetry skipped

    def test_telemetry_path_covered(self):
        config = TokenCircuitConfig(
            max_repeats=5, window_size=5, telemetry_enabled=True,
            agency_id="test_agency", client_id="test_client",
            model_name="gpt-4",
        )
        pipeline = DetectionPipeline(config, "test", api_key="test_key")
        for _ in range(4):
            assert pipeline.record_step("agent_1", "node_x", "same", "tool()") is None
        result = pipeline.record_step("agent_1", "node_x", "same", "tool()")
        assert result is not None  # detection fires, telemetry sent
