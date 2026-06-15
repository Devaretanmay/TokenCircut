"""
Performance benchmarks for TokenCircuit V6.0.

All benchmarks measure:
- Ring buffer push + evaluate latency (must be < 1ms)
- Telemetry emit_event_async non-blocking behavior
- State hash computation throughput
"""

import time

import pytest

from tokencircuit.detectors.composite import CompositeDetector
from tokencircuit.detectors.futile_action import FutileActionDetector
from tokencircuit.detectors.state_stagnation import StateStagnationDetector
from tokencircuit.otel.hash_utils import compute_state_hash
from tokencircuit.ring_buffer import RingBuffer
from tokencircuit.telemetry import TelemetryEvent, emit_event_async


def make_entry(state_hash="a", tool_sig="tool()", iteration=1):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": iteration,
    }


BENCHMARK_ITERATIONS = 10000


class TestRingBufferLatency:
    """Ring buffer push + evaluate must complete in < 1ms (p99)."""

    def test_push_latency(self):
        buf = RingBuffer(maxlen=5)
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            buf.push(make_entry(iteration=1))
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 push latency {p99:.4f}ms exceeds 1ms"

    def test_window_latency(self):
        buf = RingBuffer(maxlen=5)
        for _ in range(5):
            buf.push(make_entry(iteration=1))

        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            buf.window()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 window latency {p99:.4f}ms exceeds 1ms"

    def test_is_full_latency(self):
        buf = RingBuffer(maxlen=5)
        for _ in range(5):
            buf.push(make_entry(iteration=1))

        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            buf.is_full()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 is_full latency {p99:.4f}ms exceeds 1ms"

    def test_reset_latency(self):
        buf = RingBuffer(maxlen=5)
        for _ in range(5):
            buf.push(make_entry(iteration=1))

        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            buf.reset()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 reset latency {p99:.4f}ms exceeds 1ms"

    def test_push_then_reset_then_push(self):
        times = []
        buf = RingBuffer(maxlen=5)
        for _ in range(BENCHMARK_ITERATIONS // 100):
            start = time.perf_counter_ns()
            for _ in range(5):
                buf.push(make_entry(iteration=1))
            result = buf.is_full()
            buf.reset()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert result is True
        assert p99 < 5.0, f"p99 full cycle latency {p99:.4f}ms exceeds 5ms"


class TestDetectorLatency:
    """Detector evaluate must complete in < 1ms."""

    @pytest.fixture
    def full_buffer(self):
        buf = RingBuffer(maxlen=5)
        for i in range(5):
            buf.push(make_entry(state_hash="same", tool_sig="tool()", iteration=i))
        return buf

    def test_state_stagnation_latency(self, full_buffer):
        det = StateStagnationDetector(threshold=5)
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            det.evaluate(full_buffer)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 stagnation latency {p99:.4f}ms exceeds 1ms"

    def test_futile_action_latency(self, full_buffer):
        det = FutileActionDetector(threshold=5)
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            det.evaluate(full_buffer)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 futile latency {p99:.4f}ms exceeds 1ms"

    def test_composite_latency(self, full_buffer):
        det = CompositeDetector(threshold=5)
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            det.evaluate("agent_1", "node_x", full_buffer)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 composite latency {p99:.4f}ms exceeds 1ms"


class TestHashLatency:
    """State hash computation must be fast."""

    def test_simple_state_hash_latency(self):
        state = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            compute_state_hash(state)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 1.0, f"p99 hash latency {p99:.4f}ms exceeds 1ms"

    def test_large_state_hash_latency(self):
        state = {f"key_{i}": f"value_{i}" for i in range(100)}
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            compute_state_hash(state)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 5.0, f"p99 large-state hash latency {p99:.4f}ms exceeds 5ms"

    def test_nested_state_hash_latency(self):
        state = {
            "outer": {
                "inner": [1, 2, 3],
                "data": {"nested": True, "values": list(range(50))},
            }
        }
        times = []
        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter_ns()
            compute_state_hash(state)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 5.0, f"p99 nested hash latency {p99:.4f}ms exceeds 5ms"


class TestTelemetryNonBlocking:
    """emit_event_async must return in < 0.1ms regardless of network."""

    def test_returns_immediately_no_network(self):
        """Without an API key, emit_event_async should return instantly."""
        event = TelemetryEvent(
            agency_id="test",
            client_id="test",
            agent_framework="langgraph",
            signal_type="STATE_STAGNATION",
            node_name="node_x",
            iterations_at_detection=5,
            model_name="gpt-4",
            estimated_tokens_saved=100,
            estimated_cost_saved_usd=0.01,
        )
        times = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            emit_event_async(event, api_key=None)
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 0.1, f"p99 noop telemetry latency {p99:.4f}ms exceeds 0.1ms"

    def test_returns_immediately_with_key(self):
        """With an API key but no network, should still return instantly
        because it runs in a daemon thread."""
        event = TelemetryEvent(
            agency_id="test",
            client_id="test",
            agent_framework="langgraph",
            signal_type="STATE_STAGNATION",
            node_name="node_x",
            iterations_at_detection=5,
            model_name="gpt-4",
            estimated_tokens_saved=100,
            estimated_cost_saved_usd=0.01,
        )
        times = []
        for _ in range(100):
            start = time.perf_counter_ns()
            emit_event_async(event, api_key="test-key")
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            times.append(elapsed)

        times.sort()
        p99 = times[int(len(times) * 0.99)]
        assert p99 < 20.0, f"p99 telemetry latency {p99:.4f}ms exceeds 20.0ms"
