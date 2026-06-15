import datetime
import enum
import json
import threading
import uuid

import pytest

from tokencircuit.ring_buffer import RingBuffer
from tokencircuit.otel.hash_utils import compute_state_hash
from tokencircuit.detectors.composite import CompositeDetector, SIGNAL_STAGNATION
from tokencircuit.detectors.state_stagnation import StateStagnationDetector
from tokencircuit.detectors.futile_action import FutileActionDetector


def entry(state_hash="a", tool_sig="tool()", iteration=1):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": iteration,
    }


# ── Non-JSON-serializable values ──


class Color(enum.Enum):
    RED = 1
    BLUE = 2


class TestNonJsonSerializable:
    def test_datetime_value(self):
        state = {"timestamp_val": datetime.datetime(2024, 1, 1, 12, 0, 0)}
        h = compute_state_hash(state)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_uuid_value(self):
        state = {"id": uuid.UUID("12345678-1234-5678-1234-567812345678")}
        h = compute_state_hash(state)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_enum_value(self):
        state = {"color": Color.RED}
        h = compute_state_hash(state)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_bytes_value(self):
        state = {"data": b"hello bytes"}
        h = compute_state_hash(state)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_numpy_like_value(self):
        """Simulate a numpy-like object by using a custom class with str()."""

        class NDArray:
            def __init__(self, data):
                self.data = data

            def __str__(self):
                return f"ndarray({self.data})"

        state = {"matrix": NDArray([1, 2, 3])}
        h = compute_state_hash(state)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_mixed_non_json_values(self):
        state = {
            "dt": datetime.datetime.now(),
            "uid": uuid.uuid4(),
            "enum": Color.BLUE,
            "raw": b"\x00\x01\x02",
        }
        h = compute_state_hash(state)
        assert isinstance(h, str)

    def test_nested_non_json_values(self):
        state = {
            "outer": {
                "inner_dt": datetime.datetime(2024, 6, 15),
                "inner_uuid": uuid.uuid4(),
            }
        }
        h = compute_state_hash(state)
        assert isinstance(h, str)

    def test_non_json_serializable_does_not_raise(self):
        bad_values = [
            datetime.datetime.now(),
            uuid.uuid4(),
            Color.RED,
            b"bytes",
            object(),
            complex(1, 2),
        ]
        for val in bad_values:
            try:
                h = compute_state_hash({"val": val})
                assert isinstance(h, str)
            except Exception:
                pytest.fail(f"compute_state_hash raised for {type(val).__name__}")


# ── Pure LLM reasoning step (NO_TOOL_CALL) ──


class TestPureLLMReasoning:
    def test_reasoning_loop_detected(self):
        """If the LLM keeps reasoning without calling tools,
        state stagnation should fire."""
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(5):
            buf.push(entry(
                state_hash="same_reasoning",
                tool_sig="NO_TOOL_CALL",
                iteration=i,
            ))
        result = det.evaluate("agent_1", "node_x", buf)
        assert result is not None
        assert result.signal_type == SIGNAL_STAGNATION

    def test_reasoning_changing_no_trigger(self):
        """If the LLM's reasoning output changes each iteration
        without tools, no signal should fire."""
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(5):
            buf.push(entry(
                state_hash=f"reasoning_{i}",
                tool_sig="NO_TOOL_CALL",
                iteration=i,
            ))
        result = det.evaluate("agent_1", "node_x", buf)
        assert result is None

    def test_mixed_tool_and_no_tool(self):
        """Alternating between tool calls and NO_TOOL_CALL should not
        trigger futile action (different tool signatures)."""
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        sigs = ["search(str)", "NO_TOOL_CALL", "search(str)", "NO_TOOL_CALL", "search(str)"]
        for i, s in enumerate(sigs):
            buf.push(entry(
                state_hash=f"state_{i}",
                tool_sig=s,
                iteration=i,
            ))
        result = det.evaluate("agent_1", "node_x", buf)
        assert result is None


# ── Parallel agents, same node name ──


class TestParallelAgents:
    def test_different_agents_same_node(self):
        """Two agents running on the same node name should have
        independent buffers based on (agent_id, node_name) key."""
        buf1 = RingBuffer(maxlen=5)
        buf2 = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)

        for i in range(5):
            buf1.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
            buf2.push(entry(
                state_hash=f"different_{i}",
                tool_sig=f"other_{i}()",
                iteration=i,
            ))

        r1 = det.evaluate("agent_a", "node_x", buf1)
        r2 = det.evaluate("agent_b", "node_x", buf2)

        assert r1 is not None
        assert r2 is None

    def test_concurrent_parallel_agents(self):
        """Simulate concurrent execution of two agents on the same
        node name using threading."""
        det = CompositeDetector(threshold=5)
        results = {}
        errors = []

        def run_agent(agent_id):
            try:
                buf = RingBuffer(maxlen=5)
                for i in range(5):
                    buf.push(entry(
                        state_hash="same",
                        tool_sig="tool()",
                        iteration=i,
                    ))
                result = det.evaluate(agent_id, "shared_node", buf)
                if result:
                    results[agent_id] = result.signal_type
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run_agent, args=("agent_1",)),
            threading.Thread(target=run_agent, args=("agent_2",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 2

    def test_buffer_key_uniqueness(self):
        """Verify that buffers for different (agent_id, node_name)
        combinations are independent through the DetectionPipeline."""
        from tokencircuit.detectors.pipeline import DetectionPipeline
        from tokencircuit.config import TokenCircuitConfig

        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        pipeline = DetectionPipeline(config, "langgraph")

        # agent_a gets 4 pushes — buffer not full yet
        for _ in range(4):
            assert pipeline.record_step(
                "agent_a", "node_x", "hash1", "tool()"
            ) is None

        # agent_b gets 1 push — also not full
        assert pipeline.record_step(
            "agent_b", "node_x", "hash1", "tool()"
        ) is None

        # agent_a's 5th push fills buffer → detection fires
        r_a = pipeline.record_step("agent_a", "node_x", "hash1", "tool()")
        assert r_a is not None

        # agent_b still only has 1 push → not full → no detection
        assert pipeline.record_step(
            "agent_b", "node_x", "hash1", "tool()"
        ) is None

        # agent_b needs 3 more to fill (2 already recorded)
        for _ in range(2):
            assert pipeline.record_step(
                "agent_b", "node_x", "hash1", "tool()"
            ) is None
        r_b = pipeline.record_step("agent_b", "node_x", "hash1", "tool()")
        assert r_b is not None


# ── Buffer full edge cases ──


class TestBufferEdgeCases:
    def test_window_size_1(self):
        buf = RingBuffer(maxlen=1)
        buf.push(entry(state_hash="a", iteration=1))
        assert buf.is_full()
        assert len(buf.window()) == 1
        buf.push(entry(state_hash="b", iteration=2))
        assert buf.window()[0]["state_hash"] == "b"

    def test_window_size_100(self):
        buf = RingBuffer(maxlen=100)
        det = CompositeDetector(threshold=100)
        for i in range(100):
            buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
        assert buf.is_full()
        result = det.evaluate("agent_1", "node_x", buf)
        assert result is not None
        assert result.signal_type == SIGNAL_STAGNATION

    def test_exact_threshold_boundary(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(4):
            buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
        assert not det.evaluate("agent_1", "node_x", buf)
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=4))
        assert det.evaluate("agent_1", "node_x", buf) is not None

    def test_hit_and_run(self):
        """A single detection event followed by buffer reset, clean signal,
        then re-looping should fire a second time."""
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(5):
            buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
        r1 = det.evaluate("agent_1", "node_x", buf)
        assert r1 is not None
        buf.reset()
        for i in range(5):
            buf.push(entry(
                state_hash=f"clean_{i}", tool_sig=f"clean_{i}()", iteration=i
            ))
        r_clean = det.evaluate("agent_1", "node_x", buf)
        assert r_clean is None
        buf.reset()
        for i in range(5):
            buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
        r2 = det.evaluate("agent_1", "node_x", buf)
        assert r2 is not None
