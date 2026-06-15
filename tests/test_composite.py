
from tokencircuit.detectors.composite import (
    SIGNAL_FUTILE,
    SIGNAL_STAGNATION,
    CompositeDetector,
)
from tokencircuit.ring_buffer import RingBuffer


def make_entry(state_hash="a", tool_sig="tool()", iteration=1):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": iteration,
    }


class TestCompositeDetector:
    def test_detects_stagnation(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(5):
            buf.push(make_entry(state_hash="same", tool_sig="tool()", iteration=i))
        result = det.evaluate("agent_1", "node_1", buf)
        assert result is not None
        assert result.signal_type == SIGNAL_STAGNATION
        assert result.node_name == "node_1"
        assert result.iteration == 4

    def test_detects_futile_action(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(5):
            buf.push(
                make_entry(
                    state_hash=f"state_{i}", tool_sig="search(str)", iteration=i
                )
            )
        result = det.evaluate("agent_1", "node_1", buf)
        assert result is not None
        assert result.signal_type == SIGNAL_FUTILE

    def test_debounce_same_signal(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for _ in range(5):
            buf.push(make_entry(state_hash="same", tool_sig="tool()"))
        result1 = det.evaluate("agent_1", "node_1", buf)
        assert result1 is not None
        result2 = det.evaluate("agent_1", "node_1", buf)
        assert result2 is None

    def test_resets_on_clean_signal(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for _ in range(5):
            buf.push(make_entry(state_hash="same", tool_sig="tool()"))
        result = det.evaluate("agent_1", "node_1", buf)
        assert result is not None

        buf.reset()
        for i in range(5):
            buf.push(
                make_entry(
                    state_hash=f"diff_{i}", tool_sig=f"search_{i}(str)", iteration=i
                )
            )
        result = det.evaluate("agent_1", "node_1", buf)
        assert result is None

        buf.reset()
        for _ in range(5):
            buf.push(make_entry(state_hash="same", tool_sig="tool()"))
        result = det.evaluate("agent_1", "node_1", buf)
        assert result is not None

    def test_different_agents_independent(self):
        buf1 = RingBuffer(maxlen=5)
        buf2 = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(5):
            buf1.push(make_entry(state_hash="same", tool_sig="tool()", iteration=i))
            buf2.push(
                make_entry(
                    state_hash=f"diff_{i}", tool_sig=f"search_{i}(str)", iteration=i
                )
            )
        r1 = det.evaluate("agent_a", "node_x", buf1)
        r2 = det.evaluate("agent_b", "node_y", buf2)
        assert r1 is not None
        assert r2 is None

    def test_not_triggered_below_threshold(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for _ in range(4):
            buf.push(make_entry(state_hash="same", tool_sig="tool()"))
        result = det.evaluate("agent_1", "node_1", buf)
        assert result is None

    def test_reset_agent(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for _ in range(5):
            buf.push(make_entry(state_hash="same", tool_sig="tool()"))
        r1 = det.evaluate("agent_1", "node_1", buf)
        assert r1 is not None

        det.reset("agent_1", "node_1")
        r2 = det.evaluate("agent_1", "node_1", buf)
        assert r2 is not None

    def test_signal_window_data(self):
        buf = RingBuffer(maxlen=5)
        det = CompositeDetector(threshold=5)
        for i in range(5):
            buf.push(
                make_entry(
                    state_hash=f"hash_{i}" if i > 0 else "same",
                    tool_sig="tool()",
                    iteration=i,
                )
            )
        result = det.evaluate("agent_1", "node_1", buf)
        assert result is not None
        assert len(result.state_hashes_window) == 5
        assert len(result.tool_signatures_window) == 5
