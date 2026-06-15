import pytest

from tokencircuit.detectors.state_stagnation import StateStagnationDetector
from tokencircuit.ring_buffer import RingBuffer


def make_entry(state_hash="a", tool_sig="tool()"):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": 1,
    }


class TestStateStagnationDetector:
    def test_detects_stagnation(self):
        buf = RingBuffer(maxlen=5)
        det = StateStagnationDetector(threshold=5)
        for _ in range(5):
            buf.push(make_entry(state_hash="same"))
        assert det.evaluate(buf)

    def test_not_triggered_below_threshold(self):
        buf = RingBuffer(maxlen=5)
        det = StateStagnationDetector(threshold=5)
        for _ in range(4):
            buf.push(make_entry(state_hash="same"))
        assert not det.evaluate(buf)

    def test_not_triggered_when_hash_differs(self):
        buf = RingBuffer(maxlen=5)
        det = StateStagnationDetector(threshold=5)
        for i in range(5):
            buf.push(make_entry(state_hash=f"hash_{i}"))
        assert not det.evaluate(buf)

    def test_false_positive_guard_tool_changed(self):
        buf = RingBuffer(maxlen=5)
        det = StateStagnationDetector(threshold=5)
        sigs = [
            "search(str)", "search(str)", "query(int)",
            "search(str)", "search(str)",
        ]
        for sig in sigs:
            buf.push(make_entry(state_hash="same", tool_sig=sig))
        assert not det.evaluate(buf)

    def test_not_full_buffer(self):
        buf = RingBuffer(maxlen=5)
        det = StateStagnationDetector(threshold=5)
        for _ in range(3):
            buf.push(make_entry(state_hash="same"))
        assert not det.evaluate(buf)

    def test_threshold_different_from_maxlen(self):
        buf = RingBuffer(maxlen=10)
        det = StateStagnationDetector(threshold=5)
        for _ in range(10):
            buf.push(make_entry(state_hash="same"))
        assert det.evaluate(buf)

    def test_threshold_less_than_two_raises(self):
        with pytest.raises(ValueError, match="threshold must be >= 2"):
            StateStagnationDetector(threshold=1)

    def test_all_same_hash_all_same_tool(self):
        buf = RingBuffer(maxlen=5)
        det = StateStagnationDetector(threshold=5)
        for _ in range(5):
            buf.push(make_entry(state_hash="x", tool_sig="same()"))
        assert det.evaluate(buf)

    def test_mixed_state_hashes_different_tools(self):
        buf = RingBuffer(maxlen=5)
        det = StateStagnationDetector(threshold=5)
        hashes = ["a", "a", "b", "a", "a"]
        sigs = ["t1()", "t1()", "t2()", "t1()", "t1()"]
        for h, s in zip(hashes, sigs):
            buf.push(make_entry(state_hash=h, tool_sig=s))
        assert not det.evaluate(buf)
