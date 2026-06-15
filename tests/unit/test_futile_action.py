import pytest

from tokencircuit.detectors.futile_action import FutileActionDetector
from tokencircuit.ring_buffer import RingBuffer


def make_entry(state_hash="a", tool_sig="tool()"):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": 1,
    }


class TestFutileActionDetector:
    def test_detects_futile_action(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for i in range(5):
            buf.push(
                make_entry(state_hash=f"state_{i}", tool_sig="search(str)")
            )
        assert det.evaluate(buf)

    def test_not_triggered_when_all_noop(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for _ in range(5):
            buf.push(make_entry(tool_sig="NO_TOOL_CALL"))
        assert not det.evaluate(buf)

    def test_not_triggered_below_threshold(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for _ in range(4):
            buf.push(make_entry(tool_sig="same()"))
        assert not det.evaluate(buf)

    def test_not_triggered_when_state_never_changes(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for _ in range(5):
            buf.push(make_entry(state_hash="same", tool_sig="search(str)"))
        assert not det.evaluate(buf)

    def test_not_triggered_when_tool_sig_differs(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        sigs = [
            "search(str)", "search(str)", "query(int)",
            "search(str)", "search(str)",
        ]
        for sig in sigs:
            buf.push(make_entry(state_hash="different", tool_sig=sig))
        assert not det.evaluate(buf)

    def test_not_full_buffer(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for _ in range(3):
            buf.push(make_entry(state_hash="a", tool_sig="search(str)"))
        assert not det.evaluate(buf)

    def test_threshold_less_than_two_raises(self):
        with pytest.raises(ValueError, match="threshold must be >= 2"):
            FutileActionDetector(threshold=1)

    def test_state_changing_same_tool(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for i in range(5):
            buf.push(
                make_entry(state_hash=f"s{i}", tool_sig="query_db(str,int)")
            )
        assert det.evaluate(buf)

    def test_mixed_operation_types(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for i in range(5):
            buf.push(make_entry(state_hash=f"s{i}", tool_sig="read(str)"))
        assert det.evaluate(buf)

    def test_all_noop_with_different_state(self):
        buf = RingBuffer(maxlen=5)
        det = FutileActionDetector(threshold=5)
        for i in range(5):
            buf.push(make_entry(state_hash=f"s{i}", tool_sig="NO_TOOL_CALL"))
        assert not det.evaluate(buf)
