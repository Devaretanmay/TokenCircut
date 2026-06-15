from tokencircuit.detectors.composite import (
    SIGNAL_FUTILE,
    SIGNAL_STAGNATION,
    CompositeDetector,
    DetectionResult,
)
from tokencircuit.detectors.futile_action import FutileActionDetector
from tokencircuit.detectors.state_stagnation import StateStagnationDetector
from tokencircuit.exceptions import TokenCircuitError
from tokencircuit.ring_buffer import RingBuffer


def entry(state_hash="a", tool_sig="tool()", iteration=1):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": iteration,
    }


# ── Signal 1: 5× identical state_hash, identical tool_sig → STATE_STAGNATION ──


def test_signal1_identical_state_identical_tool():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(entry(state_hash="same", tool_sig="fetch()", iteration=i))
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is not None
    assert result.signal_type == SIGNAL_STAGNATION
    assert result.iteration == 4
    assert result.node_name == "node_x"


# ── Signal 1 false-positive: tool_sig changes at iter 3 → None ──


def test_signal1_false_positive_tool_change():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    sigs = ["fetch()", "fetch()", "search()", "fetch()", "fetch()"]
    for i, s in enumerate(sigs):
        buf.push(entry(state_hash="same", tool_sig=s, iteration=i))
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is None


# ── Signal 2: 5× identical tool_sig, state_hash varies → FUTILE_ACTION ──


def test_signal2_identical_tool_varying_state():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(
            entry(state_hash=f"state_{i}", tool_sig="search(str)", iteration=i)
        )
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is not None
    assert result.signal_type == SIGNAL_FUTILE
    assert result.iteration == 4


# ── Threshold: 4 identical then change on iter 5 → None ──


def test_threshold_not_met_with_change():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(4):
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
    buf.push(entry(state_hash="different", tool_sig="other()", iteration=4))
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is None


# ── Reset + re-fire: 5× identical, 1 clean, 5× identical → fire again ──


def test_reset_and_refire():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
    r1 = det.evaluate("agent_1", "node_x", buf)
    assert r1 is not None

    buf.reset()
    for i in range(5):
        buf.push(
            entry(
                state_hash=f"clean_{i}",
                tool_sig=f"clean_{i}()",
                iteration=i,
            )
        )
    r_clean = det.evaluate("agent_1", "node_x", buf)
    assert r_clean is None

    buf.reset()
    for i in range(5):
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
    r2 = det.evaluate("agent_1", "node_x", buf)
    assert r2 is not None
    assert r2.signal_type == SIGNAL_STAGNATION


# ── Empty buffer → None ──


def test_empty_buffer():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is None


# ── window_size=3 fires at iter 3 (custom config) ──


def test_custom_window_size_3():
    buf = RingBuffer(maxlen=3)
    det = CompositeDetector(threshold=3)
    for i in range(3):
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is not None
    assert result.signal_type == SIGNAL_STAGNATION
    assert result.iteration == 2


def test_custom_window_size_3_not_triggered_at_2():
    buf = RingBuffer(maxlen=3)
    det = CompositeDetector(threshold=3)
    for i in range(2):
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is None


# ── tool_sig = "NO_TOOL_CALL" × 5, state_hash same → STATE_STAGNATION ──


def test_no_tool_call_stagnation():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(
            entry(
                state_hash="reasoning_loop",
                tool_sig="NO_TOOL_CALL",
                iteration=i,
            )
        )
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is not None
    assert result.signal_type == SIGNAL_STAGNATION


# ── tool_sig = "NO_TOOL_CALL" × 5, state_hash varies → None ──


def test_no_tool_call_varying_state():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(
            entry(
                state_hash=f"state_{i}",
                tool_sig="NO_TOOL_CALL",
                iteration=i,
            )
        )
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is None


# ── Mixed signal: both state stagnation AND tool repeats → STATE_STAGNATION priority ──


def test_mixed_signal_priority():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(
            entry(state_hash="same", tool_sig="same_tool()", iteration=i)
        )
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is not None
    assert result.signal_type == SIGNAL_STAGNATION


# ── Concurrent pushes from two threads ──


def test_concurrent_pushes():
    import threading

    buf = RingBuffer(maxlen=100)
    errors = []

    def pusher(start, count):
        try:
            for i in range(start, start + count):
                buf.push(
                    entry(
                        state_hash=f"hash_{i}",
                        tool_sig="tool()",
                        iteration=i,
                    )
                )
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=pusher, args=(0, 50))
    t2 = threading.Thread(target=pusher, args=(50, 50))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    assert len(buf.window()) == 100
    assert len(buf) == 100
    assert buf.is_full()


# ── Concurrent evaluate from two threads ──


def test_concurrent_evaluate():
    import threading

    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))

    results: list = []
    errors: list = []

    def evaluator(agent_id):
        try:
            r = det.evaluate(agent_id, "node_x", buf)
            if r:
                results.append((agent_id, r.signal_type))
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=evaluator, args=("agent_a",)),
        threading.Thread(target=evaluator, args=("agent_b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 2


# ── StateStagnationDetector standalone false-positive guard ──


def test_stagnation_guard_tool_changes():
    buf = RingBuffer(maxlen=5)
    det = StateStagnationDetector(threshold=5)
    for i in range(5):
        buf.push(
            entry(
                state_hash="same",
                tool_sig=f"tool_{i}()",
                iteration=i,
            )
        )
    assert not det.evaluate(buf)


# ── FutileActionDetector standalone: NO_TOOL_CALL guard ──


def test_futile_action_noop_guard():
    buf = RingBuffer(maxlen=5)
    det = FutileActionDetector(threshold=5)
    for i in range(5):
        buf.push(
            entry(
                state_hash=f"state_{i}",
                tool_sig="NO_TOOL_CALL",
                iteration=i,
            )
        )
    assert not det.evaluate(buf)


# ── Same tool with different capitalization (case sensitivity) ──


def test_tool_signature_case_sensitivity():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(
            entry(
                state_hash=f"state_{i}",
                tool_sig="Search(str)" if i < 3 else "search(str)",
                iteration=i,
            )
        )
    result = det.evaluate("agent_1", "node_x", buf)
    assert result is None


# ── DetectionResult dataclass fields ──


def test_detection_result_fields():
    buf = RingBuffer(maxlen=5)
    det = CompositeDetector(threshold=5)
    for i in range(5):
        buf.push(entry(state_hash="same", tool_sig="tool()", iteration=i))
    result = det.evaluate("agent_1", "node_x", buf)
    assert isinstance(result, DetectionResult)
    assert len(result.state_hashes_window) == 5
    assert len(result.tool_signatures_window) == 5
    assert all(h == "same" for h in result.state_hashes_window)


# ── TokenCircuitError is a RuntimeError ──


def test_token_circuit_error_is_runtime_error():
    assert issubclass(TokenCircuitError, RuntimeError)
