"""Detector for identical consecutive state hashes."""

from ..ring_buffer import RingBuffer


class StateStagnationDetector:
    def __init__(self, threshold: int = 5) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        self.threshold = threshold

    def evaluate(self, buffer: RingBuffer) -> bool:
        if not buffer.is_full():
            return False

        window = buffer.window()
        if len(window) < self.threshold:
            return False

        state_hashes = [e["state_hash"] for e in window]
        first_hash = state_hashes[0]

        if not all(h == first_hash for h in state_hashes):
            return False

        tool_sigs = [e["tool_type_signature"] for e in window]
        first_sig = tool_sigs[0]
        tool_changed = any(s != first_sig for s in tool_sigs)

        if tool_changed:
            return False

        return True
