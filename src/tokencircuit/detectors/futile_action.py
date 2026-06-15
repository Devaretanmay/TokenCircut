from ..ring_buffer import RingBuffer


class FutileActionDetector:
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

        tool_sigs = [e["tool_type_signature"] for e in window]

        tool_noop = all(s == "NO_TOOL_CALL" for s in tool_sigs)
        if tool_noop:
            return False

        first_sig = tool_sigs[0]
        if not all(s == first_sig for s in tool_sigs):
            return False

        state_hashes = [e["state_hash"] for e in window]
        first_hash = state_hashes[0]
        state_never_changed = all(h == first_hash for h in state_hashes)

        if state_never_changed:
            return False

        return True
