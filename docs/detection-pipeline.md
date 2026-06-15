# Detection Pipeline

## Overview

TokenCircut's detection pipeline answers one question: *is this agent stuck?* It watches a sliding window of execution steps and looks for two patterns a human would recognize as "going nowhere" — repeating the same action with the same result, or repeating the same action with different results.

## Code Walkthrough

### `ring_buffer.py` — The sliding window

```python
class RingBuffer:
    def __init__(self, maxlen: int = 5) -> None:
        self._maxlen = maxlen
        self._deque: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = Lock()
```

A `deque` with a fixed `maxlen` is the right structure here. When you push past capacity, the oldest entry is silently evicted. This means you never need to manage an index or check bounds — the data structure handles windowing automatically.

The schema for each entry is enforced at push time:

```python
ENTRY_SCHEMA = frozenset({"state_hash", "tool_type_signature", "iteration"})
```

The `Lock` handles concurrent access. Agents often run multiple nodes in parallel, and without the lock, two threads pushing simultaneously could corrupt the deque.

### `state_stagnation.py` — "Same state, same tool"

```python
class StateStagnationDetector:
    def evaluate(self, buffer: RingBuffer) -> bool:
        if not buffer.is_full():
            return False
```

The buffer must be full (all N slots occupied) before detection triggers. This prevents false positives during warm-up.

```python
        state_hashes = [e["state_hash"] for e in window]
        first_hash = state_hashes[0]
        if not all(h == first_hash for h in state_hashes):
            return False
```

Checks that **every** state hash in the window is identical. `all()` short-circuits on the first mismatch, so worst case is O(N) and typical case is much faster.

```python
        tool_sigs = [e["tool_type_signature"] for e in window]
        first_sig = tool_sigs[0]
        tool_changed = any(s != first_sig for s in tool_sigs)
        if tool_changed:
            return False
```

The tool-signature guard prevents firing when the state hash happens to repeat but the agent is calling different tools. This stops false positives from agents that are cycling through a fixed state machine.

### `futile_action.py` — "Same tool, different results"

```python
class FutileActionDetector:
    def evaluate(self, buffer: RingBuffer) -> bool:
```

Mirrors the stagnation detector but flips the conditions. Instead of checking that state hashes match, it checks that they **don't** all match — the agent is doing something different each time, just not productively.

The no-tool guard:

```python
        tool_noop = all(s == "NO_TOOL_CALL" for s in tool_sigs)
        if tool_noop:
            return False
```

An agent thinking without calling tools isn't in a futile action loop — it might just be reasoning. This prevents firing during long chains of LLM-only reasoning steps.

### `composite.py` — The orchestrator

```python
class CompositeDetector:
    def evaluate(self, agent_id, node_name, buffer):
        stagnation_triggered = self._stagnation.evaluate(buffer)
        futile_triggered = self._futile.evaluate(buffer)
```

Evaluates both detectors on every push. If both fire simultaneously, stagnation takes priority (it's the more severe condition — state isn't moving at all).

The debounce logic:

```python
        if stagnation_triggered:
            if has_alert and self._active_alerts[key] == SIGNAL_STAGNATION:
                return None  # already alerted for this signal
```

Prevents re-firing for the same signal on the same agent+node pair until the buffer is reset. Without this, every subsequent push would trigger another alert after the first.

## Concepts Explained

### Ring Buffer / Sliding Window

**What**: A fixed-size buffer that automatically discards the oldest entry when new entries are pushed.

**Why**: Agent behavior is a stream, not a batch. A ring buffer lets you evaluate the last N steps without storing the full history. The O(1) push and automatic eviction mean you never need to manage window boundaries.

**When**: Any time you're monitoring a temporal pattern where only recent history matters. Token counters, rate limiters, and trend detectors all benefit from this pattern.

**Alternatives**: A list with manual index management, or storing timestamps and filtering. Both are more complex and error-prone.

### Composite Pattern

**What**: Multiple detectors evaluated together, with a priority hierarchy for conflicting results.

**Why**: No single signal perfectly identifies a stuck agent. By composing two detectors with different criteria, you catch more loop types while reducing false positives — each detector acts as a guard for the other's weak spots.

**When**: Any classification problem where multiple overlapping signals exist but none is definitive on its own. Spam detection, anomaly detection, fraud scoring.

### Debounce

**What**: Suppressing duplicate alerts for the same condition within a window.

**Why**: Without debounce, a detector that fires at iteration 5 would fire again at iteration 6, 7, 8... since the buffer still contains the same pattern. The interceptor should raise once, not flood.

**When**: Any alerting system where the condition persists beyond the detection point. Circuit breakers, monitoring alerts, error rate thresholds.

### Thread Safety (Lock)

**What**: A mutex protecting shared state from concurrent access.

**Why**: LangGraph and CrewAI can execute nodes in parallel. Without the lock, two threads pushing to the same buffer simultaneously could interleave operations and corrupt the deque.

**When**: Any shared mutable state in a concurrent system. Queues, counters, caches accessed from multiple threads.

## Learning Resources

- [collections.deque documentation](https://docs.python.org/3/library/collections.html#collections.deque) — Python's ring buffer implementation
- [When to use a ring buffer](https://www.youtube.com/watch?v=2lqR1J2Fk7c) — Visual explanation of sliding windows
- [Composite pattern (refactoring.guru)](https://refactoring.guru/design-patterns/composite) — Clean explanation with examples
- [Python threading.Lock guide](https://docs.python.org/3/library/threading.html#lock-objects) — Official docs on thread coordination
- [Debouncing explained](https://css-tricks.com/debouncing-throttling-explained-examples/) — From UI events but applies to any alert system

## Related Code

- `src/tokencircuit/ring_buffer.py`
- `src/tokencircuit/detectors/state_stagnation.py`
- `src/tokencircuit/detectors/futile_action.py`
- `src/tokencircuit/detectors/composite.py`
