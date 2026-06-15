"""Fixed-size ring buffer for tracking iteration history."""

from collections import deque
from threading import Lock
from typing import Any

ENTRY_SCHEMA = frozenset({"state_hash", "tool_type_signature", "iteration"})


class RingBuffer:
    def __init__(self, maxlen: int = 5) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._maxlen = maxlen
        self._deque: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = Lock()

    def push(self, entry: dict[str, Any]) -> None:
        if not ENTRY_SCHEMA.issubset(entry.keys()):
            raise KeyError(
                f"Entry missing required keys. Requires {sorted(ENTRY_SCHEMA)}, "
                f"got {sorted(entry.keys())}"
            )
        with self._lock:
            self._deque.append(
                {
                    "state_hash": entry["state_hash"],
                    "tool_type_signature": entry["tool_type_signature"],
                    "iteration": entry["iteration"],
                }
            )

    def window(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._deque)

    def is_full(self) -> bool:
        with self._lock:
            return len(self._deque) == self._maxlen

    def reset(self) -> None:
        with self._lock:
            self._deque.clear()

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    def __repr__(self) -> str:
        with self._lock:
            return f"RingBuffer(maxlen={self._maxlen}, items={list(self._deque)})"
