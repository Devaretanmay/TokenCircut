"""Composite detection combining stagnation and futile-action signals."""

from dataclasses import dataclass, field
from typing import Optional

from ..ring_buffer import RingBuffer
from .futile_action import FutileActionDetector
from .state_stagnation import StateStagnationDetector


@dataclass
class DetectionResult:
    signal_type: Optional[str] = None
    iteration: int = 0
    node_name: str = ""
    state_hashes_window: list[str] = field(default_factory=list)
    tool_signatures_window: list[str] = field(default_factory=list)


SIGNAL_STAGNATION = "STATE_STAGNATION"
SIGNAL_FUTILE = "FUTILE_ACTION"


class CompositeDetector:
    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold
        self._stagnation = StateStagnationDetector(threshold)
        self._futile = FutileActionDetector(threshold)
        self._active_alerts: dict[str, str] = {}

    def evaluate(
        self,
        agent_id: str,
        node_name: str,
        buffer: RingBuffer,
    ) -> Optional[DetectionResult]:
        key = f"{agent_id}:{node_name}"
        window = buffer.window()

        stagnation_triggered = self._stagnation.evaluate(buffer)
        futile_triggered = self._futile.evaluate(buffer)

        has_alert = key in self._active_alerts

        if stagnation_triggered:
            if has_alert and self._active_alerts[key] == SIGNAL_STAGNATION:
                return None
            self._active_alerts[key] = SIGNAL_STAGNATION
            return DetectionResult(
                signal_type=SIGNAL_STAGNATION,
                iteration=window[-1]["iteration"],
                node_name=node_name,
                state_hashes_window=[e["state_hash"] for e in window],
                tool_signatures_window=[e["tool_type_signature"] for e in window],
            )

        if futile_triggered:
            if has_alert and self._active_alerts[key] == SIGNAL_FUTILE:
                return None
            self._active_alerts[key] = SIGNAL_FUTILE
            return DetectionResult(
                signal_type=SIGNAL_FUTILE,
                iteration=window[-1]["iteration"],
                node_name=node_name,
                state_hashes_window=[e["state_hash"] for e in window],
                tool_signatures_window=[e["tool_type_signature"] for e in window],
            )

        if has_alert:
            del self._active_alerts[key]

        return None

    def reset(self, agent_id: str, node_name: str) -> None:
        key = f"{agent_id}:{node_name}"
        self._active_alerts.pop(key, None)
