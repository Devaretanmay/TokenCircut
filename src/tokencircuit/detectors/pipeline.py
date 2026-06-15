import logging
from typing import Optional

from ..config import TokenCircuitConfig
from ..ring_buffer import RingBuffer
from ..telemetry import TelemetryEvent, compute_cost_estimate, emit_event_async
from .composite import CompositeDetector, DetectionResult

logger = logging.getLogger("tokencircuit")


class DetectionPipeline:
    def __init__(
        self,
        config: TokenCircuitConfig,
        agent_framework: str,
        api_key: Optional[str] = None,
    ) -> None:
        self._config = config
        self._agent_framework = agent_framework
        self._detector = CompositeDetector(config.window_size)
        self._buffers: dict[str, RingBuffer] = {}
        self._iteration: dict[str, int] = {}
        self._api_key = api_key

    def record_step(
        self,
        agent_id: str,
        node_name: str,
        state_hash: str,
        tool_type_signature: str,
    ) -> Optional[DetectionResult]:
        key = f"{agent_id}:{node_name}"

        if key not in self._buffers:
            self._buffers[key] = RingBuffer(self._config.window_size)
            self._iteration[key] = 0

        self._iteration[key] += 1

        self._buffers[key].push({
            "state_hash": state_hash,
            "tool_type_signature": tool_type_signature,
            "iteration": self._iteration[key],
        })

        result = self._detector.evaluate(agent_id, node_name, self._buffers[key])
        if result is not None:
            self._emit_telemetry(result)
        return result

    def reset(self, agent_id: str, node_name: str) -> None:
        key = f"{agent_id}:{node_name}"
        self._buffers.pop(key, None)
        self._iteration.pop(key, None)
        self._detector.reset(agent_id, node_name)

    def _emit_telemetry(self, result: DetectionResult) -> None:
        if not self._config.telemetry_enabled:
            return
        agency_id = self._config.agency_id
        client_id = self._config.client_id
        if not agency_id or not client_id or not self._api_key:
            return

        model_name = self._config.model_name or "unknown"
        iterations_saved = max(self._config.window_size - result.iteration, 1)
        tokens_saved, cost_saved = compute_cost_estimate(
            model_name, iterations_saved
        )

        event = TelemetryEvent(
            agency_id=agency_id,
            client_id=client_id,
            agent_framework=self._agent_framework,
            signal_type=result.signal_type or "UNKNOWN",
            node_name=result.node_name,
            iterations_at_detection=result.iteration,
            model_name=model_name,
            estimated_tokens_saved=tokens_saved,
            estimated_cost_saved_usd=cost_saved,
        )
        emit_event_async(event, api_key=self._api_key)
