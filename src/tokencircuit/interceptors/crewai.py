import logging
from typing import Any, Optional

from ..config import TokenCircuitConfig
from ..detectors.composite import CompositeDetector, DetectionResult
from ..otel.hash_utils import compute_state_hash, extract_tool_type_signature
from ..ring_buffer import RingBuffer
from ..telemetry import TelemetryEvent, compute_cost_estimate, emit_event_async

logger = logging.getLogger("tokencircuit")


class CrewAIInterceptor:
    def __init__(
        self,
        crew: Any,
        config: Optional[TokenCircuitConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._crew = crew
        if config is not None:
            self._config = config
        else:
            from ..config import load_config
            self._config = load_config(api_key)
        self._detector = CompositeDetector(self._config.window_size)
        self._buffers: dict[str, RingBuffer] = {}
        self._iteration: dict[str, int] = {}
        self._api_key = api_key
        self._registered = False

    def _get_buffer(self, agent_id: str, node_name: str) -> RingBuffer:
        key = f"{agent_id}:{node_name}"
        if key not in self._buffers:
            self._buffers[key] = RingBuffer(maxlen=self._config.window_size)
        return self._buffers[key]

    def _increment_iteration(self, agent_id: str, node_name: str) -> int:
        key = f"{agent_id}:{node_name}"
        self._iteration[key] = self._iteration.get(key, 0) + 1
        return self._iteration[key]

    def apply(self, crew: Any) -> Any:
        self._crew = crew

        try:
            from crewai.events import (  # pyright: ignore[reportMissingImports]
                CrewAIEventsBus,
            )
            from crewai.events.task_event import (  # pyright: ignore[reportMissingImports]
                TaskCompletedEvent,
                TaskStartedEvent,
            )

            bus = CrewAIEventsBus()

            def on_task_started(event: TaskStartedEvent) -> None:
                try:
                    agent_id = event.task_id
                    agent_role = getattr(event, "agent_role", "unknown")
                    node_name = agent_role

                    state = {
                        "task": event.task_id,
                        "agent_role": agent_role,
                        "description": getattr(event, "description", ""),
                    }
                    state_hash = compute_state_hash(state)
                    tool_sig = "NO_TOOL_CALL"
                    tool_call = getattr(event, "tool_call", None)
                    if tool_call:
                        tool_sig = extract_tool_type_signature(
                            tool_call if isinstance(tool_call, dict) else None
                        )

                    iteration = self._increment_iteration(agent_id, node_name)
                    buffer = self._get_buffer(agent_id, node_name)
                    buffer.push({
                        "state_hash": state_hash,
                        "tool_type_signature": tool_sig,
                        "iteration": iteration,
                    })

                    result = self._detector.evaluate(
                        agent_id, node_name, buffer
                    )
                    if result is not None:
                        self._handle_detection(result, agent_id)
                except Exception:
                    logger.exception("TokenCircuit: error in task_started handler")

            def on_task_completed(event: TaskCompletedEvent) -> None:
                try:
                    agent_id = event.task_id
                    agent_role = getattr(event, "agent_role", "unknown")
                    node_name = agent_role

                    output = getattr(event, "output", "")
                    if isinstance(output, dict):
                        output_hash = compute_state_hash(output)
                    elif isinstance(output, str):
                        import hashlib
                        output_hash = hashlib.sha256(
                            output.encode()
                        ).hexdigest()
                    else:
                        import hashlib
                        output_hash = hashlib.sha256(
                            str(output).encode()
                        ).hexdigest()

                    tool_call = getattr(event, "tool_call", None)
                    if tool_call:
                        tool_sig = extract_tool_type_signature(
                            tool_call if isinstance(tool_call, dict) else None
                        )
                    else:
                        tool_sig = "NO_TOOL_CALL"

                    iteration = self._increment_iteration(agent_id, node_name)
                    buffer = self._get_buffer(agent_id, node_name)
                    buffer.push({
                        "state_hash": output_hash,
                        "tool_type_signature": tool_sig,
                        "iteration": iteration,
                    })

                    result = self._detector.evaluate(
                        agent_id, node_name, buffer
                    )
                    if result is not None:
                        self._handle_detection(result, agent_id)
                except Exception:
                    logger.exception(
                        "TokenCircuit: error in task_completed handler"
                    )

            bus.subscribe(TaskStartedEvent, on_task_started)
            bus.subscribe(TaskCompletedEvent, on_task_completed)
            self._registered = True
            logger.info("TokenCircuit: registered CrewAI event listeners")
        except ImportError:
            logger.warning(
                "TokenCircuit: crewai not installed; interceptor is no-op"
            )
        except Exception:
            logger.exception(
                "TokenCircuit: failed to register CrewAI listeners"
            )

        return self._crew

    def _handle_detection(
        self, result: DetectionResult, agent_id: str
    ) -> None:
        model_name = self._config.model_name
        iterations_saved = self._config.max_repeats - result.iteration
        tokens_saved, cost_saved = compute_cost_estimate(
            model_name, max(iterations_saved, 1)
        )

        if (
            self._config.telemetry_enabled
            and self._config.agency_id
            and self._config.client_id
            and self._api_key
        ):
            event = TelemetryEvent(
                agency_id=self._config.agency_id,
                client_id=self._config.client_id,
                agent_framework="crewai",
                signal_type=result.signal_type or "UNKNOWN",
                node_name=result.node_name,
                iterations_at_detection=result.iteration,
                model_name=model_name,
                estimated_tokens_saved=tokens_saved,
                estimated_cost_saved_usd=cost_saved,
            )
            emit_event_async(event, api_key=self._api_key)

        msg = (
            f"TokenCircuit [{result.signal_type}]: "
            f"agent='{result.node_name}' at iteration {result.iteration} "
            f"(est. {tokens_saved} tokens saved, ~${cost_saved:.4f})"
        )
        logger.warning(msg)

        try:
            from crewai.exceptions import (  # pyright: ignore[reportMissingImports]
                DelegationLoopException,
            )

            raise DelegationLoopException(msg)
        except ImportError:
            from ..detectors.composite import (
                SIGNAL_FUTILE,
                SIGNAL_STAGNATION,
            )
            from ..exceptions import (
                FutileActionError,
                StateStagnationError,
            )

            if result.signal_type == SIGNAL_STAGNATION:
                raise StateStagnationError(msg)
            elif result.signal_type == SIGNAL_FUTILE:
                raise FutileActionError(msg)
            else:
                raise RuntimeError(msg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._crew, name)
