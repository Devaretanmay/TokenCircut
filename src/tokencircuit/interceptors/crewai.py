import logging
from typing import Any, Optional

from ..config import TokenCircuitConfig, load_config
from ..detectors.pipeline import DetectionPipeline
from ..otel.hash_utils import compute_state_hash, extract_tool_type_signature
from ..telemetry import compute_cost_estimate

logger = logging.getLogger("tokencircuit")


class CrewAIInterceptor:
    def __init__(
        self,
        crew: Any,
        config: Optional[TokenCircuitConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._crew = crew
        if config is None:
            config = load_config(api_key)
        self._config = config
        self._pipeline = DetectionPipeline(config, "crewai", api_key=api_key)
        self._registered = False

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
                agent_id = event.task_id
                agent_role = getattr(event, "agent_role", "unknown")
                node_name = agent_role

                state = {
                    "task": event.task_id,
                    "agent_role": agent_role,
                    "description": getattr(event, "description", ""),
                }
                state_hash = compute_state_hash(state)

                tool_call = getattr(event, "tool_call", None)
                if tool_call:
                    tool_sig = extract_tool_type_signature(
                        tool_call if isinstance(tool_call, dict) else None
                    )
                else:
                    tool_sig = "NO_TOOL_CALL"

                try:
                    result = self._pipeline.record_step(
                        agent_id, node_name, state_hash, tool_sig
                    )
                except Exception:
                    logger.exception(
                        "TokenCircuit: error in task_started handler"
                    )
                    return

                if result is not None:
                    self._handle_detection(result, agent_id)

            def on_task_completed(event: TaskCompletedEvent) -> None:
                agent_id = event.task_id
                agent_role = getattr(event, "agent_role", "unknown")
                node_name = agent_role

                output = getattr(event, "output", "")
                if isinstance(output, dict):
                    state_hash = compute_state_hash(output)
                elif isinstance(output, str):
                    import hashlib
                    state_hash = hashlib.sha256(output.encode()).hexdigest()
                else:
                    import hashlib
                    state_hash = hashlib.sha256(
                        str(output).encode()
                    ).hexdigest()

                tool_call = getattr(event, "tool_call", None)
                if tool_call:
                    tool_sig = extract_tool_type_signature(
                        tool_call if isinstance(tool_call, dict) else None
                    )
                else:
                    tool_sig = "NO_TOOL_CALL"

                try:
                    result = self._pipeline.record_step(
                        agent_id, node_name, state_hash, tool_sig
                    )
                except Exception:
                    logger.exception(
                        "TokenCircuit: error in task_completed handler"
                    )
                    return

                if result is not None:
                    self._handle_detection(result, agent_id)

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
        self, result: Any, agent_id: str
    ) -> None:
        model_name = self._config.model_name
        iterations_saved = max(self._config.max_repeats - result.iteration, 1)
        tokens_saved, cost_saved = compute_cost_estimate(
            model_name, max(iterations_saved, 1)
        )

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
