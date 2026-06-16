"""
CrewAIAdapter — hooks into CrewAI's agent lifecycle.

Provides the step_callback required to monitor CrewAI agents
and trigger TokenCircuit interventions.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..engine import InterventionConfig, InterventionEngine, TokenCircuitError
from ..types import InterventionStage

logger = logging.getLogger("tokencircuit.adapters.crewai")


class CrewAIAdapter:
    """
    Adapter connecting InterventionEngine to CrewAI's before_llm_call hook.

    This is the PRE-FRONTAL CORTEX for CrewAI. It intercepts messages
    immediately before they are sent to the LLM, enabling NUDGE and OVERRIDE
    without crashing the agent.
    """

    def __init__(
        self,
        *,
        config: Optional[InterventionConfig] = None,
        engine: Optional[InterventionEngine] = None,
    ) -> None:
        self._config = config or InterventionConfig()
        self._engine = engine or InterventionEngine(config=self._config)

    def hook(self, context: Any) -> bool:
        """
        The before_llm_call hook for CrewAI.

        Args:
            context: LLMCallHookContext containing messages and state.

        Returns:
            True to proceed, False to block (though we usually raise TokenCircuitError).
        """
        try:
            # context.messages is mutable
            messages = getattr(context, "messages", [])
            if not messages:
                return True

            # Use thread_id or agent/task ID for isolation
            agent_id = getattr(getattr(context, "agent", None), "role", "unknown_agent")
            task_id = str(id(getattr(context, "task", None)))
            thread_id = f"crewai_{agent_id}_{task_id}"

            # Run engine
            decision = self._engine.process(
                messages=messages,
                # CrewAI state is less structured than LangGraph,
                # but we can extract more later
                state={},
                thread_id=thread_id,
                node_name=agent_id,
            )

            # Handle decision
            if decision.stage == InterventionStage.PASS:
                return True

            if decision.stage == InterventionStage.NUDGE:
                # Inject ephemeral coaching message
                if decision.llm_input_messages:
                    # CrewAI expects dicts in context.messages
                    # We merge the coaching messages (only the system coaching)
                    context.messages.extend(decision.llm_input_messages[-1:])
                return True

            if decision.stage == InterventionStage.OVERRIDE:
                # Compact transcript + inject directive
                if decision.llm_input_messages:
                    context.messages[:] = decision.llm_input_messages
                return True

            if decision.stage == InterventionStage.HARD_STOP:
                reason = decision.termination_reason or "Hard stop triggered"
                raise TokenCircuitError(reason)

            return True

        except Exception as exc:
            if isinstance(exc, TokenCircuitError):
                raise
            logger.error("CrewAIAdapter hook failed: %s", exc, exc_info=True)
            return True
