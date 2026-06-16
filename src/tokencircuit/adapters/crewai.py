"""
CrewAIInterventionAdapter — hooks into CrewAI tasks using the V7 InterventionEngine.
"""

import logging
from typing import Any, Optional

from ..engine import InterventionConfig, InterventionEngine
from ..exceptions import TokenCircuitError
from ..state_schema import default_intervention_state, tc_state_reducer

logger = logging.getLogger("tokencircuit.adapters.crewai")


def _lazy_import_langchain_messages():
    """Lazily import langchain_core message types."""
    try:
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            ToolMessage,
        )
        return AIMessage, HumanMessage, ToolMessage, BaseMessage
    except ImportError:
        raise ImportError(
            "CrewAIInterventionAdapter requires langchain-core. "
            "Install it with: pip install langchain-core"
        )


class CrewAIInterventionAdapter:
    """
    Adapter that integrates the V7 InterventionEngine with CrewAI tasks.
    It hooks into the step-by-step execution loop of an agent.
    """

    def __init__(
        self,
        crew: Any,
        config: Optional[InterventionConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._crew = crew
        self._config = config or InterventionConfig()
        self._engine = InterventionEngine(config=self._config)
        # Store state externally since CrewAI doesn't natively expose a mutable state dict like LangGraph
        self._intervention_state: dict[str, Any] = default_intervention_state()
        self._messages_cache: list[Any] = []

    def apply(self) -> Any:
        """
        Apply the intervention adapter to the crew's agents.
        This modifies the agents' step_callbacks to route through TokenCircuit.
        """
        for agent in self._crew.agents:
            # Save the original callback if it exists
            original_callback = getattr(agent, "step_callback", None)

            # Capture agent_role and original_callback in closure properly
            def _make_callback(role: str, orig_cb: Any):
                def wrapped_callback(step_output: Any) -> None:
                    self._intercept_step(step_output, role)
                    if orig_cb:
                        orig_cb(step_output)
                return wrapped_callback

            agent.step_callback = _make_callback(agent.role, original_callback)

        return self._crew

    def _intercept_step(self, step_output: Any, agent_role: str) -> None:
        """
        Intercept a single step in a CrewAI task.
        """
        AIMessage, HumanMessage, ToolMessage, BaseMessage = _lazy_import_langchain_messages()

        # Convert CrewAI step output to Canonical LangChain messages
        # CrewAI step_output is an AgentStep or AgentFinish object

        # 1. Update our message cache with the new step
        if hasattr(step_output, "tool_input"):
            # It's a tool call step
            tool_name = getattr(step_output, "tool", "unknown_tool")
            tool_input = getattr(step_output, "tool_input", {})
            call_id = f"crew_{len(self._messages_cache)}"

            self._messages_cache.append(
                AIMessage(
                    content=getattr(step_output, "log", ""),
                    tool_calls=[{"name": tool_name, "args": tool_input, "id": call_id, "type": "tool_call"}]
                )
            )
            # We assume the result will come next, but for now we simulate
            self._messages_cache.append(
                ToolMessage(
                    content=getattr(step_output, "result", "Unknown result"),
                    name=tool_name,
                    tool_call_id=call_id
                )
            )

        elif hasattr(step_output, "return_values"):
            # Agent finish
            self._messages_cache.append(
                AIMessage(content=getattr(step_output, "log", ""))
            )

        # 2. Run engine
        state = {"_tc_intervention": self._intervention_state, "messages": self._messages_cache}
        decision = self._engine.process(
            messages=self._messages_cache,
            state=state,
            thread_id=f"crew_{id(self._crew)}",
            node_name=agent_role,
        )

        # 3. Apply state patch
        if decision.state_patch:
            self._intervention_state = tc_state_reducer(
                self._intervention_state, decision.state_patch
            )

        # 4. Handle audit mode
        if self._config.audit_mode:
            if decision.stage > 0:
                logger.info(
                    "TokenCircuit AUDIT: CrewAI agent='%s', would have triggered %s. Signals: %s",
                    agent_role,
                    decision.stage.name,
                    [s.value for s in decision.signals],
                )
            return

        # 5. Handle interventions
        if decision.should_terminate:
            raise TokenCircuitError(
                decision.termination_reason or "TokenCircuit HARD_STOP",
                signal_type="HARD_STOP",
                node_name=agent_role,
                iteration=self._intervention_state.get("turn_counter", 0),
            )

        if decision.llm_input_messages:
            # For CrewAI, NUDGE and OVERRIDE require modifying the agent's scratchpad/instructions
            # This requires deep integration with CrewAI's internals (e.g., injecting into agent.instructions)
            # For the scope of this adapter, we will log a warning or raise an exception to pivot.
            logger.warning(
                "TokenCircuit NUDGE/OVERRIDE triggered for CrewAI agent '%s'. "
                "Coaching message: %s",
                agent_role,
                decision.coaching_message
            )
            # In a full implementation, we'd mutate the step_output or agent state here.

