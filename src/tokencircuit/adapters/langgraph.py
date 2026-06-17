"""
LangGraphPreModelAdapter — hooks into LangGraph's pre_model_hook.

This is the PRIMARY integration path for V7. The adapter:
1. Receives graph state from pre_model_hook.
2. Runs the InterventionEngine pipeline.
3. Returns {"llm_input_messages": [...]} for ephemeral message mutations.
4. For HARD_STOP, returns a LangGraph Command to terminate or raises.

The hook's return value is EPHEMERAL — LangGraph uses llm_input_messages for
the LLM call but never writes them to the checkpointed state.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional, Sequence

from ..engine import InterventionConfig, InterventionEngine
from ..state_schema import InterventionStateSchema, tc_state_reducer
from ..types import InterventionStage

logger = logging.getLogger("tokencircuit.adapters.langgraph")


class LangGraphPreModelAdapter:
    """
    Adapter connecting InterventionEngine to LangGraph's pre_model_hook.

    Usage:
        adapter = LangGraphPreModelAdapter(config=my_config)

        # Option A: Named hook per node
        graph.add_node("agent", call_model,
                       pre_model_hook=adapter.create_hook(node_name="agent"))

        # Option B: Default hook (uses node_name="agent")
        graph.add_node("agent", call_model, pre_model_hook=adapter.hook)
    """

    def __init__(
        self,
        *,
        config: Optional[InterventionConfig] = None,
        engine: Optional[InterventionEngine] = None,
    ) -> None:
        """
        Args:
            config: V7 configuration. If None, uses defaults.
            engine: Pre-built engine. If None, creates one from config.
        """
        self._config = config or InterventionConfig()
        self._engine = engine or InterventionEngine(config=self._config)
        self._default_node_name = "agent"

    async def hook(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        The pre_model_hook callback for LangGraph.

        This is the async version called by LangGraph before each LLM invocation.

        Contract:
        - Input: Full graph state dict.
        - Output: Dict that MAY contain 'llm_input_messages'.
          - PASS: return {} (no override)
          - NUDGE: return {"llm_input_messages": [...original + coaching...]}
          - OVERRIDE: return {"llm_input_messages": [...compacted + directive...]}
          - HARD_STOP: return Command(goto=END) or raise

        Args:
            state: Full graph state dict from LangGraph.

        Returns:
            Dict for LangGraph to process.
        """
        return self._execute_hook(state, node_name=self._default_node_name)

    def create_hook(
        self,
        *,
        node_name: str,
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        """
        Factory creating a pre_model_hook bound to a specific node name.

        Args:
            node_name: The node name to associate with this hook.

        Returns:
            Async callable matching pre_model_hook signature.
        """
        async def _bound_hook(state: dict[str, Any]) -> dict[str, Any]:
            return self._execute_hook(state, node_name=node_name)

        return _bound_hook

    def _record_usage_if_present(self, messages: Sequence[Any]) -> None:
        """Extract token usage from the last AI message and record it."""
        if not messages:
            return

        last_msg = messages[-1]
        # LangChain AIMessage often has usage_metadata
        usage = getattr(last_msg, "usage_metadata", None)
        if not usage and isinstance(last_msg, dict):
            usage = last_msg.get("usage_metadata")

        if usage and isinstance(usage, dict):
            total_tokens = usage.get("total_tokens", 0)
            if total_tokens > 0:
                # We don't necessarily know the model here,
                # but we can try to find it in the message or use default
                model = getattr(last_msg, "model", "unknown")
                if isinstance(model, dict):
                    model = model.get("name", "unknown")
                self._engine.record_usage(str(model), total_tokens)

    def _execute_hook(self, state: dict[str, Any], *, node_name: str) -> dict[str, Any]:
        """
        Core hook execution logic.

        Extracts messages and thread_id from state, runs the engine,
        and translates the decision into LangGraph's expected return format.
        """
        try:
            # Extract messages from state
            messages = state.get("messages", [])
            if not messages:
                return {}

            # Record token usage if present in the last AI message
            self._record_usage_if_present(messages)

            # Extract thread_id from configurable or state
            thread_id = self._extract_thread_id(state)

            # Run the intervention engine
            decision = self._engine.process(
                messages=messages,
                state=state,
                thread_id=thread_id,
                node_name=node_name,
            )

            # Translate decision to LangGraph return format
            return self._decision_to_hook_response(decision, state)

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            # Re-raise TokenCircuitError from HARD_STOP — it's intentional termination
            from ..engine import TokenCircuitError

            if isinstance(exc, TokenCircuitError):
                raise
            # FAIL-SAFE: Never block the LLM call for unexpected errors
            logger.error(
                "LangGraphPreModelAdapter hook failed, returning passthrough: %s",
                exc,
                exc_info=True,
            )
            return {}

    def _decision_to_hook_response(
        self,
        decision: Any,  # InterventionDecision
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate InterventionDecision to pre_model_hook return dict."""
        result: dict[str, Any] = {}

        # Apply state patch to _tc_intervention
        if decision.state_patch:
            existing_tc = state.get("_tc_intervention", {})
            merged = tc_state_reducer(existing_tc, decision.state_patch)
            result["_tc_intervention"] = merged

        # If in audit mode, we log but return PASS behavior (no LLM mutations/stops)
        if self._config.audit_mode:
            if decision.stage > InterventionStage.PASS:
                logger.info(
                    "TokenCircuit AUDIT: node='%s', would have triggered %s. "
                    "Signals: %s",
                    state.get("_tc_node_name", "?"),
                    decision.stage.name,
                    [s.value for s in decision.signals],
                )
            if "_tc_intervention" not in result:
                return {}
            return result

        # Handle based on stage
        if decision.stage == InterventionStage.PASS:
            # No message modification — return state patch only (if any)
            if "_tc_intervention" not in result:
                return {}
            return result

        elif decision.stage == InterventionStage.NUDGE:
            # Append coaching to messages (ephemeral)
            if decision.llm_input_messages:
                result["llm_input_messages"] = decision.llm_input_messages
                logger.info(
                    "TokenCircuit NUDGE: node='%s', signals=%s, sim=%.3f",
                    state.get("_tc_node_name", "?"),
                    [s.value for s in decision.signals],
                    0.0,
                )
            return result

        elif decision.stage == InterventionStage.OVERRIDE:
            # Replace messages with compacted + directive (ephemeral)
            if decision.llm_input_messages:
                result["llm_input_messages"] = decision.llm_input_messages
                logger.warning(
                    "TokenCircuit OVERRIDE: node='%s', signals=%s",
                    state.get("_tc_node_name", "?"),
                    [s.value for s in decision.signals],
                )
            return result

        elif decision.stage == InterventionStage.HARD_STOP:
            # Attempt to use LangGraph Command for clean termination
            logger.error(
                "TokenCircuit HARD_STOP: %s", decision.termination_reason
            )
            try:
                from langgraph.types import Command

                # Return a Command that sends the graph to END
                return Command(goto="__end__", update=result)  # type: ignore[return-value]
            except ImportError:
                # If Command not available, raise to terminate
                from ..engine import TokenCircuitError

                raise TokenCircuitError(
                    decision.termination_reason or "TokenCircuit HARD_STOP",
                )

        return result

    def _extract_thread_id(self, state: dict[str, Any]) -> str:
        """Extract thread_id from state or configurable."""
        # Try _tc_intervention first
        tc_state = state.get("_tc_intervention", {})
        if isinstance(tc_state, dict):
            tid = tc_state.get("thread_id")
            if tid:
                return str(tid)

        # Try configurable (LangGraph pattern)
        configurable = state.get("configurable", {})
        if isinstance(configurable, dict):
            tid = configurable.get("thread_id")
            if tid:
                return str(tid)

        return "default_thread"

    def get_state_schema_annotation(self) -> tuple[type, Callable]:
        """
        Returns (InterventionStateSchema, tc_state_reducer) for graph state definition.

        Usage:
            schema, reducer = adapter.get_state_schema_annotation()
            class AgentState(TypedDict):
                messages: Annotated[list, add_messages]
                _tc_intervention: Annotated[schema, reducer]
        """
        return InterventionStateSchema, tc_state_reducer

    @property
    def engine(self) -> InterventionEngine:
        """Access the underlying InterventionEngine."""
        return self._engine

    @property
    def config(self) -> InterventionConfig:
        """Access the configuration."""
        return self._config
