"""
ModelNodeWrapper — fallback adapter for graphs without pre_model_hook support.

Wraps model-calling node functions to intercept messages before LLM invocation.
Use this for custom graphs, older LangGraph versions, or non-LangGraph frameworks.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from ..engine import InterventionConfig, InterventionEngine
from ..state_schema import tc_state_reducer
from ..types import InterventionStage

logger = logging.getLogger("tokencircuit.adapters.wrapper")


class ModelNodeWrapper:
    """
    Wraps model-calling node functions with V7 intervention logic.

    The wrapper intercepts state["messages"] before the wrapped function executes,
    runs the InterventionEngine, and if needed replaces messages with ephemeral
    coaching/directive content.

    Usage:
        wrapper = ModelNodeWrapper(config=my_config)

        @wrapper.wrap_with(node_name="agent")
        async def call_model(state: AgentState) -> dict:
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}
    """

    def __init__(
        self,
        *,
        config: Optional[InterventionConfig] = None,
        engine: Optional[InterventionEngine] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._config = config or InterventionConfig()
        self._engine = engine or InterventionEngine(config=self._config)
        self._api_key = api_key

    def wrap(
        self,
        func: Callable[..., Any],
        *,
        node_name: Optional[str] = None,
        message_key: str = "messages",
    ) -> Callable[..., Any]:
        """
        Wrap a model node function with intervention logic.

        Args:
            func: Original model node function (sync or async).
            node_name: Override node name (defaults to func.__name__).
            message_key: Key in state dict where messages are stored.

        Returns:
            Wrapped function with the same signature.
        """
        resolved_name = node_name or getattr(func, "__name__", "unknown_node")
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
                modified_state = self._intercept(state, resolved_name, message_key)
                if modified_state is None:
                    # HARD_STOP — do not call the function
                    return self._hard_stop_response(state, message_key)

                result = await func(modified_state, *args, **kwargs)

                # Ensure the intervention state patch is forwarded to the graph
                if isinstance(result, dict) and "_tc_intervention" in modified_state:
                    if "_tc_intervention" not in result:
                        result["_tc_intervention"] = modified_state["_tc_intervention"]

                return result

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
                modified_state = self._intercept(state, resolved_name, message_key)
                if modified_state is None:
                    return self._hard_stop_response(state, message_key)

                result = func(modified_state, *args, **kwargs)

                # Ensure the intervention state patch is forwarded to the graph
                if isinstance(result, dict) and "_tc_intervention" in modified_state:
                    if "_tc_intervention" not in result:
                        result["_tc_intervention"] = modified_state["_tc_intervention"]

                return result

            return sync_wrapper

    def wrap_with(
        self,
        *,
        node_name: str,
        message_key: str = "messages",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator factory for wrapping with explicit configuration.

        Usage:
            @wrapper.wrap_with(node_name="agent")
            async def call_model(state):
                ...
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self.wrap(func, node_name=node_name, message_key=message_key)
        return decorator

    def _intercept(
        self,
        state: dict[str, Any],
        node_name: str,
        message_key: str,
    ) -> Optional[dict[str, Any]]:
        """
        Run intervention and return modified state, or None for HARD_STOP.

        The modified state has messages replaced with ephemeral content if
        the engine decides on NUDGE or OVERRIDE. The original state's messages
        are NOT mutated — a shallow copy is created.
        """
        try:
            messages = state.get(message_key, [])
            if not messages:
                return state

            # Extract thread_id
            thread_id = "default_thread"
            configurable = state.get("configurable", {})
            if isinstance(configurable, dict):
                thread_id = str(configurable.get("thread_id", thread_id))

            # Run engine
            decision = self._engine.process(
                messages=messages,
                state=state,
                thread_id=thread_id,
                node_name=node_name,
            )

            # Act on decision
            modified = dict(state)
            if decision.state_patch:
                existing_tc = state.get("_tc_intervention", {})
                modified["_tc_intervention"] = tc_state_reducer(existing_tc, decision.state_patch)

            # If in audit mode, we log but return PASS behavior (no LLM mutations/stops)
            if self._config.audit_mode:
                if decision.stage > InterventionStage.PASS:
                    logger.info(
                        "TokenCircuit AUDIT: node='%s', would have triggered %s. Signals: %s",
                        node_name,
                        decision.stage.name,
                        [s.value for s in decision.signals],
                    )
                return modified

            if decision.should_terminate:
                # HARD_STOP
                logger.error("TokenCircuit HARD_STOP via ModelNodeWrapper: %s", decision.termination_reason)
                return None

            if decision.stage == InterventionStage.PASS:
                return modified

            if decision.llm_input_messages:
                # NUDGE or OVERRIDE: replace messages in a shallow copy
                modified[message_key] = decision.llm_input_messages
                return modified

            return modified

        except Exception as exc:
            logger.error(
                "ModelNodeWrapper._intercept failed, passing through: %s",
                exc,
                exc_info=True,
            )
            return state

    def _hard_stop_response(self, state: dict[str, Any], message_key: str) -> dict[str, Any]:
        """
        Generate a response for HARD_STOP without calling the LLM.
        Returns a state update that signals termination.
        """
        from ..exceptions import TokenCircuitError

        raise TokenCircuitError(
            "TokenCircuit HARD_STOP: Agent loop terminated by intervention engine.",
            signal_type="HARD_STOP",
            node_name="",
            iteration=0,
        )

    @property
    def engine(self) -> InterventionEngine:
        """Access the underlying InterventionEngine."""
        return self._engine
