"""High-level instrumentation wrappers for one-line integration."""

import logging
from typing import Any, Optional

from .engine import InterventionConfig

logger = logging.getLogger("tokencircuit.instrumentation")


def instrument_langgraph(
    builder: Any,
    *,
    config: Optional[InterventionConfig] = None,
    node_names: Optional[list[str]] = None,
) -> Any:
    """
    Instrument a LangGraph StateGraph builder with TokenCircuit.

    Injects the LangGraphPreModelAdapter into the specified nodes
    (or all nodes ending in "agent" or "model" by default).

    Args:
        builder: A LangGraph StateGraph builder.
        config: TokenCircuit InterventionConfig.
        node_names: List of specific node names to instrument. If None,
            heuristically instruments nodes likely to be LLM callers.

    Returns:
        The instrumented builder.
    """
    from .adapters.langgraph import LangGraphPreModelAdapter

    adapter = LangGraphPreModelAdapter(config=config)

    # We must operate on the builder before compile() for clean hook injection
    if not hasattr(builder, "nodes"):
        logger.warning("Object does not appear to be a LangGraph builder. Skipping.")
        return builder

    target_nodes = node_names
    if target_nodes is None:
        target_nodes = [
            name for name in builder.nodes.keys()
            if any(k in name.lower() for k in ("agent", "model", "llm"))
        ]

    for name in target_nodes:
        if name in builder.nodes:
            # LangGraph builder.nodes[name] is a Node object.
            # We can't easily modify pre_model_hook post-hoc, but we
            # can wrap the runnable.
            # This ensures the intervention engine runs before the
            # node execution.
            node = builder.nodes[name]

            # Using LangChain Runnable sequence for clean injection
            # Note: This is a 1-line Trojan wrapper.
            original_runnable = node.runnable

            # We use the adapter's hook logic to wrap the runnable
            # Since pre_model_hook return is merged into the input,
            # we can simulate it:
            async def wrapped_runnable(
                input_data: Any, config: Any = None, **kwargs: Any
            ) -> Any:
                # 1. Run the hook
                patch = await adapter.hook(input_data)
                # 2. Merge patch into input_data (ephemeral mutation)
                merged_input = input_data
                if isinstance(input_data, dict):
                    merged_input = {**input_data, **patch}
                # 3. Call original
                return await original_runnable.ainvoke(merged_input, config, **kwargs)

            node.runnable = wrapped_runnable

    return builder


def instrument_crewai(
    crew: Any,
    *,
    config: Optional[InterventionConfig] = None,
) -> Any:
    """
    Instrument a CrewAI Crew with TokenCircuit.

    Injects before_llm_call hooks into the crew's execution flow.

    Args:
        crew: A CrewAI Crew instance.
        config: TokenCircuit InterventionConfig.

    Returns:
        The instrumented crew.
    """
    from .adapters.crewai import CrewAIAdapter

    adapter = CrewAIAdapter(config=config)

    # CrewAI 0.60+ supports execution hooks.
    # We register our hook specifically for this crew.
    if hasattr(crew, "before_llm_call"):
        # If the crew already has a hook system, we append ours.
        # Some versions use a list of hooks, others a single callable.
        crew.before_llm_call(adapter.hook)
    elif hasattr(crew, "agents"):
        # Fallback: patch agents directly if global hooks aren't available
        for agent in crew.agents:
            if hasattr(agent, "step_callback"):
                # We use step_callback as a secondary signal if pre_llm isn't available
                pass

    return crew
