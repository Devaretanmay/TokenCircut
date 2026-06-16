"""TokenCircuit — Pre-Model Intervention Engine."""

import importlib.metadata
from typing import Any

from .adapters.langgraph import LangGraphPreModelAdapter
from .canonicalizer import MessageCanonicalizer
from .engine import InterventionConfig, InterventionEngine
from .exceptions import TokenCircuitError
from .ledger import ToolTransactionLedger
from .semantic_detector import SemanticStagnationDetector
from .state_schema import (
    InterventionStateSchema,
    default_intervention_state,
    tc_state_reducer,
)
from .types import (
    CanonicalRole,
    InterventionStage,
    SignalType,
    TransactionOutcome,
    TransactionStatus,
)
from .validator import TranscriptValidator

try:
    __version__ = importlib.metadata.version("tokencircuit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"

# CrewAIInterventionAdapter is imported lazily to avoid hard dependency on langchain_core
# Use: from tokencircuit.adapters.crewai import CrewAIInterventionAdapter

__all__ = [
    "__version__",
    "CanonicalRole",
    "InterventionStage",
    "SignalType",
    "TransactionOutcome",
    "TransactionStatus",
    "InterventionStateSchema",
    "default_intervention_state",
    "tc_state_reducer",
    "MessageCanonicalizer",
    "ToolTransactionLedger",
    "TranscriptValidator",
    "SemanticStagnationDetector",
    "InterventionEngine",
    "InterventionConfig",
    "LangGraphPreModelAdapter",
    "CrewAIInterventionAdapter",  # lazy import — access via tokencircuit.adapters.crewai
    "instrument_langgraph",
    "instrument_crewai",
    "TokenCircuitError",
]

def instrument_langgraph(
    builder: Any,
    api_key: str | None = None,
    config: InterventionConfig | None = None,
    nodes_to_wrap: list[str] | None = None,
) -> Any:
    """
    Instrument a LangGraph StateGraph builder before compilation.
    This dynamically wraps the specified LLM nodes using ModelNodeWrapper.

    Args:
        builder: A langgraph StateGraph object.
        api_key: Optional API key.
        config: InterventionConfig for the engine.
        nodes_to_wrap: List of node names that call the LLM. Defaults to ["agent", "llm"].

    Returns:
        The instrumented builder.
    """
    import warnings
    warnings.warn(
        "instrument_langgraph is deprecated and relies on LangGraph private APIs. "
        "Use LangGraphPreModelAdapter as a node or pre_model_hook instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from .adapters.wrapper import ModelNodeWrapper

    config = config or InterventionConfig()
    wrapper = ModelNodeWrapper(config=config, api_key=api_key)
    nodes_to_wrap = nodes_to_wrap or ["agent", "llm", "chatbot", "assistant"]

    # In LangGraph, builder.nodes is a dict of name -> node spec
    if hasattr(builder, "nodes"):
        for name, node in builder.nodes.items():
            if name not in nodes_to_wrap:
                continue

            if hasattr(node, "runnable"):
                # Handle async vs sync functions properly
                orig_func = getattr(node.runnable, "func", None)
                orig_afunc = getattr(node.runnable, "afunc", None)

                wrapped_func = wrapper.wrap(orig_func, node_name=name) if orig_func else None
                wrapped_afunc = wrapper.wrap(orig_afunc, node_name=name) if orig_afunc else None

                from langgraph._internal._runnable import RunnableCallable
                node.runnable = RunnableCallable(
                    func=wrapped_func,
                    afunc=wrapped_afunc,
                    name=getattr(node.runnable, "name", name),
                    tags=getattr(node.runnable, "tags", None),
                    trace=getattr(node.runnable, "trace", True),
                )
            elif callable(node) and not hasattr(node, "invoke") and not hasattr(node, "runnable"):
                builder.nodes[name] = wrapper.wrap(node, node_name=name)

    return builder

def instrument_crewai(
    crew: Any,
    api_key: str | None = None,
    config: InterventionConfig | None = None,
) -> Any:
    """
    Instrument a CrewAI Crew object using the V7 Intervention Engine.
    This hooks into the step_callbacks of all agents in the crew.
    """
    from .adapters.crewai import CrewAIInterventionAdapter

    config = config or InterventionConfig()
    adapter = CrewAIInterventionAdapter(crew, config=config, api_key=api_key)
    return adapter.apply()
