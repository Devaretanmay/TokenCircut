from typing import Any, Optional

from .config import TokenCircuitConfig, load_config
from .exceptions import (
    FutileActionError,
    StateStagnationError,
    TokenCircuitError,
)

__all__ = [
    "TokenCircuitConfig",
    "load_config",
    "instrument_langgraph",
    "instrument_crewai",
    "TokenCircuitError",
    "StateStagnationError",
    "FutileActionError",
]


def instrument_langgraph(
    graph: Any,
    api_key: Optional[str] = None,
    config: Optional[TokenCircuitConfig] = None,
) -> Any:
    from .interceptors.langgraph import LangGraphInterceptor

    return LangGraphInterceptor(graph, config=config, api_key=api_key)


def instrument_crewai(
    crew: Any,
    api_key: Optional[str] = None,
    config: Optional[TokenCircuitConfig] = None,
) -> Any:
    from .interceptors.crewai import CrewAIInterceptor

    interceptor = CrewAIInterceptor(crew, config=config, api_key=api_key)
    return interceptor.apply(crew)
