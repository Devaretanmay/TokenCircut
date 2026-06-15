from typing import Any

__version__ = "0.1.0"
VERSION = __version__

from .config import TokenCircuitConfig, load_config
from .exceptions import (
    FutileActionError,
    StateStagnationError,
    TokenCircuitError,
)

__all__ = [
    "TokenCircuitConfig",
    "load_config",
    "TokenCircuitClient",
    "instrument_langgraph",
    "instrument_crewai",
    "TokenCircuitError",
    "StateStagnationError",
    "FutileActionError",
]


def instrument_langgraph(
    graph: Any,
    api_key: str | None = None,
    config: TokenCircuitConfig | None = None,
) -> Any:
    from .interceptors.langgraph import LangGraphInterceptor
    return LangGraphInterceptor(graph, config=config, api_key=api_key)


def instrument_crewai(
    crew: Any,
    api_key: str | None = None,
    config: TokenCircuitConfig | None = None,
) -> Any:
    from .interceptors.crewai import CrewAIInterceptor
    return CrewAIInterceptor(crew, config=config, api_key=api_key).apply(crew)


def TokenCircuitClient(
    client: Any,
    api_key: str | None = None,
    config: TokenCircuitConfig | None = None,
) -> Any:
    from .clients.openai import TokenCircuitClient as _TokenCircuitClient
    return _TokenCircuitClient(client, config=config, api_key=api_key)
