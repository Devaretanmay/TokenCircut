from tokencircuit import (
    TokenCircuitConfig,
    TokenCircuitClient,
    load_config,
    instrument_langgraph,
    instrument_crewai,
    TokenCircuitError,
    StateStagnationError,
    FutileActionError,
)
from tokencircuit.config import TokenCircuitConfig as Config


class TestExports:
    def test_token_circuit_config_importable(self):
        cfg = TokenCircuitConfig()
        assert cfg.window_size == 5

    def test_load_config_returns_defaults(self):
        cfg = load_config()
        assert isinstance(cfg, TokenCircuitConfig)

    def test_error_types(self):
        assert issubclass(TokenCircuitError, RuntimeError)
        assert issubclass(StateStagnationError, TokenCircuitError)
        assert issubclass(FutileActionError, TokenCircuitError)

    def test_token_circuit_client_factory(self):
        from tokencircuit.clients.openai import TokenCircuitClient as TC
        assert callable(TC)

    def test_token_circuit_client_wrapper(self):
        from tokencircuit.clients.openai import TokenCircuitClient
        assert callable(TokenCircuitClient)

    def test_instrument_langgraph_importable(self):
        assert callable(instrument_langgraph)
