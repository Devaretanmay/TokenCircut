from tokencircuit import (
    FutileActionError,
    StateStagnationError,
    TokenCircuitClient,
    TokenCircuitConfig,
    TokenCircuitError,
    instrument_langgraph,
    load_config,
)


class _MockCompletions:
    def create(self, *args, **kwargs):
        return None


class _MockChat:
    completions = _MockCompletions()


class _MockClient:
    chat = _MockChat()


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

    def test_token_circuit_client_wrapper_called(self):
        wrapped = TokenCircuitClient(_MockClient())
        assert wrapped is not None

    def test_instrument_langgraph_importable(self):
        assert callable(instrument_langgraph)
