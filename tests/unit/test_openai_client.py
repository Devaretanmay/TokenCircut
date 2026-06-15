import pytest

from tokencircuit.clients.openai import TokenCircuitClient
from tokencircuit.config import TokenCircuitConfig
from tokencircuit.exceptions import TokenCircuitError


class MockMessage:
    def __init__(self, content: str, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, name: str, arguments: str):
        self.function = MockFunction(name, arguments)


class MockChoice:
    def __init__(self, content: str, tool_calls=None):
        self.message = MockMessage(content, tool_calls or [])


class MockResponse:
    def __init__(self, choices):
        self.choices = choices


class MockClient:
    def __init__(self, response_factory):
        self.call_count = 0
        self._factory = response_factory
        self.chat = _ChatProxy(self)

    def create(self, *args, **kwargs):
        self.call_count += 1
        return self._factory()


class _ChatProxy:
    def __init__(self, client):
        self._client = client
        self.completions = _CompletionsProxy(client)


class _CompletionsProxy:
    def __init__(self, client):
        self._client = client

    def create(self, *args, **kwargs):
        return self._client.create(*args, **kwargs)


class TestTokenCircuitClient:
    def test_returns_response_normally(self):
        raw = MockClient(lambda: MockResponse([MockChoice("hello world")]))
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        client = TokenCircuitClient(raw, config=config)
        resp = client.chat.completions.create(model="gpt-4", messages=[])
        assert resp.choices[0].message.content == "hello world"

    def test_raises_on_repeated_identical_responses(self):
        raw = MockClient(lambda: MockResponse([MockChoice("same")]))
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        client = TokenCircuitClient(raw, config=config, session_id="test")
        for _ in range(4):
            client.chat.completions.create(model="gpt-4", messages=[])
        with pytest.raises(TokenCircuitError):
            client.chat.completions.create(model="gpt-4", messages=[])

    def test_calls_underlying_client_each_time(self):
        raw = MockClient(lambda: MockResponse([MockChoice("ok")]))
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        client = TokenCircuitClient(raw, config=config, session_id="count")
        for _ in range(3):
            client.chat.completions.create(model="gpt-4", messages=[])
        assert raw.call_count == 3

    def test_different_sessions_independent(self):
        config = TokenCircuitConfig(max_repeats=5, window_size=5)

        raw_a = MockClient(lambda: MockResponse([MockChoice("repeat")]))
        client_a = TokenCircuitClient(raw_a, config=config, session_id="a")

        raw_b = MockClient(lambda: MockResponse([MockChoice("repeat")]))
        client_b = TokenCircuitClient(raw_b, config=config, session_id="b")

        for _ in range(5):
            try:
                client_a.chat.completions.create(model="gpt-4", messages=[])
            except TokenCircuitError:
                pass

        client_b.chat.completions.create(model="gpt-4", messages=[])
        assert raw_b.call_count == 1

    def test_handles_tool_calls_in_response(self):
        tc = [MockToolCall("search", '{"q": "hello"}')]
        raw = MockClient(lambda: MockResponse([MockChoice("", tool_calls=tc)]))
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        client = TokenCircuitClient(raw, config=config, session_id="tools")
        resp = client.chat.completions.create(model="gpt-4", messages=[])
        assert resp.choices[0].message.content == ""

    def test_passthrough_attribute_access(self):
        raw = MockClient(lambda: MockResponse([MockChoice("test")]))
        raw.custom_attr = 42
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        client = TokenCircuitClient(raw, config=config)
        assert client.custom_attr == 42

    def test_proxy_passthrough(self):
        raw = MockClient(lambda: MockResponse([MockChoice("test")]))
        raw.chat.custom_repr = "hello"
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        client = TokenCircuitClient(raw, config=config)
        assert client.chat.custom_repr == "hello"

    def test_default_config_uses_load_config(self):
        raw = MockClient(lambda: MockResponse([MockChoice("ok")]))
        client = TokenCircuitClient(raw)
        assert client._config is not None
