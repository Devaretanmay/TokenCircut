from typing import Any

import pytest

from tokencircuit import InterventionConfig, TokenCircuitError
from tokencircuit.adapters.openai import TokenCircuitClient


class MockCompletions:
    def create(self, *args: Any, **kwargs: Any) -> str:
        return "sync response"


class MockChat:
    def __init__(self) -> None:
        self.completions = MockCompletions()


class MockClient:
    def __init__(self) -> None:
        self.chat = MockChat()


class MockAsyncCompletions:
    async def create(self, *args: Any, **kwargs: Any) -> str:
        return "async response"


class MockAsyncChat:
    def __init__(self) -> None:
        self.completions = MockAsyncCompletions()


class MockAsyncClient:
    def __init__(self) -> None:
        self.chat = MockAsyncChat()


def test_openai_sync_hard_stop():
    client = TokenCircuitClient(
        MockClient(),
        config=InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3
        ),
    )

    messages = [{"role": "user", "content": "stagnant"}]
    with pytest.raises(TokenCircuitError, match="TokenCircuit HARD_STOP"):
        for _ in range(5):
            client.chat.completions.create(messages=messages)
            messages.append({"role": "assistant", "content": "stagnant response"})


@pytest.mark.asyncio
async def test_openai_async_hard_stop():
    client = TokenCircuitClient(
        MockAsyncClient(),
        config=InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3
        ),
    )

    messages = [{"role": "user", "content": "stagnant"}]
    with pytest.raises(TokenCircuitError, match="TokenCircuit HARD_STOP"):
        for _ in range(5):
            await client.chat.completions.create(messages=messages)
            messages.append({"role": "assistant", "content": "stagnant response"})


def test_openai_sync_pass():
    client = TokenCircuitClient(MockClient())

    for i in range(5):
        messages = [{"role": "user", "content": f"unique {i}"}]
        res = client.chat.completions.create(messages=messages)
        assert res == "sync response"
