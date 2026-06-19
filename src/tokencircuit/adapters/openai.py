"""OpenAI native adapter."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from ..engine import InterventionConfig, InterventionEngine, TokenCircuitError
from ..state_schema import default_intervention_state, tc_state_reducer
from ..types import InterventionStage

logger = logging.getLogger("tokencircuit.adapters.openai")


class _CompletionsWrapper:
    def __init__(
        self,
        completions: Any,
        engine: InterventionEngine,
        thread_id: str,
        client: Any,
    ) -> None:
        self.completions = completions
        self.engine = engine
        self.thread_id = thread_id
        self.client = client

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.completions, name)
        if name == "create":
            if inspect.iscoroutinefunction(attr):

                async def wrapped_create_async(*args: Any, **kwargs: Any) -> Any:
                    messages = kwargs.get("messages", [])
                    decision = self.engine.process(
                        messages=messages,
                        state={"_tc_intervention": self.client.tc_state},
                        thread_id=self.thread_id,
                        node_name="openai",
                    )
                    self.client.tc_state = tc_state_reducer(
                        self.client.tc_state, decision.state_patch
                    )
                    if decision.stage == InterventionStage.HARD_STOP:
                        raise TokenCircuitError(
                            decision.termination_reason or "TokenCircuit HARD_STOP"
                        )
                    if decision.llm_input_messages:
                        kwargs["messages"] = decision.llm_input_messages
                    return await attr(*args, **kwargs)

                return wrapped_create_async
            else:

                def wrapped_create_sync(*args: Any, **kwargs: Any) -> Any:
                    messages = kwargs.get("messages", [])
                    decision = self.engine.process(
                        messages=messages,
                        state={"_tc_intervention": self.client.tc_state},
                        thread_id=self.thread_id,
                        node_name="openai",
                    )
                    self.client.tc_state = tc_state_reducer(
                        self.client.tc_state, decision.state_patch
                    )
                    if decision.stage == InterventionStage.HARD_STOP:
                        raise TokenCircuitError(
                            decision.termination_reason or "TokenCircuit HARD_STOP"
                        )
                    if decision.llm_input_messages:
                        kwargs["messages"] = decision.llm_input_messages
                    return attr(*args, **kwargs)

                return wrapped_create_sync
        return attr


class _ChatWrapper:
    def __init__(
        self, chat: Any, engine: InterventionEngine, thread_id: str, client: Any
    ) -> None:
        self.chat = chat
        self.engine = engine
        self.thread_id = thread_id
        self.client = client

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.chat, name)
        if name == "completions":
            return _CompletionsWrapper(attr, self.engine, self.thread_id, self.client)
        return attr


class TokenCircuitClient:
    """Wrapper for openai.OpenAI and openai.AsyncOpenAI clients."""

    def __init__(
        self,
        client: Any,
        config: InterventionConfig | None = None,
        thread_id: str = "default_thread",
    ) -> None:
        self.client = client
        self.engine = InterventionEngine(config=config or InterventionConfig())
        self.thread_id = thread_id
        self.tc_state = default_intervention_state()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.client, name)
        if name == "chat":
            return _ChatWrapper(attr, self.engine, self.thread_id, self)
        return attr
