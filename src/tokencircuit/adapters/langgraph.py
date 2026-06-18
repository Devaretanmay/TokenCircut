"""Native LangGraph v1.0.8+ adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages.tool import ToolMessage  # pyright: ignore
from langchain_core.runnables.config import RunnableConfig  # pyright: ignore

# ToolCallWrapper is a runtime type alias, not a class — import only what we
# need as concrete symbols; reconstruct the wrapper type locally so mypy
# doesn't trip on the alias instantiation.
from langgraph.prebuilt.tool_node import (  # pyright: ignore[reportMissingImports]
    ToolCallRequest,
)
from langgraph.types import Command  # pyright: ignore[reportMissingImports]

from ..engine import InterventionEngine, TokenCircuitError
from ..ledger import ToolTransactionLedger
from ..types import InterventionStage

# ToolCallWrapper is: Callable[[ToolCallRequest, Callable[[ToolCallRequest],
#                     ToolMessage | Command]], ToolMessage | Command]
# We declare it as a plain Callable to avoid importing the alias directly.
LedgerProvider = Callable[[str], ToolTransactionLedger]
_ExecuteFn = Callable[[ToolCallRequest], "ToolMessage | Command[Any]"]
_WrapFn = Callable[[ToolCallRequest, _ExecuteFn], "ToolMessage | Command[Any]"]

logger = logging.getLogger("tokencircuit.adapters.langgraph")


def tc_pre_model_hook(
    state: dict[str, Any],
    *,
    engine: InterventionEngine,
    node_name: str = "agent",
) -> dict[str, Any]:
    """Run TokenCircuit from a LangGraph pre_model_hook.

    Returns:
        ``{"llm_input_messages": [...]}`` for Stage 1/2 interventions —
        ephemeral override that does NOT touch the checkpointed ``messages``
        channel.  Empty dict for PASS (no-op).

    Raises:
        TokenCircuitError: propagated on HARD_STOP so LangGraph surfaces it.
    """
    try:
        messages = state.get("messages")
        if not isinstance(messages, list) or not messages:
            return {}

        thread_id = _extract_thread_id(state)
        decision = engine.process(
            messages=messages,
            state=state,
            thread_id=thread_id,
            node_name=node_name,
        )
        engine.set_last_decision(thread_id, decision)

        if engine.config.audit_mode or decision.stage == InterventionStage.PASS:
            return {}
        if decision.stage == InterventionStage.HARD_STOP:
            raise TokenCircuitError(
                decision.termination_reason or "TokenCircuit HARD_STOP"
            )
        # NUDGE / OVERRIDE: return ephemeral llm_input_messages only.
        # Using the "messages" key here would write RemoveMessage ops into the
        # checkpointed state, corrupting the persistent transcript.
        if decision.llm_input_messages:
            return {"llm_input_messages": decision.llm_input_messages}
        return {}
    except (KeyboardInterrupt, SystemExit, TokenCircuitError):
        raise
    except Exception:
        logger.exception("tc_pre_model_hook failed, returning passthrough")
        return {}


def tc_wrap_tool_call(ledger_provider: LedgerProvider) -> _WrapFn:
    """Return a LangGraph wrap_tool_call interceptor that records tool transactions.

    The returned callable matches LangGraph's ``ToolCallWrapper`` signature:
    ``(request: ToolCallRequest, execute: Callable) -> ToolMessage | Command``.

    tool_call_id on ToolMessage is preserved verbatim — we never mutate it.
    """

    def _wrap(
        request: ToolCallRequest,
        execute: _ExecuteFn,
    ) -> ToolMessage | Command[Any]:
        # ToolCallRequest.runtime is a ToolRuntime dataclass; .config is a
        # RunnableConfig TypedDict (i.e. a dict subclass).  Key-access is
        # correct; attribute access would silently return the wrong thing.
        runtime = request.runtime
        config: RunnableConfig | None = (
            getattr(runtime, "config", None) if runtime is not None else None
        )
        configurable: dict[str, Any] = (
            config.get("configurable", {}) if config is not None else {}
        )
        thread_id = str(configurable.get("thread_id", "default_thread"))

        ledger = ledger_provider(thread_id)

        # ToolCall is a TypedDict with mandatory 'name', 'args', 'id' fields.
        tool_call = request.tool_call
        call_id: str = str(tool_call.get("id") or "")
        tool_name: str = str(tool_call.get("name") or "")

        turn = ledger.current_turn + 1
        ledger.register_call(call_id, tool_name, 0, turn)

        result = execute(request)

        if isinstance(result, Command):
            return result

        # result must be ToolMessage — record outcome without touching content
        if isinstance(result, ToolMessage):
            # Preserve tool_call_id exactly; never modify the ToolMessage.
            content = str(result.content or "")
            ledger.register_result(
                result.tool_call_id,
                content[:200],
                len(content),
                0,
                turn,
            )
        return result

    return _wrap


def _extract_thread_id(state: dict[str, Any]) -> str:
    configurable = state.get("configurable", {})
    if isinstance(configurable, dict):
        tid = configurable.get("thread_id")
        if tid:
            return str(tid)

    tc_state = state.get("_tc_intervention", {})
    if isinstance(tc_state, dict):
        tid = tc_state.get("thread_id")
        if tid:
            return str(tid)

    return "default_thread"
