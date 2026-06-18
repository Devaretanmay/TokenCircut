"""Integration tests for LangGraph v1.0.8 native hooks — tc_wrap_tool_call + tc_pre_model_hook."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tokencircuit.adapters.langgraph import tc_pre_model_hook, tc_wrap_tool_call
from tokencircuit.engine import InterventionConfig, InterventionEngine
from tokencircuit.state_schema import (
    InterventionStateSchema,
    default_intervention_state,
    tc_state_reducer,
)

# ─────────────────────────────────────────────────────────────────────────────
# Failing tool — returns 403 every time
# ─────────────────────────────────────────────────────────────────────────────


@tool
def _403_tool(query: str) -> str:
    """Tool that always returns 403."""
    return "403 Forbidden: Access denied"


# ─────────────────────────────────────────────────────────────────────────────
# Mock ChatModel — returns tool_calls until it sees the override directive
# ─────────────────────────────────────────────────────────────────────────────


class _MockChatModel:
    """LangGraph-compatible mock LLM. Returns tool_calls until the override directive appears in the input messages, then pivots to text."""

    _tool_name: str
    _text_response: str
    _call_count: int
    _bind_tools_called: bool

    def __init__(
        self, tool_name: str = "error_tool", text_response: str = "Strategy pivot."
    ) -> None:
        self._tool_name = tool_name
        self._text_response = text_response
        self._call_count = 0
        self._bind_tools_called = False

    def bind_tools(self, tools: list, **kwargs: Any) -> _MockChatModel:
        self._bind_tools_called = True
        return self

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self, messages: list, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        self._call_count += 1
        for m in messages:
            if hasattr(m, "content"):
                c = str(m.content) if m.content else ""
            elif isinstance(m, dict):
                c = str(m.get("content", ""))
            else:
                c = ""
            if "SYSTEM DIRECTIVE" in c:
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content=self._text_response))
                    ]
                )
        call_id = f"call_{self._call_count}"
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": self._tool_name,
                                "args": {"query": "x"},
                                "id": call_id,
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )

    @property
    def InputType(self) -> type:
        return list

    @property
    def OutputType(self) -> type:
        return AIMessage

    def invoke(self, input_data: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        return self._generate(input_data).generations[0].message

    def stream(self, input_data: Any, config: Any = None, **kwargs: Any):
        yield self.invoke(input_data, config, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# State schemas
# ─────────────────────────────────────────────────────────────────────────────


class _AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    _tc_intervention: Annotated[InterventionStateSchema, tc_state_reducer]


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder — StateGraph with proper _tc_intervention checkpointing
# ─────────────────────────────────────────────────────────────────────────────


def _build_custom_graph(
    model: _MockChatModel,
    tools: list,
    *,
    engine: InterventionEngine,
    hook_node_name: str = "agent",
) -> Any:
    """
    Build a StateGraph functionally identical to create_react_agent
    but with proper _tc_intervention checkpointing.

    LangGraph's add_node() does not support the pre_model_hook parameter
    (it is only supported by create_react_agent). To work around this,
    tc_pre_model_hook is called directly inside the node function and its
    llm_input_messages are used in place of state["messages"].
    """
    tool_node = ToolNode(
        tools, wrap_tool_call=tc_wrap_tool_call(engine.get_thread_ledger)
    )

    def call_model(state: _AgentState) -> dict[str, Any]:
        hook_out = tc_pre_model_hook(state, engine=engine, node_name=hook_node_name)
        messages = hook_out.get("llm_input_messages") or state["messages"]
        response = model.invoke(messages)
        decision = engine.pop_last_decision("default_thread")
        patch = decision.state_patch if decision else {}
        return {"messages": [response], "_tc_intervention": patch}

    builder = StateGraph(_AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Safety validator — replicates OpenAI's 400 Bad Request check on transcripts
# ─────────────────────────────────────────────────────────────────────────────


class _OpenAITranscriptError(Exception):
    pass


def _assert_valid_transcript(messages: list) -> None:
    """Replicate OpenAI server-side validation: no orphaned call_ids, no dangling tool_calls."""
    produced: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                if isinstance(tc, dict) and tc.get("id"):
                    produced.add(tc["id"])
        if isinstance(m, ToolMessage):
            if m.tool_call_id not in produced:
                raise _OpenAITranscriptError(f"orphan tool_call_id={m.tool_call_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCall403Override:
    """
    403-Forcing integration test.

    A tool that always returns "403 Forbidden" drives the agent into a loop.
    TokenCircuit MUST:
    1. Detect the stagnation via the ledger's error classification.
    2. Escalate to OVERRIDE after consecutive_stagnation_count thresholds.
    3. Strip the failing tool-call transactions from llm_input_messages
       (preventing OpenAI 400 errors from orphan call_ids).
    4. Force the LLM to output a text-based strategy pivot.
    """

    def _make_engine(self) -> InterventionEngine:
        config = InterventionConfig(
            nudge_threshold=2,
            override_threshold=3,
            hard_stop_threshold=5,
            enable_semantic_detection=True,
            window_size=5,
        )
        return InterventionEngine(config=config)

    def _make_initial_state(self) -> _AgentState:
        return {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Do something."),
            ],
            "_tc_intervention": default_intervention_state(),
        }

    # ── custom StateGraph (full state accumulation) ───────────────────────

    def test_override_triggers_on_fourth_iteration(self):
        engine = self._make_engine()
        model = _MockChatModel(tool_name="_403_tool")
        graph = _build_custom_graph(model, [_403_tool], engine=engine)

        result = graph.invoke(self._make_initial_state(), {"recursion_limit": 10})

        msgs = result["messages"]
        tc_state = result.get("_tc_intervention", default_intervention_state())
        final_stage = tc_state.get("current_stage", "pass")

        assert final_stage == "override", f"Expected OVERRIDE, got {final_stage}"

        final_msg = msgs[-1]
        assert isinstance(final_msg, AIMessage), (
            f"Expected final AIMessage, got {type(final_msg).__name__}"
        )
        assert final_msg.content and not getattr(final_msg, "tool_calls", None), (
            f"Expected text pivot, got tool_calls={getattr(final_msg, 'tool_calls', None)}"
        )

        _assert_valid_transcript(msgs)

    def test_strips_failing_transactions_safely(self):
        """Verify the hook's llm_input_messages never contains orphan call_ids that would trigger a 400."""
        engine = self._make_engine()
        model = _MockChatModel(tool_name="_403_tool")
        graph = _build_custom_graph(model, [_403_tool], engine=engine)

        result = graph.invoke(self._make_initial_state(), {"recursion_limit": 10})
        tc_state = result.get("_tc_intervention", {})

        assert tc_state.get("current_stage") in ("override", "nudge")
        assert tc_state.get("total_interventions", 0) >= 1

    def test_override_preserves_valid_context(self):
        """The override should preserve the user's original request while stripping failed tool loops."""
        engine = self._make_engine()
        model = _MockChatModel(
            tool_name="_403_tool",
            text_response="I will use a different strategy — looking up the data via the search API instead.",
        )
        graph = _build_custom_graph(model, [_403_tool], engine=engine)

        result = graph.invoke(self._make_initial_state(), {"recursion_limit": 10})
        msgs = result["messages"]

        assert any("strategy" in (m.content or "").lower() for m in msgs), (
            "The LLM should output a strategy pivot message"
        )
        _assert_valid_transcript(msgs)


class TestErrorAmplification:
    """
    Error amplification tests.

    When tool outputs contain error indicators (401, 500, timeout),
    the ledger classifies them as TRANSIENT_ERROR or PERMANENT_ERROR.
    The hook MUST surface these outcomes and produce NUDGE/OVERRIDE
    decisions faster than a standard text-only stagnation loop.
    """

    @staticmethod
    def _detect_signals(
        messages: list, *, engine: InterventionEngine, thread_id: str = "test"
    ) -> list:
        state = {"messages": messages, "_tc_intervention": default_intervention_state()}
        decision = engine.process(
            messages, state, thread_id=thread_id, node_name="test"
        )
        return decision.signals

    def _build_error_transcript(self, error_content: str, iterations: int) -> list:
        msgs: list = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Do something."),
        ]
        for i in range(iterations):
            cid = f"call_{i}"
            msgs.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "test_tool",
                            "args": {"x": "1"},
                            "id": cid,
                            "type": "tool_call",
                        }
                    ],
                )
            )
            msgs.append(
                ToolMessage(content=error_content, tool_call_id=cid, name="test_tool")
            )
        return msgs

    @pytest.mark.parametrize(
        "error_content,error_label",
        [
            ("401 Unauthorized", "401"),
            ("500 Internal Server Error", "500"),
            ("timeout connecting to upstream", "timeout"),
        ],
    )
    def test_errors_generate_signals(self, error_content: str, error_label: str):
        engine = InterventionEngine(
            config=InterventionConfig(enable_transcript_validation=True)
        )
        thread_id = f"err_{error_label}"

        msgs = self._build_error_transcript(error_content, 3)
        signals = self._detect_signals(msgs, engine=engine, thread_id=thread_id)
        assert len(signals) > 0, (
            f"Error '{error_label}' should produce at least one signal, got none"
        )

    def test_standard_loop_no_signals_without_errors(self):
        """A normal tool result with a success response should not generate error signals."""
        engine = InterventionEngine(
            config=InterventionConfig(enable_transcript_validation=True)
        )

        msgs: list = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Do something."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ok_tool",
                        "args": {"x": "1"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Operation completed successfully.",
                tool_call_id="call_1",
                name="ok_tool",
            ),
        ]
        signals = self._detect_signals(msgs, engine=engine, thread_id="ok")
        assert len(signals) == 0, f"Success should produce zero signals, got {signals}"


class TestWrapToolCallLedgerRecording:
    """Verify tc_wrap_tool_call correctly registers calls and outcomes in the thread-ledger."""

    def test_ledger_records_call_and_result(self):
        engine = InterventionEngine()
        wrapper = tc_wrap_tool_call(engine.get_thread_ledger)

        from langgraph.prebuilt.tool_node import ToolCallRequest

        request = ToolCallRequest(
            tool_call={
                "name": "test_tool",
                "args": {"x": "1"},
                "id": "call_1",
                "type": "tool_call",
            },
            tool=None,
            state={},
            runtime=None,
        )

        result = wrapper(
            request,
            lambda req: ToolMessage(
                content="403 Forbidden", tool_call_id="call_1", name="test_tool"
            ),
        )

        assert isinstance(result, ToolMessage)
        # The wrapper uses runtime config to extract thread_id; with no runtime,
        # it falls back to "default_thread".
        ledger = engine.get_thread_ledger("default_thread")
        txn = ledger._transactions.get("call_1")
        assert txn is not None, "Transaction should exist in ledger"
        assert txn.status.value == "committed"
        assert txn.outcome.value == "permanent_error"

    def test_command_forwarded_unmodified(self):
        engine = InterventionEngine()
        wrapper = tc_wrap_tool_call(engine.get_thread_ledger)

        from langgraph.types import Command

        cmd = Command(goto="other_node")

        from langgraph.prebuilt.tool_node import ToolCallRequest

        request = ToolCallRequest(
            tool_call={
                "name": "test_tool",
                "args": {},
                "id": "call_cmd",
                "type": "tool_call",
            },
            tool=None,
            state={},
            runtime=None,
        )

        result = wrapper(request, lambda req: cmd)
        assert result is cmd
        # Command returns are forwarded unmodified (no result registered).
        # The call IS registered before execution, so the transaction exists
        # with PENDING status (no result was registered).
        txn = engine.get_thread_ledger("default_thread")._transactions.get("call_cmd")
        assert txn is not None
        assert txn.status.value == "pending"
