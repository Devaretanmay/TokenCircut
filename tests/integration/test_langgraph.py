import pytest

pytest.importorskip("langgraph")

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph

from tokencircuit import instrument_langgraph
from tokencircuit.config import TokenCircuitConfig
from tokencircuit.exceptions import TokenCircuitError


def _looping_node(state):
    messages = list(state.get("messages", []))
    messages.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_url",
                    "args": {"url": "http://example.com"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
    )
    messages.append(ToolMessage(content="Error: timeout", tool_call_id="call_1"))
    return {"messages": messages}


def _should_continue(state) -> Literal["looper", "__end__"]:
    return "looper"


def build_looping_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("looper", _looping_node)
    builder.set_entry_point("looper")
    builder.add_conditional_edges("looper", _should_continue)
    return builder.compile(checkpointer=MemorySaver())


def build_non_looping_graph():
    builder = StateGraph(MessagesState)

    def node_a(state):
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content="Hello from A"))
        return {"messages": messages}

    def node_b(state):
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content="Hello from B"))
        return {"messages": messages}

    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.set_entry_point("node_a")
    builder.add_edge("node_a", "node_b")
    return builder.compile(checkpointer=MemorySaver())


def build_varying_tool_graph():
    builder = StateGraph(MessagesState)
    tool_index = [0]
    tool_names = ["search", "query", "compute", "fetch", "parse"]

    def varying_node(state):
        idx = tool_index[0]
        tool_index[0] += 1
        messages = list(state.get("messages", []))
        tid = f"call_{idx}"
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_names[idx % len(tool_names)],
                        "args": {"input": f"data_{idx}"},
                        "id": tid,
                        "type": "tool_call",
                    }
                ],
            )
        )
        messages.append(ToolMessage(content=f"Result {idx}", tool_call_id=tid))
        return {"messages": messages}

    def router(state) -> Literal["varying_node", "__end__"]:
        if tool_index[0] >= 10:
            return "__end__"
        return "varying_node"

    builder.add_node("varying_node", varying_node)
    builder.set_entry_point("varying_node")
    builder.add_conditional_edges("varying_node", router)
    return builder.compile(checkpointer=MemorySaver())


class TestLangGraphIntegration:
    """
    Integration tests verify:
    - Looping graph raises TokenCircuitError at iteration 5
    - Interrupt message contains signal_type, node_name
    - Non-looping DAG does NOT trigger
    - Varying tool signatures do NOT trigger
    - State is checkpointed and resumable after interruption
    """

    @pytest.mark.asyncio
    async def test_loop_detected_at_iteration_5(self):
        graph = build_looping_graph()
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        safe = instrument_langgraph(graph, config=config)

        with pytest.raises(TokenCircuitError) as exc_info:
            async for _ in safe.astream(
                {"messages": []},
                {"configurable": {"thread_id": "test_loop_1"}},
            ):
                pass

        msg = str(exc_info.value)
        assert "TokenCircuit" in msg
        assert "STATE_STAGNATION" in msg or "FUTILE_ACTION" in msg
        assert "looper" in msg
        assert "iteration" in msg or "5" in msg

    @pytest.mark.asyncio
    async def test_interrupt_message_contains_metadata(self):
        graph = build_looping_graph()
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        safe = instrument_langgraph(graph, config=config)

        with pytest.raises(TokenCircuitError) as exc_info:
            async for _ in safe.astream(
                {"messages": []},
                {"configurable": {"thread_id": "test_loop_2"}},
            ):
                pass

        msg = str(exc_info.value)
        assert "STATE_STAGNATION" in msg
        assert "looper" in msg
        assert "est." in msg

    @pytest.mark.asyncio
    async def test_non_looping_graph_does_not_trigger(self):
        graph = build_non_looping_graph()
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        safe = instrument_langgraph(graph, config=config)

        outputs = []
        try:
            async for out in safe.astream(
                {"messages": []},
                {"configurable": {"thread_id": "test_non_loop"}},
            ):
                outputs.append(out)
        except TokenCircuitError as e:
            pytest.fail(f"Non-looping graph should not raise: {e}")

        assert len(outputs) > 0

    @pytest.mark.asyncio
    async def test_varying_tool_signatures_no_trigger(self):
        graph = build_varying_tool_graph()
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        safe = instrument_langgraph(graph, config=config)

        outputs = []
        try:
            async for out in safe.astream(
                {"messages": []},
                {"configurable": {"thread_id": "test_vary"}},
            ):
                outputs.append(out)
        except TokenCircuitError:
            pytest.fail("Varying tool signatures should not trigger")

        assert len(outputs) > 0

    @pytest.mark.asyncio
    async def test_state_is_resumable(self):
        """After interruption, the checkpointed state is retrievable
        and contains the messages produced so far."""
        graph = build_looping_graph()
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        safe = instrument_langgraph(graph, config=config)

        thread_config = {"configurable": {"thread_id": "test_resume"}}

        with pytest.raises(TokenCircuitError):
            async for _ in safe.astream({"messages": []}, thread_config):
                pass

        state = graph.get_state(thread_config)
        assert state is not None
        messages = state.values.get("messages", [])
        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_fires_at_exactly_iteration_5(self):
        """With window_size=5, the buffer triggers on push 5, yielding 4
        successful steps before interruption."""
        graph = build_looping_graph()
        config = TokenCircuitConfig(max_repeats=5, window_size=5)
        safe = instrument_langgraph(graph, config=config)

        iterations_received = 0
        with pytest.raises(TokenCircuitError):
            async for _ in safe.astream(
                {"messages": []},
                {"configurable": {"thread_id": "test_exact"}},
            ):
                iterations_received += 1

        assert iterations_received == 4, (
            f"Expected 4 steps before interruption (buffer triggers on push 5), "
            f"got {iterations_received}"
        )
